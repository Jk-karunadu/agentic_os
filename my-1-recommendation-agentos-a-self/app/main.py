from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, status, BackgroundTasks

from app.config import settings
from app.local_llm import LocalLLMUnavailable, OllamaClient
from app.memory import MemoryStore
from app.models import (
    EntityRelation, ExecutionBudget, FailureRecord, MemoryItem,
    MemoryQuery, MemoryScoreBreakdown, MemoryType, NodeCompletion,
    NodeStatus, NodeType, OrchestratedRunResult, PlanNode, NodeEvent, ReflectionLesson,
    ReplanRequest, RunTrace, Task, TaskCreate, TaskPlan, TaskResult,
    TaskStatus, VerificationResult, VerifiedRunResult, VerifyRequest,
    WorkerContext, WorkerResult,
)
from app.reflection import build_failure_record, deterministic_lesson, llm_reflect
from app.scheduler import PlanScheduler, PlanTransitionError
from app.store import TaskStore
from app.verification import deterministic_verify
from app.workers import WorkerDispatcher

store = TaskStore(settings.database_path)
memory_store = MemoryStore(settings.database_path)
llm = OllamaClient(settings)

# Register tools
from app.tools.base import ToolRegistry
from app.tools.memory import MemoryQueryTool
from app.tools.search import WebSearchTool
from app.tools.url_reader import URLReaderTool
from app.tools.python_executor import PythonExecutorTool
from app.tools.file_reader import FileReaderTool

tool_registry = ToolRegistry()
tool_registry.register(MemoryQueryTool(memory_store))
tool_registry.register(WebSearchTool())
tool_registry.register(URLReaderTool())
tool_registry.register(PythonExecutorTool())
tool_registry.register(FileReaderTool())

dispatcher = WorkerDispatcher(llm, tool_registry)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version="0.3.0", lifespan=lifespan)


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, bool]:
    return {"api": True, "ollama": llm.healthcheck()}


# ── Tasks ────────────────────────────────────────────────────────────

@app.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(request: TaskCreate) -> Task:
    return store.create(Task(objective=request.objective, evidence=request.evidence))


@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: UUID) -> Task:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/tasks/{task_id}/traces", response_model=list[RunTrace])
def get_task_traces(task_id: UUID) -> list[RunTrace]:
    if store.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return store.list_traces(task_id)


# ── Planning ─────────────────────────────────────────────────────────

@app.post("/tasks/{task_id}/plan", response_model=TaskPlan)
def create_plan(task_id: UUID, budget: ExecutionBudget = ExecutionBudget()) -> TaskPlan:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        plan = llm.create_plan(str(task_id), task.objective, budget)
        return store.save_plan(PlanScheduler.refresh_ready(plan))
    except LocalLLMUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/tasks/{task_id}/plan", response_model=TaskPlan)
def get_latest_plan(task_id: UUID) -> TaskPlan:
    if store.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    plan = store.latest_plan(task_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan exists for this task")
    return plan


@app.get("/tasks/{task_id}/plan/ready", response_model=list[PlanNode])
def get_ready_nodes(task_id: UUID) -> list[PlanNode]:
    plan = store.latest_plan(task_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan exists for this task")
    refreshed = PlanScheduler.refresh_ready(plan)
    if refreshed != plan:
        store.update_plan(refreshed)
    return [node for node in refreshed.nodes if node.status == NodeStatus.READY]


@app.post("/tasks/{task_id}/plan/nodes/{node_id}/start", response_model=TaskPlan)
def start_plan_node(task_id: UUID, node_id: str) -> TaskPlan:
    plan = store.latest_plan(task_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan exists for this task")
    try:
        return store.update_plan(PlanScheduler.start(plan, node_id))
    except PlanTransitionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/tasks/{task_id}/plan/nodes/{node_id}/complete", response_model=TaskPlan)
def complete_plan_node(task_id: UUID, node_id: str, completion: NodeCompletion) -> TaskPlan:
    plan = store.latest_plan(task_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No plan exists for this task")
    try:
        return store.update_plan(PlanScheduler.complete(plan, node_id, completion.succeeded))
    except PlanTransitionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/tasks/{task_id}/replan", response_model=TaskPlan)
def replan(task_id: UUID, request: ReplanRequest) -> TaskPlan:
    task = store.get(task_id)
    previous_plan = store.latest_plan(task_id)
    if task is None or previous_plan is None:
        raise HTTPException(status_code=404, detail="Task or existing plan not found")
    if request.failed_node_id not in {node.id for node in previous_plan.nodes}:
        raise HTTPException(status_code=422, detail="Failed node is not part of the latest plan")
    try:
        plan = llm.create_plan(
            str(task_id), task.objective, previous_plan.budget, previous_plan, request.failure_reason
        )
        plan.version = previous_plan.version + 1
        store.save_plan(plan)
        return plan
    except LocalLLMUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


# ── Verification ─────────────────────────────────────────────────────

@app.post("/tasks/{task_id}/verify", response_model=VerificationResult)
def verify_task(task_id: UUID, request: VerifyRequest) -> VerificationResult:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    result = deterministic_verify(str(task_id), request.answer, task.evidence)
    if result.passed and task.evidence:
        try:
            issues = llm.verify_evidence(
                str(task_id), request.answer, [item.model_dump(mode="json") for item in task.evidence]
            )
            result = VerificationResult(
                task_id=task_id, answer=request.answer, passed=not issues, issues=issues, verifier="local_llm"
            )
        except LocalLLMUnavailable as error:
            result = VerificationResult(
                task_id=task_id, answer=request.answer, passed=False,
                issues=result.issues, verifier=f"deterministic_fallback: {error}",
            )
    return store.add_verification(result)


# ── Reflection ───────────────────────────────────────────────────────

@app.post("/tasks/{task_id}/reflect", response_model=ReflectionLesson)
def reflect_on_failure(task_id: UUID) -> ReflectionLesson:
    if store.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    verification = store.latest_verification(task_id)
    if verification is None:
        raise HTTPException(status_code=409, detail="Verify an answer before reflecting")
    if verification.passed:
        raise HTTPException(status_code=409, detail="Reflections are only stored after verification failures")
    return store.add_lesson(deterministic_lesson(str(task_id), verification))


# ── Failure database ─────────────────────────────────────────────────

@app.get("/failures", response_model=list[FailureRecord])
def list_failures(task_id: UUID | None = None) -> list[FailureRecord]:
    return store.list_failures(task_id)


@app.get("/failures/by-type/{error_type}", response_model=list[FailureRecord])
def get_failures_by_type(error_type: str) -> list[FailureRecord]:
    return store.get_failures_by_type(error_type)


# ── Simple run (Phase 1 baseline) ───────────────────────────────────

@app.post("/tasks/{task_id}/run", response_model=TaskResult)
def run_task(task_id: UUID) -> TaskResult:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    store.set_status(task_id, TaskStatus.RUNNING)
    try:
        result = llm.research_brief(
            str(task.id), task.objective, [item.model_dump(mode="json") for item in task.evidence]
        )
    except LocalLLMUnavailable as error:
        store.set_status(task_id, TaskStatus.FAILED)
        store.add_trace(
            RunTrace(
                task_id=task_id,
                model=settings.ollama_model,
                status=TaskStatus.FAILED,
                error=str(error),
            )
        )
        raise HTTPException(status_code=503, detail=str(error)) from error
    store.set_status(task_id, TaskStatus.COMPLETED)
    store.add_trace(result.trace)
    return result


# ── Verified run (Phase 3: run → verify → reflect → retry once) ────

@app.post("/tasks/{task_id}/run-verified", response_model=VerifiedRunResult)
def run_verified(task_id: UUID) -> VerifiedRunResult:
    """Execute, verify, and — on failure — reflect and retry once.

    Flow:
    1. Run the task with the local LLM.
    2. Deterministically verify the answer.
    3. If passed → return.
    4. If failed → reflect → store lesson + failure → retry with lessons → return.
    """
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    evidence_dicts = [item.model_dump(mode="json") for item in task.evidence]

    # ── First run ────────────────────────────────────────────────
    store.set_status(task_id, TaskStatus.RUNNING)
    try:
        first_result = llm.research_brief(str(task.id), task.objective, evidence_dicts)
    except LocalLLMUnavailable as error:
        store.set_status(task_id, TaskStatus.FAILED)
        raise HTTPException(status_code=503, detail=str(error)) from error
    store.add_trace(first_result.trace)

    # ── Verify ───────────────────────────────────────────────────
    verification = deterministic_verify(str(task_id), first_result.answer, task.evidence)
    store.add_verification(verification)

    if verification.passed:
        store.set_status(task_id, TaskStatus.COMPLETED)
        _auto_store_episodic(task, first_result.answer, passed=True)
        return VerifiedRunResult(
            task_id=task_id, answer=first_result.answer, verification=verification,
        )

    # ── Reflect ──────────────────────────────────────────────────
    lesson = deterministic_lesson(str(task_id), verification)
    store.add_lesson(lesson)
    failure = build_failure_record(
        str(task_id), verification, lesson, model=settings.ollama_model,
    )
    failure = store.add_failure(failure)

    # Auto-store procedural memory from the lesson.
    _auto_store_procedural(task, lesson)

    # ── Retry once with lessons ──────────────────────────────────
    # Gather all prior lessons for this task.
    all_lessons = store.list_lessons(task_id)
    lesson_texts = [l.recommended_action for l in all_lessons]
    try:
        retry_result = llm.research_brief_with_lessons(
            str(task.id), task.objective, evidence_dicts, lesson_texts,
        )
    except LocalLLMUnavailable:
        store.set_status(task_id, TaskStatus.FAILED)
        return VerifiedRunResult(
            task_id=task_id, answer=first_result.answer,
            verification=verification, lesson=lesson, failure=failure,
            retried=False,
        )
    store.add_trace(retry_result.trace)

    retry_verification = deterministic_verify(str(task_id), retry_result.answer, task.evidence)
    store.add_verification(retry_verification)

    # Update failure record with repair outcome
    store.update_failure_repair(failure.id, retry_verification.passed)

    final_status = TaskStatus.COMPLETED if retry_verification.passed else TaskStatus.FAILED
    store.set_status(task_id, final_status)

    # Auto-store episodic memory from the final result.
    final_answer = retry_result.answer if retry_verification.passed else first_result.answer
    _auto_store_episodic(task, final_answer, passed=retry_verification.passed)

    return VerifiedRunResult(
        task_id=task_id, answer=first_result.answer,
        verification=verification, lesson=lesson, failure=failure,
        retried=True, retry_answer=retry_result.answer,
        retry_verification=retry_verification,
    )


# ── Memory endpoints ─────────────────────────────────────────────────

@app.post("/memory", response_model=MemoryItem, status_code=status.HTTP_201_CREATED)
def store_memory(item: MemoryItem) -> MemoryItem:
    return memory_store.store_memory(item)


@app.post("/memory/query", response_model=list[MemoryItem])
def query_memory(query: MemoryQuery) -> list[MemoryItem]:
    return memory_store.retrieve(query, query_type=query.memory_type)


@app.get("/memory/{memory_id}", response_model=MemoryItem)
def get_memory(memory_id: str) -> MemoryItem:
    item = memory_store.get_memory(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return item


@app.post("/memory/explain", response_model=list[MemoryScoreBreakdown])
def explain_memory_retrieval(query: MemoryQuery) -> list[MemoryScoreBreakdown]:
    return memory_store.explain_retrieval(query, query_type=query.memory_type)


@app.post("/memory/relations", response_model=EntityRelation, status_code=status.HTTP_201_CREATED)
def store_relation(rel: EntityRelation) -> EntityRelation:
    return memory_store.store_relation(rel)


@app.get("/memory/relations/query", response_model=list[EntityRelation])
def query_relations(
    subject: str | None = None, predicate: str | None = None, object_: str | None = None,
) -> list[EntityRelation]:
    return memory_store.query_relations(subject=subject, predicate=predicate, object_=object_)


# ── Auto-memory helpers ──────────────────────────────────────────────

def _auto_store_episodic(task: Task, answer: str, passed: bool) -> None:
    """Store an episodic memory after a task completes."""
    summary = f"Task: {task.objective[:200]}\nOutcome: {'passed' if passed else 'failed'}\nAnswer excerpt: {answer[:300]}"
    memory_store.store_memory(MemoryItem(
        type=MemoryType.EPISODIC,
        text=summary,
        importance=0.7 if passed else 0.5,
        source_task_id=str(task.id),
        success_score=1.0 if passed else 0.0,
    ))


def _auto_store_procedural(task: Task, lesson: ReflectionLesson) -> None:
    """Store a procedural memory from a reflection lesson."""
    text = f"Lesson from task '{task.objective[:150]}': {lesson.lesson} Action: {lesson.recommended_action}"
    memory_store.store_memory(MemoryItem(
        type=MemoryType.PROCEDURAL,
        text=text,
        importance=lesson.confidence,
        source_task_id=str(task.id),
        success_score=0.0,
    ))


# ── Phase 5 Orchestrated run ─────────────────────────────────────────

_ORCHESTRATION_RESULTS: dict[UUID, OrchestratedRunResult | Exception] = {}

def _do_run_orchestrated(task_id: UUID) -> None:
    """Execute the task plan node-by-node using specialized multi-agent workers in the background."""
    try:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        plan = store.latest_plan(task_id)
        if plan is None:
            # Create a basic plan if none exists
            budget = ExecutionBudget()
            try:
                plan = llm.create_plan(str(task_id), task.objective, budget)
                plan = store.save_plan(PlanScheduler.refresh_ready(plan))
            except LocalLLMUnavailable as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

        store.set_status(task_id, TaskStatus.RUNNING)

        context = WorkerContext(
            task_id=task_id,
            objective=task.objective,
            evidence=task.evidence,
            previous_results={},
            lessons=[l.recommended_action for l in store.list_lessons()],
            budget=plan.budget,
        )

        node_results = {}
        total_tokens = 0
        total_duration = 0
        final_answer = None
        verification_passed = False
        verification_output = None

        # Continuously execute ready nodes until plan completes or fails
        while True:
            plan = PlanScheduler.refresh_ready(plan)
            ready_nodes = [n for n in plan.nodes if n.status == NodeStatus.READY]
        
            if not ready_nodes:
                break

            for node in ready_nodes:
                # Start node
                try:
                    plan = PlanScheduler.start(plan, node.id)
                    store.update_plan(plan)
                except PlanTransitionError:
                    continue

                # Execute
                def log_event(role: str, content: str, name: str | None = None):
                    for n in plan.nodes:
                        if n.id == node.id:
                            n.events.append(NodeEvent(role=role, content=content, name=name))
                            store.update_plan(plan)
                            break

                result = dispatcher.execute_node(node, context, log_event)
            
                node_results[node.id] = result
                total_tokens += result.tokens_used
                total_duration += result.duration_ms

                if result.error:
                    # Complete with failure
                    plan = PlanScheduler.complete(plan, node.id, succeeded=False)
                    store.update_plan(plan)
                    store.set_status(task_id, TaskStatus.FAILED)
                    _ORCHESTRATION_RESULTS[task_id] = OrchestratedRunResult(
                        task_id=task_id,
                        final_answer=final_answer,
                        verification_passed=False,
                        verification_output=result.output if node.type == NodeType.VERIFY else verification_output,
                        node_results=node_results,
                        total_tokens=total_tokens,
                        total_duration_ms=total_duration,
                    )
                    return
            
                # Store result in context
                context.previous_results[node.id] = result.output

                if node.type == NodeType.DRAFT:
                    final_answer = result.output
                elif node.type == NodeType.VERIFY:
                    verification_passed = "failed" not in result.output.lower()
                    verification_output = result.output

                    if not verification_passed and context.budget.max_retries > 0:
                        context.budget.max_retries -= 1
                        # Reload lessons so the new lesson is in context
                        context.lessons = [l.recommended_action for l in store.list_lessons()]
                        
                        # Find the draft node this verify node depended on
                        draft_node_id = node.depends_on[0] if node.depends_on else None
                        draft_node = next((n for n in plan.nodes if n.id == draft_node_id), None) if draft_node_id else None
                        
                        if draft_node:
                            retry_idx = context.budget.max_retries
                            new_draft_id = f"retry_draft_{retry_idx}"
                            new_verify_id = f"retry_verify_{retry_idx}"
                            
                            log_event("system", f"Verification failed. Initiating self-improvement retry loop ({new_draft_id})...")
                            
                            # Add the original draft and the failed verify node to dependencies so the LLM can see what it wrote and what failed
                            retry_deps = draft_node.depends_on.copy() + [draft_node.id, node.id]
                            
                            plan.nodes.append(PlanNode(
                                id=new_draft_id,
                                type=NodeType.DRAFT,
                                objective="Rewrite the 'draft' output to fix the verification issues from 'verify'. Do NOT repeat the errors.",
                                depends_on=retry_deps,
                                status=NodeStatus.PENDING
                            ))
                            plan.nodes.append(PlanNode(
                                id=new_verify_id,
                                type=NodeType.VERIFY,
                                objective="Verify revised draft",
                                depends_on=[new_draft_id],
                                status=NodeStatus.PENDING
                            ))
                            # Note: plan will be updated in DB at the end of the while loop, or we can just update it here:
                            store.update_plan(plan)

                # Complete with success
                plan = PlanScheduler.complete(plan, node.id, succeeded=True)

        store.update_plan(plan)
    
        # Check if plan fully completed
        is_completed = all(n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED) for n in plan.nodes)
        if is_completed:
            store.set_status(task_id, TaskStatus.COMPLETED)
            if final_answer:
                _auto_store_episodic(task, final_answer, passed=verification_passed)
        else:
            store.set_status(task_id, TaskStatus.FAILED)

        _ORCHESTRATION_RESULTS[task_id] = OrchestratedRunResult(
            task_id=task_id,
            final_answer=final_answer,
            verification_passed=verification_passed,
            verification_output=verification_output,
            node_results=node_results,
            total_tokens=total_tokens,
            total_duration_ms=total_duration,
        )
    except Exception as e:
        _ORCHESTRATION_RESULTS[task_id] = e

@app.post("/tasks/{task_id}/run-orchestrated")
def run_orchestrated(task_id: UUID, background_tasks: BackgroundTasks) -> dict:
    if task_id in _ORCHESTRATION_RESULTS:
        del _ORCHESTRATION_RESULTS[task_id]
    background_tasks.add_task(_do_run_orchestrated, task_id)
    return {"status": "started"}

@app.get("/tasks/{task_id}/run-orchestrated/status")
def run_orchestrated_status(task_id: UUID) -> dict | OrchestratedRunResult:
    if task_id not in _ORCHESTRATION_RESULTS:
        # If still running, return the latest plan state so UI can track progress
        plan = store.latest_plan(task_id)
        plan_dict = plan.model_dump() if plan else None
        return {"status": "running", "plan": plan_dict}
    
    result = _ORCHESTRATION_RESULTS[task_id]
    if isinstance(result, Exception):
        raise HTTPException(status_code=500, detail=str(result))
        
    return result
