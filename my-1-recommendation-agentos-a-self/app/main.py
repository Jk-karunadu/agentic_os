from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, status, BackgroundTasks

from app.config import settings
from app.local_llm import LocalLLMUnavailable, OllamaClient
from app.memory import MemoryStore
from app.models import (
    ChatRequest, ChatResponse,
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
from app.tools.url_scraper import URLScraperTool
from app.tools.python_executor import PythonExecutorTool
from app.tools.file_reader import FileReaderTool
from app.tools.file_search import FileSearchTool
from app.tools.list_directory import ListDirectoryTool
from app.tools.run_shell import RunShellTool
from app.tools.tool_generator import ToolGeneratorTool

tool_registry = ToolRegistry()
tool_registry.register(MemoryQueryTool(memory_store))
tool_registry.register(WebSearchTool())
tool_registry.register(URLReaderTool())
tool_registry.register(URLScraperTool())
tool_registry.register(PythonExecutorTool())
tool_registry.register(FileReaderTool())
tool_registry.register(FileSearchTool())
tool_registry.register(ListDirectoryTool())
tool_registry.register(RunShellTool())
tool_gen = ToolGeneratorTool(llm, tool_registry, store)
tool_registry.register(tool_gen)
tool_registry.set_healer(tool_gen)

# Load persisted generated tools from previous sessions
_saved_tools = store.load_generated_tools()
if _saved_tools:
    tool_registry.load_generated_tools(_saved_tools)

from app.agent import ConversationalAgent
agent = ConversationalAgent(settings, llm, tool_registry, memory_store=memory_store)

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


@app.get("/lessons", response_model=list[ReflectionLesson])
def list_all_lessons(task_id: UUID | None = None, min_effectiveness: float = 0.0) -> list[ReflectionLesson]:
    return store.list_lessons(task_id=task_id, min_effectiveness=min_effectiveness)


@app.get("/reflection/stats")
def get_reflection_stats() -> dict:
    all_lessons = store.list_lessons()
    failures = store.list_failures()
    tools = store.load_generated_tools()

    avg_effectiveness = (
        round(sum(l.effectiveness_score for l in all_lessons) / len(all_lessons), 2)
        if all_lessons else 0.0
    )

    failure_types: dict[str, int] = {}
    for f in failures:
        t = f.error_type if isinstance(f.error_type, str) else f.error_type.value
        failure_types[t] = failure_types.get(t, 0) + 1

    return {
        "total_lessons": len(all_lessons),
        "avg_effectiveness": avg_effectiveness,
        "total_failures": len(failures),
        "failures_by_type": failure_types,
        "generated_tools_count": len(tools),
        "generated_tools": tools,
        "lessons": [l.model_dump() for l in all_lessons],
    }


@app.post("/reflection/prune")
def prune_reflections(threshold: float = 0.20) -> dict:
    pruned_lessons = store.prune_ineffective_lessons(threshold)
    pruned_memory = memory_store.prune_ineffective(threshold)
    return {"pruned_lessons": pruned_lessons, "pruned_memory": pruned_memory}


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
    rel_mems = memory_store.retrieve_relevant_lessons(task.objective, top_k=3)
    lesson_texts = [m.text for m in rel_mems] if rel_mems else [l.recommended_action for l in store.list_lessons(task_id)]
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

    # Update failure record and lesson effectiveness
    store.update_failure_repair(failure.id, retry_verification.passed)
    if lesson.id:
        if retry_verification.passed:
            store.reward_lesson(lesson.id)
        else:
            store.penalize_lesson(lesson.id)

    final_status = TaskStatus.COMPLETED if retry_verification.passed else TaskStatus.FAILED
    store.set_status(task_id, final_status)

    # Auto-store episodic memory and exemplar from the final result.
    final_answer = retry_result.answer if retry_verification.passed else first_result.answer
    _auto_store_episodic(task, final_answer, passed=retry_verification.passed)
    if retry_verification.passed:
        memory_store.store_exemplar(
            task_id=str(task_id),
            objective=task.objective,
            plan_summary="direct_retry_with_lessons",
            final_answer=final_answer,
            tokens_used=retry_result.trace.output_tokens or 0,
        )

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
    memory_store.store_procedural_lesson(
        task_id=str(task.id),
        trigger=lesson.trigger,
        lesson_text=lesson.lesson,
        recommended_action=lesson.recommended_action,
        confidence=lesson.confidence,
        lesson_id=lesson.id,
    )


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

        # Retrieve semantically relevant procedural lessons and exemplars
        relevant_mem_lessons = memory_store.retrieve_relevant_lessons(task.objective, top_k=3)
        applied_lessons_texts: list[str] = []
        applied_lesson_ids: list[int] = []
        import json as _json
        for mem in relevant_mem_lessons:
            applied_lessons_texts.append(mem.text)
            try:
                meta = _json.loads(mem.metadata_json or "{}")
                if meta.get("lesson_id"):
                    applied_lesson_ids.append(int(meta["lesson_id"]))
            except Exception:
                pass

        if not applied_lessons_texts:
            fallback_lessons = store.list_lessons(min_effectiveness=0.3)[:3]
            for l in fallback_lessons:
                applied_lessons_texts.append(f"{l.trigger}: {l.recommended_action}")
                if l.id:
                    applied_lesson_ids.append(l.id)

        exemplars = [ex.text for ex in memory_store.retrieve_exemplars(task.objective, top_k=1)]

        context = WorkerContext(
            task_id=task_id,
            objective=task.objective,
            evidence=task.evidence,
            previous_results={},
            lessons=applied_lessons_texts,
            applied_lesson_ids=applied_lesson_ids,
            exemplars=exemplars,
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
                        plan=plan.model_dump(),
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

                    if not verification_passed:
                        from app.models import VerificationResult, VerificationIssue
                        from app.reflection import deterministic_lesson, build_failure_record
                        
                        draft_node_id = node.depends_on[0] if node.depends_on else None
                        draft_answer = context.previous_results.get(draft_node_id, "unknown") if draft_node_id else "unknown"

                        issues = []
                        for line in result.output.split("\n"):
                            if line.startswith("- "):
                                parts = line[2:].split(":", 1)
                                if len(parts) == 2:
                                    severity = "medium"
                                    # Try to extract severity if it exists in the output like [- [high] type: claim]
                                    import re
                                    sev_match = re.search(r"\[(.*?)\]", parts[0])
                                    if sev_match:
                                        severity = sev_match.group(1).lower()
                                        parts[0] = parts[0].replace(f"[{sev_match.group(1)}]", "").strip()
                                        
                                    issues.append(VerificationIssue(
                                        type=parts[0].strip() or "unknown_error", 
                                        severity=severity if severity in ["low", "medium", "high"] else "medium", 
                                        claim=parts[1].strip() or "Unknown claim",
                                        evidence="Parsed from orchestrator", 
                                        repair_instruction="Fix this issue."
                                    ))
                        if not issues:
                            issues.append(VerificationIssue(type="unknown_error", severity="high", claim=result.output[:999], evidence="None", repair_instruction="None"))
                        
                        v_result = VerificationResult(
                            task_id=task_id,
                            answer=draft_answer,
                            passed=False,
                            issues=issues,
                            verifier="orchestrator"
                        )
                        lesson = deterministic_lesson(str(task_id), v_result)
                        store.add_lesson(lesson)
                        failure = build_failure_record(str(task_id), v_result, lesson, failing_node=node.id, model=settings.ollama_model)
                        store.add_failure(failure)
                        _auto_store_procedural(task, lesson)
                        
                        can_retry = context.budget.max_retries > 0
                        if can_retry:
                            context.budget.max_retries -= 1
                            # Reload lessons semantically so the newly learned lesson is in context
                            revised_mems = memory_store.retrieve_relevant_lessons(task.objective, top_k=3)
                            context.lessons = [m.text for m in revised_mems] if revised_mems else [lesson.recommended_action]
                            if lesson.id and lesson.id not in context.applied_lesson_ids:
                                context.applied_lesson_ids.append(lesson.id)
                        
                            draft_node = next((n for n in plan.nodes if n.id == draft_node_id), None) if draft_node_id else None
                            if draft_node:
                                existing_retries = sum(1 for n in plan.nodes if n.id.startswith("retry_draft_"))
                                new_retry_num = existing_retries + 1
                                new_draft_id = f"retry_draft_{new_retry_num}"
                                new_verify_id = f"retry_verify_{new_retry_num}"
                                
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
                                # Increment max_nodes to prevent validation errors when appending to graph
                                plan.budget.max_nodes = max(plan.budget.max_nodes, len(plan.nodes))
                                store.update_plan(plan)
                        else:
                            log_event("system", "Verification failed and retry budget is exhausted. Concluding task.")

                # Complete with success
                plan = PlanScheduler.complete(plan, node.id, succeeded=True)

        store.update_plan(plan)
    
        # Check if plan fully completed
        is_completed = all(n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED) for n in plan.nodes)
        rewarded_ids: list[int] = []
        penalized_ids: list[int] = []

        if is_completed and verification_passed:
            store.set_status(task_id, TaskStatus.COMPLETED)
            if final_answer:
                _auto_store_episodic(task, final_answer, passed=True)
                plan_summary = " -> ".join(n.id for n in plan.nodes)
                memory_store.store_exemplar(
                    task_id=str(task_id),
                    objective=task.objective,
                    plan_summary=plan_summary,
                    final_answer=final_answer,
                    tokens_used=total_tokens,
                )
            for lid in context.applied_lesson_ids:
                if store.reward_lesson(lid):
                    rewarded_ids.append(lid)
        else:
            store.set_status(task_id, TaskStatus.FAILED)
            for lid in context.applied_lesson_ids:
                if store.penalize_lesson(lid):
                    penalized_ids.append(lid)

        _ORCHESTRATION_RESULTS[task_id] = OrchestratedRunResult(
            task_id=task_id,
            plan=plan.model_dump(),
            final_answer=final_answer,
            verification_passed=verification_passed,
            verification_output=verification_output,
            node_results=node_results,
            total_tokens=total_tokens,
            total_duration_ms=total_duration,
            applied_lessons=context.lessons,
            lessons_rewarded=rewarded_ids,
            lessons_penalized=penalized_ids,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        store.set_status(task_id, TaskStatus.FAILED)
        plan_dict = plan.model_dump() if 'plan' in locals() and plan else None
        _ORCHESTRATION_RESULTS[task_id] = OrchestratedRunResult(
            task_id=task_id,
            plan=plan_dict,
            final_answer=final_answer if 'final_answer' in locals() and final_answer else f"Execution stopped due to error: {e}",
            verification_passed=False,
            verification_output=f"Error in orchestrator: {e}",
            node_results=node_results if 'node_results' in locals() else {},
            total_tokens=total_tokens if 'total_tokens' in locals() else 0,
            total_duration_ms=total_duration if 'total_duration' in locals() else 0,
        )

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
        try:
            plan = store.latest_plan(task_id)
            plan_dict = plan.model_dump() if plan else None
        except Exception:
            plan_dict = None
        return {"status": "running", "plan": plan_dict}
    
    result = _ORCHESTRATION_RESULTS[task_id]
    if isinstance(result, Exception):
        return {
            "status": "error",
            "error": str(result),
            "verification_passed": False,
            "final_answer": f"Backend error: {result}",
        }
        
    return result


# ── Chat (Self-Evolving Agent) ────────────────────────────────────────

import uuid as _uuid

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint for the self-evolving conversational agent."""
    if not llm.healthcheck():
        raise HTTPException(status_code=503, detail="Ollama is not running at the configured local address.")

    session_id = request.session_id or str(_uuid.uuid4())

    # Load conversation history
    history_records = store.get_history(session_id, limit=settings.chat_history_limit)
    history_for_llm = [{"role": r["role"], "content": r["content"]} for r in history_records]

    # Save user message
    store.add_message(session_id, "user", request.prompt)

    # Run agent
    response = agent.run(request.prompt, history_for_llm)

    # Save assistant response
    store.add_message(
        session_id, "assistant", response.content,
        tools_used=response.tools_used,
        new_tools=response.new_tools_created,
    )

    return ChatResponse(
        content=response.content,
        session_id=session_id,
        tools_used=response.tools_used,
        new_tools_created=response.new_tools_created,
        duration_ms=response.duration_ms,
        timed_out=response.timed_out,
        prompt_tokens=response.prompt_tokens,
        output_tokens=response.output_tokens,
    )


from fastapi.responses import StreamingResponse
import json as _json

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming chat endpoint yielding live thoughts, tool calls, and final answer via SSE."""
    if not llm.healthcheck():
        raise HTTPException(status_code=503, detail="Ollama is not running at the configured local address.")

    session_id = request.session_id or str(_uuid.uuid4())
    history_records = store.get_history(session_id, limit=settings.chat_history_limit)
    history_for_llm = [{"role": r["role"], "content": r["content"]} for r in history_records]

    store.add_message(session_id, "user", request.prompt)

    def event_generator():
        final_content = ""
        tools_used = []
        new_tools = []
        for event in agent.run_stream(request.prompt, history_for_llm):
            if event["type"] == "final":
                final_content = event.get("content", "")
                tools_used = event.get("tools_used", [])
                new_tools = event.get("new_tools", [])
                event["session_id"] = session_id

            yield f"data: {_json.dumps(event)}\n\n"

        if final_content:
            store.add_message(
                session_id, "assistant", final_content,
                tools_used=tools_used,
                new_tools=new_tools,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str) -> list[dict]:
    """Returns the conversation history for a session."""
    return store.get_history(session_id, limit=100)


@app.get("/chat/sessions")
def get_sessions() -> list[str]:
    """Returns all chat session IDs."""
    return store.list_sessions()


@app.get("/tools")
def list_tools() -> list[dict]:
    """Returns all currently registered tools."""
    return tool_registry.list_all()


@app.delete("/tools/generated/{tool_name}")
def delete_generated_tool(tool_name: str) -> dict:
    """Removes a generated tool from the registry and database."""
    db_ok = store.delete_generated_tool(tool_name)
    reg_ok = tool_registry.remove(tool_name)
    if not db_ok and not reg_ok:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")
    return {"deleted": tool_name, "from_db": db_ok, "from_registry": reg_ok}
