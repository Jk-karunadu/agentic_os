"""Tests for the Phase 5 multi-agent workers and orchestration."""

from uuid import uuid4
from fastapi.testclient import TestClient

from app.main import app, llm
from app.models import (
    Evidence,
    ExecutionBudget,
    NodeType,
    PlanNode,
    TaskPlan,
    WorkerContext,
)
from app.workers.dispatcher import WorkerDispatcher


def _mock_llm_chat(
    prompt: str | None = None, 
    system_prompt: str | None = None, 
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    **kwargs
) -> tuple[dict, dict]:
    """Mock for generic chat."""
    
    # Try to extract system prompt from messages if not passed directly
    if messages and not system_prompt:
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content")
                break
                
    system_lower = system_prompt.lower() if system_prompt else ""
    
    if "research assistant" in system_lower:
        content = "Extracted fact: agents are smart."
    elif "logical analyst" in system_lower:
        content = "Analysis: The fact means agents can do work."
    elif "technical writer" in system_lower:
        content = (
            "Final draft: Agents are smart and can do work. They are very useful for many things "
            "like research and analysis. This requires a longer response to pass the minimum length "
            "verification check. https://example.com"
        )
    else:
        content = "Mocked LLM output."

    message = {"role": "assistant", "content": content}
    return message, {"prompt_tokens": 10, "output_tokens": 10, "duration_ms": 100}


def test_research_worker() -> None:
    original_chat = llm.chat
    llm.chat = _mock_llm_chat
    try:
        dispatcher = WorkerDispatcher(llm)
        node = PlanNode(id="r1", type=NodeType.RESEARCH, objective="Find facts")
        context = WorkerContext(
            task_id=uuid4(),
            objective="Do research",
            evidence=[Evidence(title="A", url="https://a.com", excerpt="A excerpt")],
            budget=ExecutionBudget(),
        )
        result = dispatcher.execute_node(node, context)
        assert result.error is None
        assert "Extracted fact" in result.output
        assert len(result.evidence_used) == 1
    finally:
        llm.chat = original_chat


def test_analyst_worker() -> None:
    original_chat = llm.chat
    llm.chat = _mock_llm_chat
    try:
        dispatcher = WorkerDispatcher(llm)
        node = PlanNode(id="a1", type=NodeType.ANALYZE, objective="Synthesize", depends_on=["r1"])
        context = WorkerContext(
            task_id=uuid4(),
            objective="Do research",
            previous_results={"r1": "Extracted fact: agents are smart."},
            budget=ExecutionBudget(),
        )
        result = dispatcher.execute_node(node, context)
        assert result.error is None
        assert "Analysis" in result.output
    finally:
        llm.chat = original_chat


def test_writer_worker() -> None:
    original_chat = llm.chat
    llm.chat = _mock_llm_chat
    try:
        dispatcher = WorkerDispatcher(llm)
        node = PlanNode(id="d1", type=NodeType.DRAFT, objective="Draft", depends_on=["a1"])
        context = WorkerContext(
            task_id=uuid4(),
            objective="Do research",
            evidence=[Evidence(title="A", url="https://a.com", excerpt="A excerpt")],
            previous_results={"a1": "Analysis: The fact means agents can do work."},
            budget=ExecutionBudget(),
        )
        result = dispatcher.execute_node(node, context)
        assert result.error is None
        assert "Final draft" in result.output
    finally:
        llm.chat = original_chat


def test_verifier_worker_passes() -> None:
    dispatcher = WorkerDispatcher(llm)
    node = PlanNode(id="v1", type=NodeType.VERIFY, objective="Verify draft", depends_on=["d1"])
    context = WorkerContext(
        task_id=uuid4(),
        objective="Do research",
        previous_results={"d1": "Agents can do work."},
        budget=ExecutionBudget(),
    )
    result = dispatcher.execute_node(node, context)
    assert result.error is None
    assert "Verification passed" in result.output


def test_orchestrated_run_endpoint() -> None:
    client = TestClient(app)

    # 1. Create a task
    task_res = client.post("/tasks", json={
        "objective": "Research something interesting.",
        "evidence": [
            {"title": "Src A", "url": "https://example.com", "excerpt": "Agents are cool."}
        ]
    })
    assert task_res.status_code == 201
    task_id = task_res.json()["id"]

    # Mock the LLM create_plan to return a deterministic 4-node plan
    def mock_create_plan(task_id_str, obj, budget, *args, **kwargs):
        return TaskPlan(
            task_id=task_id_str,
            goal="Do a full workflow",
            nodes=[
                PlanNode(id="r1", type=NodeType.RESEARCH, objective="Research objective", depends_on=[]),
                PlanNode(id="a1", type=NodeType.ANALYZE, objective="Analyze objective", depends_on=["r1"]),
                PlanNode(id="d1", type=NodeType.DRAFT, objective="Draft objective", depends_on=["a1"]),
                PlanNode(id="v1", type=NodeType.VERIFY, objective="Verify objective", depends_on=["d1"]),
            ],
            budget=ExecutionBudget(),
        )

    original_create_plan = llm.create_plan
    original_chat = llm.chat
    original_verify = getattr(llm, "verify_evidence", None)
    
    llm.create_plan = mock_create_plan
    llm.chat = _mock_llm_chat
    llm.verify_evidence = lambda *args, **kwargs: []

    try:
        res = client.post(f"/tasks/{task_id}/run-orchestrated")
        assert res.status_code == 200
        data = res.json()
        
        print("DEBUG DATA:", data)
        assert data["task_id"] == task_id
        assert data["final_answer"] is not None
        assert data["verification_passed"] is True
        assert "r1" in data["node_results"]
        assert "a1" in data["node_results"]
        assert "d1" in data["node_results"]
        assert "v1" in data["node_results"]
        
    finally:
        llm.create_plan = original_create_plan
        llm.chat = original_chat
        if original_verify:
            llm.verify_evidence = original_verify
