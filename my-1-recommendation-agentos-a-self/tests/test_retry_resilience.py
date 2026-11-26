import json
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app, llm
from app.models import (
    ExecutionBudget,
    NodeType,
    PlanNode,
    TaskPlan,
)
from app.store import TaskStore


def test_retry_loop_uniqueness_and_graceful_exhaustion(monkeypatch) -> None:
    client = TestClient(app)

    t_res = client.post(
        "/tasks",
        json={
            "objective": "Summarize research on transformers with citations.",
            "evidence": [
                {"title": "Paper", "url": "https://arxiv.org/abs/1706.03762", "excerpt": "Attention is all you need."}
            ],
        },
    )
    assert t_res.status_code == 201
    task_id = t_res.json()["id"]

    def mock_create_plan(task_id_str, obj, budget, *args, **kwargs):
        return TaskPlan(
            task_id=task_id_str,
            goal="Research transformers",
            budget=ExecutionBudget(max_nodes=14, max_retries=2),
            nodes=[
                PlanNode(id="research", type=NodeType.RESEARCH, objective="Inspect evidence."),
                PlanNode(id="analyze", type=NodeType.ANALYZE, objective="Analyze facts.", depends_on=["research"]),
                PlanNode(id="draft", type=NodeType.DRAFT, objective="Draft synthesis.", depends_on=["analyze"]),
                PlanNode(id="verify", type=NodeType.VERIFY, objective="Verify draft.", depends_on=["draft"]),
            ],
        )

    monkeypatch.setattr(llm, "create_plan", mock_create_plan)
    def mock_chat(prompt: str = "", system_prompt: str | None = None, **kwargs):
        return (
            {"role": "assistant", "content": "Draft content without citations."},
            {"prompt_tokens": 10, "output_tokens": 10, "duration_ms": 50},
        )

    monkeypatch.setattr(llm, "chat", mock_chat)
    monkeypatch.setattr(
        llm,
        "verify_evidence",
        lambda *args, **kwargs: [{"type": "unsupported_claim", "claim": "No citation", "severity": "high"}],
    )

    run_res = client.post(f"/tasks/{task_id}/run-orchestrated")
    assert run_res.status_code == 200

    status_res = client.get(f"/tasks/{task_id}/run-orchestrated/status")
    assert status_res.status_code == 200
    data = status_res.json()

    assert data["task_id"] == task_id
    assert data["verification_passed"] is False
    assert "Verification failed" in data["verification_output"]
    assert data["final_answer"] is not None
    assert data["plan"] is not None

    plan_nodes = data["plan"]["nodes"]
    node_ids = [n["id"] for n in plan_nodes]
    assert len(node_ids) == len(set(node_ids)), f"Duplicate node IDs: {node_ids}"
    assert "retry_draft_1" in node_ids
    assert "retry_verify_1" in node_ids
    assert "retry_draft_2" in node_ids
    assert "retry_verify_2" in node_ids
    assert "retry_draft_3" not in node_ids


def test_latest_plan_gracefully_handles_corrupted_legacy_plans(tmp_path) -> None:
    db_file = tmp_path / "test_store.db"
    store = TaskStore(str(db_file))

    task_id = uuid4()
    corrupted_dict = {
        "id": str(uuid4()),
        "task_id": str(task_id),
        "version": 1,
        "goal": "Test goal for corrupted recovery",
        "budget": {"max_nodes": 8, "max_retries": 1, "timeout_seconds": 120},
        "nodes": [
            {"id": "draft", "type": "draft", "objective": "First draft", "depends_on": [], "status": "completed", "events": []},
            {"id": "verify", "type": "verify", "objective": "Verify", "depends_on": ["draft"], "status": "completed", "events": []},
            {"id": "retry_draft_0", "type": "draft", "objective": "Retry 1", "depends_on": ["draft"], "status": "completed", "events": []},
            {"id": "retry_verify_0", "type": "verify", "objective": "Verify retry 1", "depends_on": ["retry_draft_0"], "status": "completed", "events": []},
            {"id": "retry_draft_0", "type": "draft", "objective": "Retry 1 dup", "depends_on": ["draft"], "status": "completed", "events": []},
            {"id": "retry_verify_0", "type": "verify", "objective": "Verify retry 1 dup", "depends_on": ["retry_draft_0"], "status": "completed", "events": []},
        ],
    }

    with store._connect() as conn:
        conn.execute(
            "INSERT INTO plans (id, task_id, version, plan_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (corrupted_dict["id"], str(task_id), 1, json.dumps(corrupted_dict), "2026-09-19T00:00:00Z"),
        )

    plan = store.latest_plan(task_id)
    assert plan is not None
    ids = [n.id for n in plan.nodes]
    assert len(ids) == len(set(ids))
    assert ids == ["draft", "verify", "retry_draft_0", "retry_verify_0"]
