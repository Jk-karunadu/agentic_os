from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, llm
from app.models import ExecutionBudget, PlanNode, TaskPlan
from app.models import NodeStatus
from app.scheduler import PlanScheduler, PlanTransitionError


def make_plan(task_id: str, version: int = 1) -> TaskPlan:
    return TaskPlan(
        task_id=task_id,
        version=version,
        goal="Create an evidence-backed research brief.",
        budget=ExecutionBudget(max_nodes=4),
        nodes=[
            PlanNode(id="research", type="research", objective="Inspect supplied evidence."),
            PlanNode(
                id="analyze", type="analyze", objective="Compare evidence.", depends_on=["research"]
            ),
            PlanNode(id="draft", type="draft", objective="Draft response.", depends_on=["analyze"]),
            PlanNode(id="verify", type="verify", objective="Check every claim.", depends_on=["draft"]),
        ],
    )


def test_plan_rejects_cycles() -> None:
    with pytest.raises(ValidationError, match="cycles"):
        TaskPlan(
            task_id=uuid4(),
            goal="Create an evidence-backed research brief.",
            nodes=[
                PlanNode(id="first", type="research", objective="Do research.", depends_on=["second"]),
                PlanNode(id="second", type="verify", objective="Verify research.", depends_on=["first"]),
            ],
        )


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown dependencies"):
        TaskPlan(
            task_id=uuid4(),
            goal="Create an evidence-backed research brief.",
            nodes=[PlanNode(id="verify", type="verify", objective="Verify research.", depends_on=["missing"])],
        )


def test_scheduler_enforces_dependency_order() -> None:
    plan = make_plan(str(uuid4()))
    assert PlanScheduler.ready_nodes(plan) == ["research"]
    with pytest.raises(PlanTransitionError, match="not ready"):
        PlanScheduler.start(plan, "analyze")
    running = PlanScheduler.start(plan, "research")
    after_first = PlanScheduler.complete(running, "research", succeeded=True)
    assert next(node for node in after_first.nodes if node.id == "analyze").status == NodeStatus.READY


def test_failed_node_does_not_unlock_dependents() -> None:
    plan = make_plan(str(uuid4()))
    running = PlanScheduler.start(plan, "research")
    failed = PlanScheduler.complete(running, "research", succeeded=False)
    analyze = next(node for node in failed.nodes if node.id == "analyze")
    assert analyze.status == NodeStatus.PENDING


def test_plan_and_replan_are_persisted(monkeypatch) -> None:
    def fake_create_plan(task_id: str, objective: str, budget: ExecutionBudget, previous_plan=None, failure_reason=None):
        return make_plan(task_id, 1 if previous_plan is None else previous_plan.version + 1)

    monkeypatch.setattr(llm, "create_plan", fake_create_plan)
    client = TestClient(app)
    task_id = client.post(
        "/tasks", json={"objective": "Create a concise evidence-backed response."}
    ).json()["id"]
    created = client.post(f"/tasks/{task_id}/plan", json={"max_nodes": 4})
    assert created.status_code == 200
    assert created.json()["version"] == 1
    fetched = client.get(f"/tasks/{task_id}/plan")
    assert fetched.status_code == 200
    replanned = client.post(
        f"/tasks/{task_id}/replan",
        json={"failed_node_id": "research", "failure_reason": "Primary source was unreachable."},
    )
    assert replanned.status_code == 200
    assert replanned.json()["version"] == 2


def test_plan_node_api_enforces_order(monkeypatch) -> None:
    monkeypatch.setattr(llm, "create_plan", lambda task_id, objective, budget: make_plan(task_id))
    client = TestClient(app)
    task_id = client.post(
        "/tasks", json={"objective": "Create a concise evidence-backed response."}
    ).json()["id"]
    assert client.post(f"/tasks/{task_id}/plan").status_code == 200
    assert client.post(f"/tasks/{task_id}/plan/nodes/analyze/start").status_code == 409
    assert client.post(f"/tasks/{task_id}/plan/nodes/research/start").status_code == 200
    completed = client.post(
        f"/tasks/{task_id}/plan/nodes/research/complete", json={"succeeded": True}
    )
    assert completed.status_code == 200
    assert client.get(f"/tasks/{task_id}/plan/ready").json()[0]["id"] == "analyze"
