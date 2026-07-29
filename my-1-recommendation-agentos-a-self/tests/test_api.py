from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_api_status() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["api"] is True


def test_task_round_trip() -> None:
    client = TestClient(app)
    create_response = client.post(
        "/tasks",
        json={
            "objective": "Compare two approaches using the supplied evidence only.",
            "evidence": [
                {
                    "title": "Example source",
                    "url": "https://example.com/source",
                    "excerpt": "This is supplied evidence for the test.",
                }
            ],
        },
    )
    assert create_response.status_code == 201
    task = create_response.json()
    fetched = client.get(f"/tasks/{task['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["objective"] == task["objective"]


def test_task_requires_meaningful_objective() -> None:
    response = TestClient(app).post("/tasks", json={"objective": "too short"})
    assert response.status_code == 422


def test_run_fails_safely_without_local_runtime(monkeypatch) -> None:
    from app.main import llm

    monkeypatch.setattr(llm, "healthcheck", lambda: False)
    client = TestClient(app)
    created = client.post(
        "/tasks",
        json={"objective": "Create a brief from supplied evidence only."},
    ).json()
    response = client.post(f"/tasks/{created['id']}/run")
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_successful_run_is_recorded(monkeypatch) -> None:
    from app.main import llm
    from app.models import RunTrace, TaskResult, TaskStatus

    def fake_research_brief(task_id: str, objective: str, evidence: list[dict]) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            answer="Evidence-backed response.",
            trace=RunTrace(task_id=task_id, model="test-model", status=TaskStatus.COMPLETED),
        )

    monkeypatch.setattr(llm, "research_brief", fake_research_brief)
    client = TestClient(app)
    task_id = client.post(
        "/tasks", json={"objective": "Create a concise evidence-backed response."}
    ).json()["id"]
    assert client.post(f"/tasks/{task_id}/run").status_code == 200
    traces = client.get(f"/tasks/{task_id}/traces")
    assert traces.status_code == 200
    assert traces.json()[-1]["model"] == "test-model"
