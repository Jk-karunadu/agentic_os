"""Tests for reflection, failure classification, and failure database (Phase 3)."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app, llm
from app.models import (
    ErrorType, Evidence, IssueSeverity, RunTrace, TaskResult, TaskStatus,
    VerificationIssue, VerificationResult,
)
from app.reflection import build_failure_record, classify_error, deterministic_lesson


# ── Reflection unit tests ────────────────────────────────────────────

def test_deterministic_lesson_aggregates_all_issues() -> None:
    result = VerificationResult(
        task_id=uuid4(), answer="Bad answer.", passed=False, verifier="test",
        issues=[
            VerificationIssue(
                type="citation_mismatch", severity=IssueSeverity.HIGH,
                claim="https://bad.com", evidence="Not supplied.", repair_instruction="Remove URL.",
            ),
            VerificationIssue(
                type="missing_section", severity=IssueSeverity.MEDIUM,
                claim="Section 'limitations' missing.", evidence="Required by rubric.",
                repair_instruction="Add limitations section.",
            ),
        ],
    )
    lesson = deterministic_lesson(str(result.task_id), result)
    assert "citation_mismatch" in lesson.trigger
    assert "missing_section" in lesson.trigger
    assert "2 issue(s)" in lesson.lesson
    assert lesson.confidence == 0.92  # highest severity is HIGH


def test_deterministic_lesson_handles_empty_issues() -> None:
    result = VerificationResult(
        task_id=uuid4(), answer="Good.", passed=True, verifier="test", issues=[],
    )
    lesson = deterministic_lesson(str(result.task_id), result)
    assert lesson.confidence == 1.0
    assert "no verification" in lesson.lesson.lower()


# ── Error classification ─────────────────────────────────────────────

def test_classify_citation_mismatch() -> None:
    issue = VerificationIssue(
        type="citation_mismatch", severity=IssueSeverity.HIGH,
        claim="x", evidence="y", repair_instruction="z",
    )
    assert classify_error(issue) == ErrorType.CITATION_MISMATCH


def test_classify_unknown_defaults_to_unsupported_claim() -> None:
    issue = VerificationIssue(
        type="unknown_new_check", severity=IssueSeverity.LOW,
        claim="x", evidence="y", repair_instruction="z",
    )
    assert classify_error(issue) == ErrorType.UNSUPPORTED_CLAIM


# ── Failure record builder ───────────────────────────────────────────

def test_build_failure_record_from_verification() -> None:
    task_id = str(uuid4())
    result = VerificationResult(
        task_id=task_id, answer="Bad.", passed=False, verifier="test",
        issues=[
            VerificationIssue(
                type="missing_citation", severity=IssueSeverity.MEDIUM,
                claim="no url", evidence="evidence given", repair_instruction="add url",
            ),
        ],
    )
    lesson = deterministic_lesson(task_id, result)
    failure = build_failure_record(task_id, result, lesson, plan_version=1)
    assert failure.error_type == ErrorType.UNSUPPORTED_CLAIM
    assert failure.plan_version == 1
    assert not failure.repair_attempted


# ── Failure database via API ─────────────────────────────────────────

def test_failure_database_stores_and_lists() -> None:
    """run-verified flow with mocked LLM stores a failure when verification fails."""

    def fake_research_brief(task_id, objective, evidence):
        return TaskResult(
            task_id=task_id, answer="Short.",
            trace=RunTrace(task_id=task_id, model="test", status=TaskStatus.COMPLETED),
        )

    def fake_research_with_lessons(task_id, objective, evidence, lessons):
        long_answer = (
            "## Comparison\nPer https://example.com/src, agents use memory. " * 3
        )
        return TaskResult(
            task_id=task_id, answer=long_answer,
            trace=RunTrace(task_id=task_id, model="test", status=TaskStatus.COMPLETED),
        )

    client = TestClient(app)
    # Monkey-patch the LLM methods.
    original_brief = llm.research_brief
    original_lessons = getattr(llm, "research_brief_with_lessons", None)
    llm.research_brief = fake_research_brief
    llm.research_brief_with_lessons = fake_research_with_lessons
    try:
        task_id = client.post(
            "/tasks",
            json={
                "objective": "Compare approaches using supplied evidence only.",
                "evidence": [{"title": "Src", "url": "https://example.com/src", "excerpt": "Evidence content here."}],
            },
        ).json()["id"]
        response = client.post(f"/tasks/{task_id}/run-verified")
        assert response.status_code == 200
        body = response.json()
        # First run was "Short." → should fail content_too_short + missing_citation.
        assert body["verification"]["passed"] is False
        assert body["lesson"] is not None
        assert body["failure"] is not None
        assert body["retried"] is True

        # Failures are stored.
        failures = client.get(f"/failures?task_id={task_id}")
        assert failures.status_code == 200
        assert len(failures.json()) >= 1
    finally:
        llm.research_brief = original_brief
        if original_lessons is not None:
            llm.research_brief_with_lessons = original_lessons


# ── Reflect endpoint requires a failed verification ──────────────────

def test_reflect_rejects_without_verification() -> None:
    client = TestClient(app)
    task_id = client.post(
        "/tasks", json={"objective": "Compare approaches for a research brief."},
    ).json()["id"]
    response = client.post(f"/tasks/{task_id}/reflect")
    assert response.status_code == 409
