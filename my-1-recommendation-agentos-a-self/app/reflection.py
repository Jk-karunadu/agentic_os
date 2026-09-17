"""Reflection: derive reusable lessons from verification failures.

The deterministic path processes *all* issues (not just the first) and maps
each issue type to the project's failure taxonomy.  An optional LLM path
asks Qwen3 for a root-cause analysis when the model is available.
"""

from __future__ import annotations

import json
from uuid import UUID

import httpx

from app.config import settings
from app.models import (
    ErrorType,
    FailureRecord,
    IssueSeverity,
    ReflectionLesson,
    VerificationIssue,
    VerificationResult,
)

# ── Issue-type → ErrorType mapping ───────────────────────────────────
_ISSUE_TO_ERROR: dict[str, ErrorType] = {
    "citation_mismatch": ErrorType.CITATION_MISMATCH,
    "fabricated_url": ErrorType.CITATION_MISMATCH,
    "missing_citation": ErrorType.UNSUPPORTED_CLAIM,
    "source_not_cited": ErrorType.SOURCE_QUALITY_FAILURE,
    "missing_section": ErrorType.MISSING_SECTION,
    "content_too_short": ErrorType.CONTENT_TOO_SHORT,
    "unsupported_claim": ErrorType.UNSUPPORTED_CLAIM,
}


def classify_error(issue: VerificationIssue) -> ErrorType:
    """Map a verification issue to a failure-taxonomy error type."""
    return _ISSUE_TO_ERROR.get(issue.type, ErrorType.UNSUPPORTED_CLAIM)


def deterministic_lesson(task_id: str, result: VerificationResult) -> ReflectionLesson:
    """Build a composite lesson from *all* verification issues.

    Previous implementation only looked at ``issues[0]``; this version
    aggregates every issue into a single actionable lesson.
    """
    if not result.issues:
        return ReflectionLesson(
            task_id=task_id,
            trigger="none",
            lesson="No verification issues were found.",
            recommended_action="No repair needed.",
            confidence=1.0,
        )

    # Collect unique triggers and repair instructions.
    triggers: list[str] = []
    repairs: list[str] = []
    max_severity = IssueSeverity.LOW

    for issue in result.issues:
        if issue.type not in triggers:
            triggers.append(issue.type)
        if issue.repair_instruction not in repairs:
            repairs.append(issue.repair_instruction)
        if _severity_rank(issue.severity) > _severity_rank(max_severity):
            max_severity = issue.severity

    trigger_text = ", ".join(triggers)
    lesson_text = (
        f"When drafting research responses, avoid these issues: {trigger_text}. "
        f"Found {len(result.issues)} issue(s) across {len(triggers)} category(ies)."
    )
    action_text = " | ".join(repairs)

    confidence = 0.92 if max_severity == IssueSeverity.HIGH else (
        0.80 if max_severity == IssueSeverity.MEDIUM else 0.65
    )

    return ReflectionLesson(
        task_id=task_id,
        trigger=trigger_text[:160],
        lesson=lesson_text[:1_000],
        recommended_action=action_text[:1_000],
        confidence=confidence,
    )


def build_failure_record(
    task_id: str,
    result: VerificationResult,
    lesson: ReflectionLesson,
    plan_version: int | None = None,
    failing_node: str | None = None,
    model: str | None = None,
) -> FailureRecord:
    """Create a structured failure record from a verification + lesson."""
    primary_type = classify_error(result.issues[0]) if result.issues else ErrorType.UNSUPPORTED_CLAIM
    detail = "; ".join(f"[{i.severity}] {i.type}: {i.claim}" for i in result.issues)[:2_000]

    return FailureRecord(
        task_id=UUID(task_id) if isinstance(task_id, str) else task_id,
        plan_version=plan_version,
        failing_node=failing_node,
        error_type=primary_type,
        error_detail=detail or "Verification failed with no specific issues.",
        verification_json=result.model_dump_json(),
        lesson_json=lesson.model_dump_json(),
        model=model or settings.ollama_model,
    )


def llm_reflect(task_id: str, result: VerificationResult) -> ReflectionLesson:
    """Ask the local LLM for a root-cause analysis.

    Falls back to ``deterministic_lesson`` if Ollama is unavailable.
    """
    issues_text = "\n".join(
        f"- [{i.severity}] {i.type}: {i.claim} — {i.evidence}" for i in result.issues
    )
    prompt = (
        "You are a reflection agent. Analyse these verification failures and produce a concise "
        "root-cause lesson. Return JSON only with keys: trigger, lesson, recommended_action, "
        "confidence (float 0-1). Be specific and actionable.\n\n"
        f"ISSUES:\n{issues_text}"
    )
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model, "stream": False,                 "format": "json",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": settings.context_window, "num_predict": 400},
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        body = json.loads(response.json().get("message", {}).get("content", "{}"))
        return ReflectionLesson(
            task_id=task_id,
            trigger=str(body.get("trigger", "llm_analysis"))[:160],
            lesson=str(body.get("lesson", "LLM reflection produced no lesson."))[:1_000],
            recommended_action=str(body.get("recommended_action", "Review manually."))[:1_000],
            confidence=min(max(float(body.get("confidence", 0.7)), 0.0), 1.0),
        )
    except Exception:
        # Graceful fallback — never block the pipeline on a reflection failure.
        return deterministic_lesson(task_id, result)


# ── helpers ──────────────────────────────────────────────────────────

def _severity_rank(severity: IssueSeverity) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(severity.value, 0)

