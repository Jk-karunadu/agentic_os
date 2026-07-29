"""Tests for the expanded deterministic verifier (Phase 3)."""

from uuid import uuid4

from app.models import Evidence, IssueSeverity
from app.verification import deterministic_verify


def _make_evidence(title: str = "Source A", url: str = "https://example.com/a",
                   excerpt: str = "Excerpt content for testing.") -> list[Evidence]:
    return [Evidence(title=title, url=url, excerpt=excerpt)]


# ── 1. citation_mismatch ────────────────────────────────────────────

def test_catches_unsupported_url() -> None:
    evidence = _make_evidence()
    answer = "According to https://unknown.com/fake, agents are great."
    result = deterministic_verify(str(uuid4()), answer, evidence)
    assert not result.passed
    types = {i.type for i in result.issues}
    assert "citation_mismatch" in types


# ── 2. missing_citation ─────────────────────────────────────────────

def test_catches_missing_citation_when_evidence_exists() -> None:
    evidence = _make_evidence()
    answer = "Agents use memory for self-improvement. No sources cited."
    result = deterministic_verify(str(uuid4()), answer, evidence)
    assert not result.passed
    types = {i.type for i in result.issues}
    assert "missing_citation" in types


# ── 3. source_not_cited ─────────────────────────────────────────────

def test_catches_unused_evidence_source() -> None:
    evidence = [
        Evidence(title="Source A", url="https://example.com/a", excerpt="A content."),
        Evidence(title="Source B", url="https://example.com/b", excerpt="B content."),
    ]
    answer = "Per https://example.com/a, agents learn from failures."
    result = deterministic_verify(str(uuid4()), answer, evidence)
    # Source B was never cited.
    types = {i.type for i in result.issues}
    assert "source_not_cited" in types


# ── 4. missing_section ──────────────────────────────────────────────

def test_catches_missing_required_section() -> None:
    evidence = _make_evidence()
    answer = (
        "## Introduction\nAgents use memory.\n"
        "Source: https://example.com/a"
    )
    result = deterministic_verify(
        str(uuid4()), answer, evidence,
        required_sections=["introduction", "limitations"],
    )
    assert not result.passed
    missing = [i for i in result.issues if i.type == "missing_section"]
    assert len(missing) == 1
    assert "limitations" in missing[0].claim.lower()


# ── 5. content_too_short ────────────────────────────────────────────

def test_catches_too_short_answer() -> None:
    evidence = _make_evidence()
    answer = "Short."
    result = deterministic_verify(str(uuid4()), answer, evidence)
    assert not result.passed
    types = {i.type for i in result.issues}
    assert "content_too_short" in types


# ── 6. Clean pass ────────────────────────────────────────────────────

def test_clean_answer_passes_all_checks() -> None:
    evidence = _make_evidence()
    answer = (
        "## Comparison\n"
        "According to https://example.com/a, episodic memory stores task-level traces "
        "while semantic memory holds durable facts. " * 3 +
        "\n## Limitations\nSome limitations remain.\n"
    )
    result = deterministic_verify(
        str(uuid4()), answer, evidence,
        required_sections=["comparison", "limitations"],
    )
    assert result.passed
    assert result.issues == []


# ── 7. No evidence, no citations → pass ─────────────────────────────

def test_no_evidence_and_no_citations_passes() -> None:
    result = deterministic_verify(str(uuid4()), "A plain informational response.", [])
    assert result.passed


# ── 8. Severity levels ──────────────────────────────────────────────

def test_citation_mismatch_is_high_severity() -> None:
    evidence = _make_evidence()
    answer = "See https://bad.com/paper for details."
    result = deterministic_verify(str(uuid4()), answer, evidence)
    mismatch = [i for i in result.issues if i.type == "citation_mismatch"]
    assert mismatch[0].severity == IssueSeverity.HIGH
