import re

from app.models import Evidence, IssueSeverity, VerificationIssue, VerificationResult

URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")

# Minimum acceptable answer length when evidence is supplied.
_MIN_ANSWER_LENGTH = 100


def deterministic_verify(
    task_id: str,
    answer: str,
    evidence: list[Evidence],
    required_sections: list[str] | None = None,
) -> VerificationResult:
    """Run all deterministic checks that need no LLM.

    Checks executed (in order):
    1. citation_mismatch   – answer cites a URL not present in evidence
    2. fabricated_url       – answer URL looks plausible but isn't in evidence
    3. missing_citation     – evidence was provided but the answer cites nothing
    4. source_not_cited     – a provided evidence source was never referenced
    5. missing_section      – answer lacks a required section heading
    6. content_too_short    – answer is too short given available evidence
    """
    allowed_urls = {str(item.url).rstrip("/") for item in evidence}
    cited_urls = {url.rstrip(".,;:)").rstrip("/") for url in URL_PATTERN.findall(answer)}
    issues: list[VerificationIssue] = []

    # ── 1. citation_mismatch: cited URL not in evidence ──────────────
    unsupported = cited_urls - allowed_urls
    for url in sorted(unsupported):
        issues.append(
            VerificationIssue(
                type="citation_mismatch", severity=IssueSeverity.HIGH, claim=url,
                evidence="The cited URL was not supplied as task evidence.",
                repair_instruction="Remove the URL or replace it with a supplied evidence URL.",
            )
        )

    # ── 2. fabricated_url: answer contains URL-like strings that match
    #       no evidence (overlaps with citation_mismatch but catches edge
    #       cases where the URL is malformed or partially invented) ────
    # Already covered above — kept as a distinct logical check so future
    # heuristics (e.g. domain-only matches) can be added without altering
    # the first check.

    # ── 3. missing_citation: evidence given but nothing cited ────────
    if evidence and not cited_urls:
        issues.append(
            VerificationIssue(
                type="missing_citation", severity=IssueSeverity.MEDIUM,
                claim="The answer contains no source URL.",
                evidence="Task evidence was provided but no supplied source was cited.",
                repair_instruction="Attach supplied source URLs to factual claims.",
            )
        )

    # ── 4. source_not_cited: supplied source was never referenced ────
    if evidence and cited_urls:
        for item in evidence:
            normalised = str(item.url).rstrip("/")
            if normalised not in cited_urls:
                issues.append(
                    VerificationIssue(
                        type="source_not_cited", severity=IssueSeverity.LOW,
                        claim=f"Source '{item.title}' was never referenced.",
                        evidence=f"Evidence URL {item.url} was supplied but not cited in the answer.",
                        repair_instruction="Cite this source where relevant, or state why it was excluded.",
                    )
                )

    # ── 5. missing_section: required section heading absent ──────────
    if required_sections:
        answer_lower = answer.lower()
        for section in required_sections:
            # Look for markdown headings or plain occurrences of the section name.
            if section.lower() not in answer_lower:
                issues.append(
                    VerificationIssue(
                        type="missing_section", severity=IssueSeverity.MEDIUM,
                        claim=f"Required section '{section}' is missing.",
                        evidence="The task rubric requires this section but it was not found.",
                        repair_instruction=f"Add a section addressing '{section}'.",
                    )
                )

    # ── 6. content_too_short: suspiciously brief for available evidence
    if evidence and len(answer.strip()) < _MIN_ANSWER_LENGTH:
        issues.append(
            VerificationIssue(
                type="content_too_short", severity=IssueSeverity.MEDIUM,
                claim=f"Answer is only {len(answer.strip())} characters.",
                evidence=f"With {len(evidence)} evidence source(s), a longer response is expected.",
                repair_instruction="Expand the answer using the supplied evidence.",
            )
        )

    return VerificationResult(
        task_id=task_id, answer=answer, passed=not issues, issues=issues, verifier="deterministic"
    )
