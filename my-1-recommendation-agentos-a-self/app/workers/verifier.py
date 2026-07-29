from app.models import NodeType, PlanNode, WorkerContext, WorkerResult
from app.verification import deterministic_verify
from app.workers.base import BaseWorker


class VerifierWorker(BaseWorker):
    """Verifies a drafted response against the supplied evidence."""

    name = "verifier"
    allowed_node_types = {NodeType.VERIFY}

    def execute(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        draft_text = ""
        for dep in node.depends_on:
            if dep in context.previous_results:
                draft_text = context.previous_results[dep]
                break

        if not draft_text:
            return WorkerResult(
                node_id=node.id,
                output="No draft available to verify.",
                error="No draft provided by dependencies.",
            )

        if log_event:
            log_event("system", f"Running strict python deterministic checks (url citations, length constraint) on {len(draft_text)} char draft...")

        # 1. Deterministic verification
        result = deterministic_verify(
            task_id=str(context.task_id),
            answer=draft_text,
            evidence=context.evidence,
        )

        if not result.passed:
            issues_str = "\n".join(f"- {i.type}: {i.claim}" for i in result.issues)
            if log_event:
                log_event("assistant", f"Failed deterministic check:\n{issues_str}")
            return WorkerResult(
                node_id=node.id,
                output=f"Verification failed (deterministic):\n{issues_str}",
            )

        # 2. LLM Verification
        if context.evidence:
            if log_event:
                log_event("system", "Running LLM semantic fact-checking...")
            evidence_dicts = [item.model_dump(mode="json") for item in context.evidence]
            try:
                llm_issues = self.llm.verify_evidence(
                    task_id=str(context.task_id),
                    answer=draft_text,
                    evidence=evidence_dicts,
                )
                if llm_issues:
                    issues_str = "\n".join(f"- {i.type}: {i.claim}" for i in llm_issues)
                    if log_event:
                        log_event("assistant", f"Failed LLM fact-checking:\n{issues_str}")
                    return WorkerResult(
                        node_id=node.id,
                        output=f"Verification failed (LLM):\n{issues_str}",
                    )
            except Exception as e:
                # Do not set result.error here, otherwise the orchestrator will hard-crash the task.
                # Instead, treat it as a standard verification failure so it can retry.
                error_msg = str(e)
                if log_event:
                    log_event("assistant", f"Failed LLM fact-checking (API/JSON Error): {error_msg}")
                return WorkerResult(
                    node_id=node.id,
                    output=f"Verification failed (LLM Error - JSON truncation or API issue):\n- json_error: {error_msg}. Please simplify your claims.",
                )
        
        if log_event:
            log_event("assistant", "Verification passed.")

        return WorkerResult(
            node_id=node.id,
            output="Verification passed.",
        )
