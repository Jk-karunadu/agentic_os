from app.models import NodeType, PlanNode, WorkerContext, WorkerResult
from app.workers.base import BaseWorker


class WriterWorker(BaseWorker):
    """Drafts the final response using synthesized context and evidence."""

    name = "writer"
    allowed_node_types = {NodeType.DRAFT}

    def execute(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        previous_text = ""
        for dep in node.depends_on:
            if dep in context.previous_results:
                previous_text += f"\n--- Output from {dep} ---\n{context.previous_results[dep]}\n"

        evidence_text = "\n\n".join(
            f"SOURCE: {item.title}\nURL: {item.url}\nEXCERPT: {item.excerpt}"
            for item in context.evidence
        ) or "No source evidence was supplied."

        lessons_text = "\n".join(f"- {lesson}" for lesson in context.lessons)

        is_retry = "retry" in node.id.lower()
        if is_retry:
            system_prompt = (
                "You are an expert technical writer. You are REVISING a previous draft that FAILED verification. "
                "Read the verification errors carefully. You MUST rewrite the draft to fix ALL the errors. "
                "If a claim was unsupported, REMOVE it or rewrite it using ONLY the provided evidence. "
                "If a citation is missing, ADD the exact URL from the evidence. "
                "Do NOT simply repeat the old draft."
            )
        else:
            system_prompt = (
                "You are an expert technical writer drafting a research synthesis. "
                "Write a concise, structured response to the OVERALL TASK using ONLY the facts "
                "from the PRIOR OUTPUTS and EVIDENCE. Do not invent facts or URLs. "
                "You MUST cite your claims using the exact URLs from the evidence.\n"
            )
            
        if lessons_text:
            system_prompt += f"\nIMPORTANT — LESSONS FROM PRIOR FAILURES:\n{lessons_text}\n"

        prompt = (
            f"OBJECTIVE FOR THIS DRAFT: {node.objective}\n\n"
            f"OVERALL TASK: {context.objective}\n\n"
            f"PRIOR OUTPUTS & ERRORS:\n{previous_text or 'None'}\n\n"
            f"AVAILABLE EVIDENCE:\n{evidence_text}"
        )

        if log_event:
            log_event("system", f"Sending {len(context.evidence)} evidence sources to LLM to formulate final draft...")

        message, meta = self.llm.chat(prompt, system_prompt=system_prompt)
        content = message.get("content", "")

        if log_event and content:
            log_event("assistant", content)

        return WorkerResult(
            node_id=node.id,
            output=content,
            evidence_used=[str(item.url) for item in context.evidence],
            tokens_used=meta["prompt_tokens"] + meta["output_tokens"],
            duration_ms=meta["duration_ms"],
        )
