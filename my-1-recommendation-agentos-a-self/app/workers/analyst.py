from app.models import NodeType, PlanNode, WorkerContext, WorkerResult
from app.workers.base import BaseWorker


class AnalystWorker(BaseWorker):
    """Synthesizes research and structures arguments for drafting."""

    name = "analyst"
    allowed_node_types = {NodeType.ANALYZE}

    def execute(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        previous_text = ""
        for dep in node.depends_on:
            if dep in context.previous_results:
                previous_text += f"\n--- Output from {dep} ---\n{context.previous_results[dep]}\n"

        system_prompt = (
            "You are a logical analyst. Review the provided prior research/analysis outputs "
            "and synthesize them to address the OBJECTIVE. Identify conflicts, summarize key points, "
            "and outline a structured argument. Do not write the final draft, only the analysis."
        )

        prompt = (
            f"OBJECTIVE: {node.objective}\n\n"
            f"OVERALL TASK: {context.objective}\n\n"
            f"PRIOR OUTPUTS:\n{previous_text or 'None'}"
        )

        if log_event:
            log_event("system", f"Sending {len(context.previous_results)} prior outputs to LLM for analytical reasoning...")

        message, meta = self.llm.chat(prompt, system_prompt=system_prompt)
        content = message.get("content", "")

        if log_event and content:
            log_event("assistant", content)

        return WorkerResult(
            node_id=node.id,
            output=content,
            tokens_used=meta["prompt_tokens"] + meta["output_tokens"],
            duration_ms=meta["duration_ms"],
        )
