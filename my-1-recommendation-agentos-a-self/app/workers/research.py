import json

from app.local_llm import OllamaClient
from app.models import NodeType, PlanNode, WorkerContext, WorkerResult
from app.tools.base import ToolRegistry
from app.workers.base import BaseWorker


class ResearchWorker(BaseWorker):
    """Extracts relevant facts and gathers new information using tools based on the objective."""

    name = "researcher"
    allowed_node_types = {NodeType.RESEARCH}

    def __init__(self, llm_client: OllamaClient, tool_registry: ToolRegistry | None = None) -> None:
        super().__init__(llm_client)
        self.tool_registry = tool_registry

    def execute(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        evidence_text = "\n\n".join(
            f"[{i}] {item.title} ({item.url}):\n{item.excerpt}"
            for i, item in enumerate(context.evidence)
        )

        system_prompt = (
            "You are a meticulous research assistant. Extract facts relevant to the OBJECTIVE "
            "from the supplied EVIDENCE. If you need more information, use your available tools "
            "to query memory or search the web. Synthesize all findings into a clear factual report. "
            "Do not invent information. Always cite tool results using their provided URLs or IDs."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": (
                    f"OBJECTIVE: {node.objective}\n\n"
                    f"OVERALL TASK: {context.objective}\n\n"
                    f"EVIDENCE:\n{evidence_text or 'None'}"
                )
            }
        ]

        tools_schema = self.tool_registry.get_schemas() if self.tool_registry else None
        
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_duration = 0

        max_calls = context.budget.max_tool_calls
        calls_made = 0

        while True:
            if log_event:
                log_event("system", f"Sending objective '{node.objective}' to LLM to determine next research steps...")

            message, meta = self.llm.chat(messages=messages, tools=tools_schema)
            
            total_prompt_tokens += meta["prompt_tokens"]
            total_output_tokens += meta["output_tokens"]
            total_duration += meta["duration_ms"]

            messages.append(message)

            if log_event and message.get("content"):
                log_event("assistant", message["content"])

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                break

            if calls_made >= max_calls:
                messages.append({
                    "role": "user",
                    "content": "You have reached the maximum number of tool calls. Please finalize your report based on the information you have."
                })
                # Prevent infinite looping by removing tools for the final call
                tools_schema = None
                continue

            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name")
                args = func.get("arguments", {})
                
                if log_event:
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                    log_event("tool_call", f"Calling '{name}' with {args_str}", name)

                result = self.tool_registry.execute_tool(name, args)

                if log_event:
                    log_event("tool_result", f"Received {len(result)} characters from '{name}':\n{result}", name)

                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": name,
                })
                calls_made += 1

        content = messages[-1].get("content", "")

        return WorkerResult(
            node_id=node.id,
            output=content,
            evidence_used=[str(item.url) for item in context.evidence],
            tokens_used=total_prompt_tokens + total_output_tokens,
            duration_ms=total_duration,
        )
