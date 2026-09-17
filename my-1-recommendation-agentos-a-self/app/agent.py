"""Self-evolving conversational agent with live thought streaming and autonomous execution."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import Settings
from app.local_llm import OllamaClient, LocalLLMUnavailable
from app.tools.base import ToolRegistry


@dataclass
class AgentResponse:
    content: str
    tools_used: list[str] = field(default_factory=list)
    new_tools_created: list[str] = field(default_factory=list)
    thought_steps: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    timed_out: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0


class ConversationalAgent:
    """
    Self-evolving conversational agent.

    Runs an autonomous ReAct loop with:
    - Live thought and execution streaming (SSE events)
    - Anti-procrastination: executes tools directly without asking for micro-approvals
    - Multi-step completion: searches, reads, compares, and synthesizes in a single run
    - Soft timeout: at 80% of max timeout, wraps up gracefully
    - Persistent tool evolution: new tools saved and reloaded across sessions
    """

    SYSTEM_PROMPT = """You are AgentOS, a self-evolving AI assistant. You solve user requests by autonomously using tools.

AVAILABLE TOOLS:
- `file_search`: Search local folders (Downloads, Documents, Desktop, OneDrive) for files matching keywords (e.g. resumes, PDFs, docs).
- `read_file`: Read the full text content of a local file path (PDF, DOCX, TXT, CSV, MD).
- `web_search`: Search DuckDuckGo for live internet info.
- `url_scrape`: Fetch and read full webpage text from a URL.
- `run_python`: Execute Python code for math, data parsing, or comparisons.
- `generate_tool`: Write and register a brand-new tool when a capability does not exist yet (e.g. Gmail, WhatsApp).

CRITICAL EXECUTION MANDATES:
1. AUTONOMOUS ACTION: When the user asks you to search, find, read, check, or compare something (e.g. "find my resumes and compare them in a table"):
   - NEVER reply with promises like "I will search now", "Let me do that", or "Proceeding with the following steps".
   - NEVER ask the user for permission to execute a step.
   - IMMEDIATELY call the appropriate tool in your turn.
2. MULTI-STEP COMPLETION:
   - If you search for files and find multiple candidates, immediately call `read_file` on them to read their contents.
   - Once you have the contents, analyze and compare them, and format the final output (e.g., a Markdown comparison table).
   - Continue calling tools across iterations until the entire task is 100% finished.
3. FINAL ANSWER:
   - Only output text to the user when your task is COMPLETE and you have the final results, tables, or conclusions.
4. AUTHENTICATION / SECRETS:
   - If an external service needs credentials (like a Gmail App Password) that were not provided in chat, generate the tool first, then politely ask the user for the email and App Password in the chat.
   - Once provided, immediately execute the tool!"""

    def __init__(self, settings: Settings, llm: OllamaClient, tool_registry: ToolRegistry) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = tool_registry
        self._soft_timeout_secs = settings.request_timeout_seconds * settings.soft_timeout_ratio

    def run_stream(self, user_prompt: str, history: list[dict]) -> Iterator[dict[str, Any]]:
        """
        Stream progress events (thoughts, tool calls, results) and the final response.

        Yields dicts with:
        - {"type": "thought", "content": str}
        - {"type": "tool_call", "tool": str, "args": dict|str}
        - {"type": "tool_result", "tool": str, "summary": str}
        - {"type": "new_tool", "tool": str}
        - {"type": "final", "content": str, "tools_used": list, ...}
        """
        start = time.perf_counter()
        tools_used: list[str] = []
        new_tools_created: list[str] = []
        thought_steps: list[dict] = []
        timed_out = False
        total_prompt_tokens = 0
        total_output_tokens = 0
        final_content = ""

        # Build initial messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(history[-self.settings.chat_history_limit:])
        messages.append({"role": "user", "content": user_prompt})

        max_iterations = self.settings.max_agent_iterations

        yield {
            "type": "thought",
            "content": f"Analyzing task: '{user_prompt[:80]}...'",
        }

        for iteration in range(max_iterations):
            elapsed = time.perf_counter() - start

            # ── Soft timeout: signal wrap-up ──────────────────────────
            if elapsed >= self._soft_timeout_secs:
                timed_out = True
                yield {
                    "type": "thought",
                    "content": "Running low on time. Wrapping up response with gathered data...",
                }
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ You are running low on time. "
                        "Please wrap up your response NOW with the best answer you can give based on what you've gathered so far. "
                        "A partial, useful answer is much better than no answer."
                    ),
                })
                message, meta = self.llm.chat(messages=messages, tools=None)
                total_prompt_tokens += meta.get("prompt_tokens", 0)
                total_output_tokens += meta.get("output_tokens", 0)
                final_content = message.get("content", "I was unable to complete the full response in time.")
                break

            # ── Dynamically refresh tool schemas ──────────────────────
            tool_schemas = self.registry.get_schemas()

            # ── Normal LLM call ───────────────────────────────────────
            try:
                message, meta = self.llm.chat(messages=messages, tools=tool_schemas)
            except LocalLLMUnavailable as e:
                err_msg = f"⚠️ LLM unavailable: {e}"
                yield {"type": "error", "content": err_msg}
                final_content = err_msg
                break

            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_output_tokens += meta.get("output_tokens", 0)
            messages.append(message)

            content = message.get("content", "").strip()
            tool_calls = message.get("tool_calls", [])

            # ── Anti-Procrastination Check ────────────────────────────
            # If the model emits conversational promises to search without calling a tool on iteration 0
            is_promising = any(
                phrase in content.lower()
                for phrase in [
                    "i will search",
                    "let me search",
                    "i'll search",
                    "i will use the",
                    "i'll use the",
                    "let's start by searching",
                    "proceeding with the following",
                    "let me do that now",
                ]
            )
            if not tool_calls and iteration == 0 and is_promising:
                yield {
                    "type": "thought",
                    "content": "Action required: executing the search directly without asking.",
                }
                messages.append({
                    "role": "user",
                    "content": "Do not explain that you will do it or ask for confirmation. Call the tool now to execute the action.",
                })
                continue

            # ── If content was generated alongside or before tools ─────
            if content and tool_calls:
                yield {
                    "type": "thought",
                    "content": content,
                }
                thought_steps.append({"type": "thought", "content": content})

            # ── No tool calls → final answer reached ──────────────────
            if not tool_calls:
                final_content = content
                break

            # ── Execute tool calls ────────────────────────────────────
            for call in tool_calls:
                func = call.get("function", {})
                tool_name = func.get("name", "")
                args = func.get("arguments", {})

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

                # Format args for display
                args_summary = ""
                if isinstance(args, dict):
                    args_summary = ", ".join(f"{k}='{v}'" for k, v in args.items() if v)
                elif isinstance(args, str):
                    args_summary = args[:80]

                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": args_summary,
                }
                thought_steps.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": args_summary,
                })

                result = self.registry.execute_tool(tool_name, args)

                # Summarize result for streaming view
                res_lines = [line.strip() for line in result.split("\n") if line.strip()]
                res_summary = res_lines[0] if res_lines else "Completed"
                if len(res_summary) > 120:
                    res_summary = res_summary[:120] + "..."

                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "summary": res_summary,
                }
                thought_steps.append({
                    "type": "tool_result",
                    "tool": tool_name,
                    "summary": res_summary,
                })

                # Track newly generated tools
                if tool_name == "generate_tool" and "✨" in result and "created" in result.lower():
                    import json as _json
                    try:
                        parsed_args = _json.loads(args) if isinstance(args, str) else args
                        created_name = parsed_args.get("tool_name", "")
                        if created_name and created_name not in new_tools_created:
                            new_tools_created.append(created_name)
                            yield {"type": "new_tool", "tool": created_name}
                    except Exception:
                        pass

                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": tool_name,
                })
        else:
            last = messages[-1]
            final_content = last.get("content", "I reached the reasoning step limit. Here is the summary of what I gathered.")

        duration_ms = int((time.perf_counter() - start) * 1000)

        yield {
            "type": "final",
            "content": final_content,
            "tools_used": tools_used,
            "new_tools": new_tools_created,
            "thought_steps": thought_steps,
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
        }

    def run(self, user_prompt: str, history: list[dict]) -> AgentResponse:
        """Synchronous run that collects all stream events into AgentResponse."""
        final_data = {}
        thought_steps = []
        for event in self.run_stream(user_prompt, history):
            if event["type"] in {"thought", "tool_call", "tool_result"}:
                thought_steps.append(event)
            elif event["type"] == "final":
                final_data = event

        return AgentResponse(
            content=final_data.get("content", ""),
            tools_used=final_data.get("tools_used", []),
            new_tools_created=final_data.get("new_tools", []),
            thought_steps=thought_steps,
            duration_ms=final_data.get("duration_ms", 0),
            timed_out=final_data.get("timed_out", False),
            prompt_tokens=final_data.get("prompt_tokens", 0),
            output_tokens=final_data.get("output_tokens", 0),
        )
