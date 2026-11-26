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
    - Autonomous retry: when a tool fails, rethinks and tries a different strategy
    - Give-up override: prevents the LLM from asking the user for help prematurely
    - Soft timeout: at 80% of max timeout, wraps up gracefully
    - Persistent tool evolution: new tools saved and reloaded across sessions
    """

    SYSTEM_PROMPT = """You are AgentOS, a self-evolving AI assistant. You solve user requests by autonomously using tools. You are running on a Windows computer.

AVAILABLE TOOLS:
- `query_memory`: Query persistent episodic memory and procedural lessons.
- `file_search`: Search local folders for files matching keywords. Supports fuzzy matching and folder search. Can search entire drives.
- `read_file`: Read the full text content of a local file path (PDF, DOCX, TXT, CSV, MD).
- `list_directory`: List files and folders at any path. Use to explore the filesystem, find project folders, or understand directory structure. Works with drive roots (C:\\, D:\\, E:\\).
- `run_shell`: Execute shell commands (PowerShell on Windows). Use to run projects, install dependencies, search with 'dir /s /b', check processes, or perform any command-line operation.
- `web_search`: Search DuckDuckGo for live internet info.
- `url_scrape`: Fetch and read full webpage text from a URL.
- `run_python`: Execute Python code for math, data parsing, or comparisons.
- `generate_tool`: Write and register a brand-new tool when a capability does not exist yet.

CRITICAL EXECUTION MANDATES:
1. AUTONOMOUS ACTION: When the user asks you to search, find, read, check, or compare something:
   - NEVER reply with promises like "I will search now", "Let me do that", or "Proceeding with the following steps".
   - NEVER ask the user for permission to execute a step.
   - IMMEDIATELY call the appropriate tool in your turn.
2. MULTI-STEP COMPLETION:
   - If you search for files and find multiple candidates, immediately call `read_file` on them.
   - Once you have the contents, analyze and compare them, and format the final output.
   - Continue calling tools across iterations until the entire task is 100% finished.
3. FINAL ANSWER:
   - Only output text to the user when your task is COMPLETE and you have the final results.
4. AUTHENTICATION / SECRETS:
   - If an external service needs credentials not provided in chat, generate the tool first, then ask for credentials.

TOOL SELECTION RULES:
1. TO LIST / BROWSE DIRECTORIES OR DRIVES:
   - When user asks to "list", "show all folders", "see what is in drive/folder X" (e.g. "what all the folders present in c drive", "just get me all folders in e drive"):
     -> Call `list_directory(path="<drive or folder>")`.
     -> NEVER call `file_search` when the user simply asks to list folders!
2. TO FIND A SPECIFIC PROJECT OR FILE:
   - Call `file_search` with the project name or keywords.
   - Or call `list_directory(path="E:\\", recursive=True, pattern="*keyword*")`.
   - If you locate a project folder (like `E:\\projects\\Face_recognisation`), immediately inspect its files with `list_directory` or `read_file`.
   - If the user asked to "run" or "check" the project, inspect the files, locate the entrypoint (e.g. `main.py`, `faces.py`, `app.py`), and use `run_shell` to execute it!
   - NEVER ask the user: "Would you like me to read the contents of any of these files?" or "Should I run this?" — PROCEED AND DO IT!

FAILURE HANDLING — THIS IS CRITICAL:
When a tool returns "No files found", "Error", empty results, or any failure:
1. DO NOT tell the user you failed. DO NOT ask the user for help, a path, or clarification.
2. THINK about why it failed (wrong path? wrong query? wrong tool?)
3. IMMEDIATELY call a DIFFERENT tool or the SAME tool with DIFFERENT parameters.
4. Strategies to try in order:
   a) Same tool, broader search (e.g., search parent directory or entire drive root like E:\\)
   b) Same tool, different keywords (try synonyms, partial names, remove underscores/hyphens)
   c) list_directory — explore the filesystem to find the right path manually
   d) run_shell — use 'dir /s /b <pattern>' or 'Get-ChildItem -Recurse -Filter *keyword*' for system-wide search
5. Only give up and report to the user AFTER trying AT LEAST 3 different strategies.
6. NEVER ask the user for a "correct path", "exact name", or "more details" — figure it out yourself.

EXAMPLE — How to handle file search failures:
User: "Find my Face Recognition project"
❌ WRONG: file_search → nothing → "Could you provide the exact path?"
✅ RIGHT:
  1. file_search(query='Face Recognition', search_path='E:\\') → nothing found
  2. list_directory(path='E:\\', pattern='*face*') → found E:\\projects\\face_recog
  3. list_directory(path='E:\\projects\\face_recog') → show project structure
  4. Report what you found and proceed with the task"""

    # Phrases that indicate the LLM is giving up and asking the user for help
    _GIVE_UP_PHRASES = [
        "could you please",
        "can you provide",
        "please provide",
        "please check",
        "i couldn't find",
        "i was unable to",
        "i could not find",
        "please specify",
        "could you check",
        "do you want me to",
        "do you want",
        "would you like to",
        "would you like me to",
        "would you like",
        "should i read",
        "should i run",
        "should i execute",
        "shall i read",
        "shall i run",
        "let me know if you want",
        "let me know if you would like",
        "let me know if you'd like",
        "tell me if you want",
        "which one would you like",
        "if you want me to",
        "if you'd like me to",
        "i need more information",
        "please double-check",
        "exact path",
        "provide the path",
        "provide the file",
        "provide the exact",
        "specify a different",
        "let me know if",
        "assist you in searching",
        "if you could share",
        "share the location",
    ]

    # Phrases that indicate the LLM is making promises instead of acting
    _PROCRASTINATION_PHRASES = [
        "i will search",
        "let me search",
        "i'll search",
        "i will use the",
        "i'll use the",
        "let's start by searching",
        "proceeding with the following",
        "let me do that now",
        "i'll look for",
        "let me look for",
        "i will look for",
    ]

    def __init__(self, settings: Settings, llm: OllamaClient, tool_registry: ToolRegistry, memory_store=None) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = tool_registry
        self.memory_store = memory_store
        self._soft_timeout_secs = settings.request_timeout_seconds * settings.soft_timeout_ratio

    def _is_giving_up(self, content: str) -> bool:
        """Detect if the LLM response is giving up and asking user for help."""
        lower = content.lower()
        return any(phrase in lower for phrase in self._GIVE_UP_PHRASES)

    def _is_procrastinating(self, content: str) -> bool:
        """Detect if the LLM is making promises instead of calling tools."""
        lower = content.lower()
        return any(phrase in lower for phrase in self._PROCRASTINATION_PHRASES)

    def _extract_fallback_tool_calls(self, content: str) -> tuple[list[dict], str]:
        """Extract tool calls emitted as raw JSON text by local models in message content."""
        if not content:
            return [], content

        available = set(self.registry._tools.keys())
        extracted = []
        cleaned_content = content

        # 1. Code blocks containing JSON: ```json ... ``` or ``` ... ```
        import re
        for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL):
            raw = m.group(1).strip()
            try:
                data = json.loads(raw)
                tool_name = data.get("name") or (
                    data.get("function") if isinstance(data.get("function"), str) else data.get("function", {}).get("name")
                )
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if tool_name in available:
                    extracted.append({"function": {"name": tool_name, "arguments": args}})
                    cleaned_content = cleaned_content.replace(m.group(0), "").strip()
            except Exception:
                pass

        # 2. Raw JSON objects in text: {"name": "...", "arguments": {...}}
        if not extracted:
            for m in re.finditer(r'(\{\s*\"(?:name|function)\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\})', content, re.DOTALL):
                raw = m.group(1).strip()
                tool_name = m.group(2).strip()
                if tool_name in available:
                    try:
                        data = json.loads(raw)
                        args = data.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        extracted.append({"function": {"name": tool_name, "arguments": args}})
                        cleaned_content = cleaned_content.replace(raw, "").strip()
                    except Exception:
                        pass

        return extracted, cleaned_content

    def _build_rethink_prompt(self, rethink_count: int, last_tool: str, last_error: str) -> str:
        """Build an escalating rethink prompt based on how many retries have happened."""
        base = (
            "STOP. Do NOT ask me for help. You are an AUTONOMOUS agent. "
            "The previous approach failed. RETHINK your strategy and try something different.\n\n"
        )

        if rethink_count == 1:
            return base + (
                "Strategy: Try the SAME tool with DIFFERENT parameters.\n"
                "- Use broader search terms (remove underscores, try partial names)\n"
                "- Search a parent directory or drive root (e.g., E:\\)\n"
                "- Try different keywords or synonyms\n"
                "DO NOT respond with text. Call a tool NOW."
            )
        elif rethink_count == 2:
            return base + (
                "Strategy: Use a DIFFERENT tool entirely.\n"
                "- Use `list_directory` to explore the filesystem and find the right path\n"
                "- Try listing the drive root: list_directory(path='E:\\')\n"
                "- Or try other drives: list_directory(path='C:\\Users'), list_directory(path='D:\\')\n"
                "DO NOT respond with text. Call a tool NOW."
            )
        elif rethink_count == 3:
            return base + (
                "Strategy: Use SYSTEM COMMANDS for deep search.\n"
                "- Use `run_shell` with: dir /s /b *keyword* or Get-ChildItem -Recurse -Filter *keyword*\n"
                f"- Try: run_shell(command='Get-ChildItem -Path E:\\ -Recurse -Directory -Filter *{last_tool}* -ErrorAction SilentlyContinue | Select -First 10')\n"
                "DO NOT respond with text. Call a tool NOW."
            )
        else:
            return base + (
                "You have tried multiple approaches. Try ONE more creative strategy:\n"
                "- Search a completely different drive (C:\\, D:\\, E:\\)\n"
                "- Try a very short/generic keyword\n"
                "- Use run_shell with 'where' command\n"
                "If you truly cannot find it after this attempt, report what you tried and ask the user."
            )

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

        # Rethink tracking
        rethink_count = 0
        max_rethinks = self.settings.max_rethink_attempts
        consecutive_failures: dict[str, int] = {}  # tool_name -> consecutive failure count
        last_tool_used = ""

        # Build initial messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(history[-self.settings.chat_history_limit:])

        # Inject procedural lessons from previous failures if memory_store available
        if self.memory_store:
            relevant_lessons = self.memory_store.retrieve_relevant_lessons(user_prompt, top_k=2)
            if relevant_lessons:
                lesson_text = "\n".join(f"- {l.text}" for l in relevant_lessons)
                yield {
                    "type": "thought",
                    "content": f"Loaded {len(relevant_lessons)} procedural lesson(s) from memory.",
                }
                messages.append({
                    "role": "system",
                    "content": f"IMPORTANT — LESSONS LEARNED FROM PREVIOUS FAILURES:\n{lesson_text}",
                })

            exemplars = self.memory_store.retrieve_exemplars(user_prompt, top_k=1)
            if exemplars:
                messages.append({
                    "role": "system",
                    "content": f"EXEMPLAR OF SUCCESSFUL EXECUTION:\n{exemplars[0].text}",
                })

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

            # ── Fallback Tool Call Extraction ─────────────────────────
            # If Ollama didn't parse tool calls, but the LLM outputted raw JSON in content
            if not tool_calls and content:
                fallback_calls, cleaned_text = self._extract_fallback_tool_calls(content)
                if fallback_calls:
                    tool_calls = fallback_calls
                    content = cleaned_text

            # ── Anti-Procrastination Check ────────────────────────────
            if not tool_calls and iteration == 0 and self._is_procrastinating(content):
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

            # ── No tool calls → check if giving up or truly done ──────
            if not tool_calls:
                # Check if the LLM is giving up prematurely
                if self._is_giving_up(content) and rethink_count < max_rethinks and iteration < max_iterations - 1:
                    rethink_count += 1

                    # Extract a keyword hint from the user prompt for the rethink
                    words = [w for w in user_prompt.split() if len(w) > 3]
                    keyword_hint = words[0] if words else "target"

                    rethink_prompt = self._build_rethink_prompt(
                        rethink_count, keyword_hint, content[:200]
                    )

                    yield {
                        "type": "thought",
                        "content": f"🔄 Rethinking strategy (attempt {rethink_count}/{max_rethinks})... Previous approach gave up too early.",
                    }
                    thought_steps.append({
                        "type": "thought",
                        "content": f"Rethink #{rethink_count}: forcing agent to try a different strategy",
                    })

                    messages.append({
                        "role": "user",
                        "content": rethink_prompt,
                    })
                    continue

                # Not giving up — this is the actual final answer
                final_content = content
                break

            # ── Execute tool calls ────────────────────────────────────
            for call in tool_calls:
                func = call.get("function", {})
                tool_name = func.get("name", "")
                args = func.get("arguments", {})

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

                last_tool_used = tool_name

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

                # Track consecutive failures per tool
                is_failure = any(phrase in result.lower() for phrase in [
                    "no files found", "no files or folders found", "error:",
                    "not found", "no accessible folders", "no results",
                    "no relevant memories", "no search results",
                    "permission denied", "does not exist",
                ])

                if is_failure:
                    consecutive_failures[tool_name] = consecutive_failures.get(tool_name, 0) + 1
                else:
                    consecutive_failures[tool_name] = 0

                # If same tool has failed too many times, hint to switch tools
                max_tool_fails = self.settings.max_consecutive_tool_failures
                if consecutive_failures.get(tool_name, 0) >= max_tool_fails:
                    result += (
                        f"\n\n⚠️ SYSTEM NOTE: '{tool_name}' has failed {max_tool_fails} times in a row. "
                        "You MUST use a DIFFERENT tool now. Try: list_directory, run_shell, or a different search strategy."
                    )
                    consecutive_failures[tool_name] = 0  # Reset after hint

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

        # Store episodic memory of successful interactions when tools were used
        if self.memory_store and final_content and not timed_out and tools_used:
            try:
                from app.models import MemoryItem, MemoryType
                self.memory_store.store_memory(MemoryItem(
                    type=MemoryType.EPISODIC,
                    text=f"Task: {user_prompt[:250]}\nTools Used: {', '.join(tools_used)}\nOutcome: {final_content[:600]}",
                    importance=0.75,
                    success_score=1.0,
                ))
            except Exception:
                pass

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
