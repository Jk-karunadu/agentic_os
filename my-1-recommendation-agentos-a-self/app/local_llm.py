import time

import httpx

from app.config import Settings
from app.models import ExecutionBudget, PlanDraft, RunTrace, TaskPlan, TaskResult, TaskStatus, VerificationIssue


class LocalLLMUnavailable(RuntimeError):
    """Raised when the required local Ollama service/model is unavailable."""


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def healthcheck(self) -> bool:
        try:
            response = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=2.0)
            return response.is_success
        except httpx.HTTPError:
            return False

    def chat(
        self, 
        prompt: str | None = None, 
        system_prompt: str | None = None, 
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.2, 
        json_format: bool = False
    ) -> tuple[dict, dict]:
        """Generic chat completion for specialized workers, supporting tool calls."""
        if not self.healthcheck():
            raise LocalLLMUnavailable("Ollama is not running at the configured local address.")
        
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "think": False,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_ctx": self.settings.context_window,
                "num_predict": self.settings.max_output_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        if json_format:
            payload["format"] = "json"

        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LocalLLMUnavailable(f"Ollama request failed: {error}") from error
            
        body = response.json()
        message = body.get("message", {})
        duration_ms = int((time.perf_counter() - start) * 1000)
        
        metadata = {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "output_tokens": body.get("eval_count", 0),
            "duration_ms": duration_ms,
        }
        return message, metadata

    def research_brief(self, task_id: str, objective: str, evidence: list[dict]) -> TaskResult:
        if not self.healthcheck():
            raise LocalLLMUnavailable(
                "Ollama is not running at the configured local address. Install Ollama, start it, "
                f"and pull '{self.settings.ollama_model}'."
            )

        evidence_text = "\n\n".join(
            f"SOURCE: {item['title']}\nURL: {item['url']}\nEXCERPT: {item['excerpt']}"
            for item in evidence
        ) or "No source evidence was supplied. State that research cannot be completed."
        prompt = (
            "You are a local research-synthesis agent. Write a concise answer only from the "
            "evidence provided. Do not invent facts, sources, URLs, or citations. If evidence "
            "is insufficient, say so explicitly. Cite claims using the supplied URLs.\n\n"
            f"OBJECTIVE:\n{objective}\n\nEVIDENCE:\n{evidence_text}"
        )
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "stream": False,
                    "think": False,
                    "keep_alive": "10m",
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": self.settings.context_window,
                        "num_predict": self.settings.max_output_tokens,
                    },
                },
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LocalLLMUnavailable(f"Ollama request failed: {error}") from error
        body = response.json()
        answer = body.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise LocalLLMUnavailable("Ollama returned no assistant content.")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return TaskResult(
            task_id=task_id,
            answer=answer.strip(),
            trace=RunTrace(
                task_id=task_id,
                model=self.settings.ollama_model,
                status=TaskStatus.COMPLETED,
                duration_ms=duration_ms,
                prompt_tokens=body.get("prompt_eval_count"),
                output_tokens=body.get("eval_count"),
            ),
        )

    def create_plan(
        self, task_id: str, objective: str, budget: ExecutionBudget, previous_plan: TaskPlan | None = None,
        failure_reason: str | None = None,
    ) -> TaskPlan:
        if not self.healthcheck():
            raise LocalLLMUnavailable("Ollama is not running at the configured local address.")
        previous = previous_plan.model_dump_json() if previous_plan else "None"
        prompt = (
            "Create an execution plan for a local research-synthesis agent. Return JSON only. "
            "Allowed node types: research, analyze, draft, verify. The graph must be acyclic and "
            f"contain no more than {budget.max_nodes} nodes. Use short lowercase node IDs. "
            "Every plan must end in one verify node. Do not include tools, code, or claims. "
            "The top-level object must have exactly `goal` and `nodes`. Each node must have exactly "
            "`id`, `type`, `objective`, and `depends_on`. Do not use keys such as `name`, "
            "`description`, `task_graph`, or `edges`.\n\n"
            "JSON SHAPE EXAMPLE:\n"
            '{"goal":"Create a source-grounded brief","nodes":['
            '{"id":"research","type":"research","objective":"Inspect supplied evidence","depends_on":[]},'
            '{"id":"draft","type":"draft","objective":"Draft only supported claims","depends_on":["research"]},'
            '{"id":"verify","type":"verify","objective":"Check draft against evidence","depends_on":["draft"]}'
            "]}\n\n"
            f"OBJECTIVE: {objective}\nBUDGET: {budget.model_dump_json()}\n"
            f"PREVIOUS_PLAN: {previous}\nFAILURE_REASON: {failure_reason or 'None'}"
        )
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": self.settings.ollama_model, "stream": False, "think": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": self.settings.context_window,
                        "num_predict": budget.max_output_tokens,
                    },
                },
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            draft = PlanDraft.model_validate_json(content)
            plan = TaskPlan(task_id=task_id, goal=draft.goal, budget=budget, nodes=draft.nodes)
        except (httpx.HTTPError, ValueError) as error:
            detail = error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error)
            raise LocalLLMUnavailable(f"Local planner returned an invalid plan: {detail}") from error
        return plan

    def verify_evidence(self, task_id: str, answer: str, evidence: list[dict]) -> list[VerificationIssue]:
        if not self.healthcheck():
            raise LocalLLMUnavailable("Ollama is not running at the configured local address.")
        sources = "\n\n".join(
            f"URL: {item['url']}\nEXCERPT: {item['excerpt']}" for item in evidence
        )
        prompt = (
            "Check this answer only against the supplied source excerpts. Return JSON only as an object "
            "with an `issues` array. Each issue must contain type, severity (low|medium|high), claim, "
            "evidence, and repair_instruction. Return an empty issues array when every claim is supported. "
            "Never treat general knowledge as evidence. If there are many issues, ONLY LIST THE TOP 3 most critical issues to avoid truncation.\n\n"
            f"ANSWER:\n{answer}\n\nSOURCES:\n{sources}"
        )
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": self.settings.ollama_model, "stream": False, "think": False, "format": "json",
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0, "num_ctx": self.settings.context_window, "num_predict": 2000},
                }, timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json().get("message", {}).get("content")
            items = __import__("json").loads(body).get("issues", [])
            return [VerificationIssue.model_validate(item) for item in items]
        except (httpx.HTTPError, ValueError) as error:
            raise LocalLLMUnavailable(f"Local verifier returned invalid output: {error}") from error

    def research_brief_with_lessons(
        self, task_id: str, objective: str, evidence: list[dict], lessons: list[str],
    ) -> TaskResult:
        """Re-run a research brief with prior lessons injected as constraints."""
        if not self.healthcheck():
            raise LocalLLMUnavailable(
                "Ollama is not running at the configured local address."
            )
        evidence_text = "\n\n".join(
            f"SOURCE: {item['title']}\nURL: {item['url']}\nEXCERPT: {item['excerpt']}"
            for item in evidence
        ) or "No source evidence was supplied. State that research cannot be completed."

        lessons_text = "\n".join(f"- {lesson}" for lesson in lessons)
        prompt = (
            "You are a local research-synthesis agent. Write a concise answer only from the "
            "evidence provided. Do not invent facts, sources, URLs, or citations. If evidence "
            "is insufficient, say so explicitly. Cite claims using the supplied URLs.\n\n"
            "IMPORTANT — LESSONS FROM PRIOR FAILURES (you MUST follow these):\n"
            f"{lessons_text}\n\n"
            f"OBJECTIVE:\n{objective}\n\nEVIDENCE:\n{evidence_text}"
        )
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "stream": False, "think": False, "keep_alive": "10m",
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": 0.15,
                        "num_ctx": self.settings.context_window,
                        "num_predict": self.settings.max_output_tokens,
                    },
                },
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LocalLLMUnavailable(f"Ollama request failed: {error}") from error
        body = response.json()
        answer = body.get("message", {}).get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise LocalLLMUnavailable("Ollama returned no assistant content.")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return TaskResult(
            task_id=task_id,
            answer=answer.strip(),
            trace=RunTrace(
                task_id=task_id, model=self.settings.ollama_model,
                status=TaskStatus.COMPLETED, duration_ms=duration_ms,
                prompt_tokens=body.get("prompt_eval_count"),
                output_tokens=body.get("eval_count"),
            ),
        )
