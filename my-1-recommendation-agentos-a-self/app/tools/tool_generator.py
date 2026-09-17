"""Self-evolution engine: generates, tests, and registers new tools at runtime."""
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.tools.base import BaseTool, ToolRegistry


class ToolGeneratorTool(BaseTool):
    """Meta-tool that generates new capabilities at runtime by writing and registering Python tools."""

    name = "generate_tool"
    description = (
        "Generate and permanently register a new tool when you need a capability that doesn't exist yet. "
        "Use when the user needs: email access, WhatsApp message reading, calendar access, "
        "specific file format parsing, or any other custom capability. "
        "The tool will be created, tested, saved, and immediately available for future sessions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Snake_case name for the new tool (e.g. 'email_reader', 'whatsapp_parser', 'calendar_reader').",
            },
            "tool_description": {
                "type": "string",
                "description": "Clear description of what this tool does and when to use it.",
            },
            "capability_needed": {
                "type": "string",
                "description": "Detailed description of the capability: what inputs, what it should return, and how it should work.",
            },
        },
        "required": ["tool_name", "tool_description", "capability_needed"],
    }

    def __init__(self, llm_client, tool_registry: ToolRegistry, store) -> None:
        self.llm = llm_client
        self.registry = tool_registry
        self.store = store

    def execute(self, **kwargs) -> str:
        tool_name = (kwargs.get("tool_name") or "").strip()
        tool_description = (kwargs.get("tool_description") or "").strip()
        capability_needed = (kwargs.get("capability_needed") or "").strip()

        if not tool_name or not capability_needed:
            return "Error: tool_name and capability_needed are required."

        # Already exists?
        if tool_name in self.registry._tools:
            return f"Tool '{tool_name}' already exists and is ready to use."

        # Generate code via LLM
        code = self._generate_code(tool_name, tool_description, capability_needed)
        if not code:
            return f"Failed to generate code for '{tool_name}'. Try rephrasing the capability description."

        # Validate
        test = self._test_code(code, tool_name)
        if not test["success"]:
            fixed = self._fix_code(code, test["error"])
            if fixed:
                test2 = self._test_code(fixed, tool_name)
                if test2["success"]:
                    code, test = fixed, test2

        if not test["success"]:
            return (
                f"I tried to create '{tool_name}' but the generated code failed validation: "
                f"{test['error'][:200]}. "
                "This capability may require manual setup (credentials, installed apps, specific file access)."
            )

        # Persist to SQLite
        self.store.save_generated_tool(tool_name, tool_description, code)

        # Dynamically register
        registered = self._register_in_registry(code, tool_name)
        status = "ready to use right now" if registered else "available after next restart"
        return (
            f"✨ New tool created: **{tool_name}**. "
            f"{tool_description} "
            f"It has been saved permanently and is {status}."
        )

    # ── Code generation ───────────────────────────────────────────────

    def _generate_code(self, tool_name: str, description: str, capability: str) -> str:
        """Ask LLM to write a BaseTool subclass."""
        class_name = "".join(w.capitalize() for w in tool_name.split("_")) + "Tool"
        prompt = (
            f"Write a Python class implementing a tool for an AI assistant.\n\n"
            f"TOOL NAME: {tool_name}\n"
            f"CLASS NAME: {class_name}\n"
            f"DESCRIPTION: {description}\n"
            f"CAPABILITY REQUIRED: {capability}\n\n"
            "REQUIREMENTS:\n"
            f"- Class must be named exactly: {class_name}\n"
            f'- Must have class attribute: name = "{tool_name}"\n'
            f'- Must have class attribute: description = "{description}"\n'
            "- Must define class attribute: parameters = dict (JSON Schema with 'type': 'object', 'properties': {...}, 'required': [...])\n"
            "- Must inherit from BaseTool (import: from app.tools.base import BaseTool)\n"
            "- Must implement: def execute(self, **kwargs) -> str\n"
            "- execute() must ALWAYS return a string\n"
            "- Handle ALL exceptions gracefully (try...except), return an error string instead of crashing\n"
            "- IMPORTANT GUIDELINES FOR SERVICES REQUIRING AUTHENTICATION:\n"
            "  * FOR EMAIL / GMAIL: DO NOT use Google Cloud OAuth2 or google-api-python-client (they require browser interaction and pip packages).\n"
            "    Instead, use Python's built-in `imaplib` (imap.gmail.com with SSL on port 993) and standard `email` package with an App Password.\n"
            "    Define 'email_address' and 'app_password' as required string properties in the `parameters` schema.\n"
            "  * FOR SERVICES REQUIRING PASSWORDS / API KEYS: Always define those credentials as parameters in the `parameters` schema so the user can supply them in the chat.\n"
            "  * DO NOT ask the user to edit the Python file or install pip packages.\n"
            "- ONLY use Python standard library (imaplib, smtplib, email, urllib, ssl, sqlite3, os, pathlib, re, json, csv, zipfile) + httpx, requests\n"
            "- NO packages requiring `pip install`\n"
            "- Write the full, working implementation without placeholders.\n\n"
            "Return ONLY the Python code. No markdown fences, no explanation."
        )
        try:
            message, _ = self.llm.chat(prompt=prompt, temperature=0.1, max_output_tokens=1500)
            raw = message.get("content", "")
            # Strip markdown fences if present
            if "```python" in raw:
                raw = raw.split("```python")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return raw.strip()
        except Exception:
            return ""

    def _fix_code(self, code: str, error: str) -> str:
        """Ask LLM to fix broken code."""
        prompt = (
            f"Fix this Python tool code which failed validation.\n\n"
            f"ERROR:\n{error[:400]}\n\n"
            f"CODE:\n{code}\n\n"
            "RULES:\n"
            "- Must inherit from BaseTool and have name, description, parameters, and execute(self, **kwargs) -> str\n"
            "- For email/Gmail: use built-in imaplib (imap.gmail.com:993 SSL) and email module with an App Password\n"
            "- Only use standard library + httpx, requests (no external pip dependencies)\n"
            "Return ONLY the fixed Python code. No markdown."
        )
        try:
            message, _ = self.llm.chat(prompt=prompt, temperature=0.1, max_output_tokens=1500)
            raw = message.get("content", "")
            if "```python" in raw:
                raw = raw.split("```python")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return raw.strip()
        except Exception:
            return ""

    # ── Validation ────────────────────────────────────────────────────

    def _test_code(self, code: str, tool_name: str) -> dict:
        """Validate the generated code structure in an isolated subprocess."""
        project_root = str(Path(__file__).parent.parent.parent)

        # Write code to temp file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as code_f:
                code_f.write(code)
                code_tmp = code_f.name

            test_script = f'''import sys, types, inspect
sys.path.insert(0, r"{project_root}")

# Minimal BaseTool stub
class BaseTool:
    name = ""; description = ""; parameters = {{}}
    def execute(self, **kwargs): return ""

m_app = types.ModuleType("app")
m_tools = types.ModuleType("app.tools")
m_base = types.ModuleType("app.tools.base")
m_base.BaseTool = BaseTool
sys.modules.update({{"app": m_app, "app.tools": m_tools, "app.tools.base": m_base}})

ns = {{"BaseTool": BaseTool}}
with open(r"{code_tmp}", encoding="utf-8") as f:
    exec(compile(f.read(), r"{code_tmp}", "exec"), ns)

tool_class = next(
    (o for o in ns.values() if inspect.isclass(o) and o is not BaseTool and hasattr(o, "execute")),
    None
)
if tool_class is None:
    print("ERROR: no tool class found in generated code"); import sys; sys.exit(1)

instance = tool_class()
if not getattr(instance, "name", None):
    instance.name = "{tool_name}"
if not getattr(instance, "description", None):
    instance.description = "Generated tool"
assert callable(instance.execute), "execute is not callable"
print("OK")
'''
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as test_f:
                test_f.write(test_script)
                test_tmp = test_f.name

            result = subprocess.run(
                [sys.executable, test_tmp],
                capture_output=True, text=True, timeout=20
            )
            os.unlink(code_tmp)
            os.unlink(test_tmp)

            if result.returncode == 0 and "OK" in result.stdout:
                return {"success": True}
            return {"success": False, "error": result.stderr or result.stdout or "unknown error"}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Validation subprocess timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Dynamic registration ──────────────────────────────────────────

    def _register_in_registry(self, code: str, tool_name: str) -> bool:
        """Exec the generated code and register the tool instance."""
        try:
            from app.tools.base import BaseTool as RealBase
            import os, pathlib, re, json as jm, httpx, csv, zipfile
            import imaplib, smtplib, email, urllib, ssl, sqlite3
            ns = {
                "BaseTool": RealBase,
                "os": os,
                "Path": pathlib.Path,
                "pathlib": pathlib,
                "re": re,
                "json": jm,
                "httpx": httpx,
                "csv": csv,
                "zipfile": zipfile,
                "imaplib": imaplib,
                "smtplib": smtplib,
                "email": email,
                "urllib": urllib,
                "ssl": ssl,
                "sqlite3": sqlite3,
                "__name__": "__generated__",
            }
            exec(compile(code, f"<generated:{tool_name}>", "exec"), ns)
            for obj in ns.values():
                if inspect.isclass(obj) and issubclass(obj, RealBase) and obj is not RealBase:
                    inst = obj()
                    actual_name = getattr(inst, "name", None) or tool_name
                    inst.name = actual_name
                    self.registry.register(inst)
                    if actual_name != tool_name:
                        self.registry._tools[tool_name] = inst
                    return True
        except Exception:
            pass
        return False
