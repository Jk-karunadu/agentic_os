import json
from abc import ABC, abstractmethod
from typing import Any


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute properly."""
    pass


class BaseTool(ABC):
    """Abstract base class for all tools available to the agents."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Executes the tool with the given parameters and returns a string result."""
        pass

    def get_schema(self) -> dict[str, Any]:
        """Returns the JSON schema definition for the tool (OpenAI compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Manages a collection of tools and handles execution."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.get_schema() for tool in self._tools.values()]

    def execute_tool(self, name: str, arguments: str | dict[str, Any]) -> str:
        """Executes a tool by name with the given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."

        try:
            if isinstance(arguments, str):
                kwargs = json.loads(arguments)
            else:
                kwargs = arguments
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided to tool."

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing '{name}': {e}"

    def load_generated_tools(self, tools_data: list[dict]) -> int:
        """Load persisted generated tools from DB on startup. Returns count loaded."""
        import inspect
        from app.tools.base import BaseTool as _BaseTool
        loaded = 0
        for tool_data in tools_data:
            try:
                import os, pathlib, re, json as json_mod, httpx, csv, zipfile
                import imaplib, smtplib, email, urllib, ssl, sqlite3
                ns = {
                    "BaseTool": _BaseTool, "os": os, "Path": pathlib.Path,
                    "pathlib": pathlib, "re": re, "json": json_mod,
                    "httpx": httpx, "csv": csv, "zipfile": zipfile,
                    "imaplib": imaplib, "smtplib": smtplib, "email": email,
                    "urllib": urllib, "ssl": ssl, "sqlite3": sqlite3,
                    "__name__": "__generated__",
                }
                exec(compile(tool_data["code"], f"<generated:{tool_data['name']}>", "exec"), ns)
                for obj in ns.values():
                    if inspect.isclass(obj) and issubclass(obj, _BaseTool) and obj is not _BaseTool:
                        inst = obj()
                        t_name = getattr(inst, "name", None) or tool_data["name"]
                        inst.name = t_name
                        self._tools[t_name] = inst
                        loaded += 1
                        break
            except Exception:
                pass
        return loaded

    def list_all(self) -> list[dict]:
        """Returns all registered tools with name and description."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def remove(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False
