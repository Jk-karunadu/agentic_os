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
