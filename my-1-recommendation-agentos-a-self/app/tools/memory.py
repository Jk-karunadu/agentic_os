from typing import Any

from app.memory import MemoryStore
from app.models import MemoryQuery, MemoryType
from app.tools.base import BaseTool


class MemoryQueryTool(BaseTool):
    """Tool to search the agent's long-term memory."""

    name = "memory_query"
    description = (
        "Search the agent's long-term persistent memory for past episodic experiences, "
        "procedural lessons, or semantic facts based on a query string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to match against memories.",
            },
            "type": {
                "type": "string",
                "description": "Optional type filter. One of: episodic, procedural, semantic.",
                "enum": ["episodic", "procedural", "semantic"],
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 3,
            }
        },
        "required": ["query"],
    }

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def execute(self, **kwargs) -> str:
        query_text = kwargs.get("query")
        if not query_text:
            return "Error: query parameter is required."

        limit = kwargs.get("limit", 3)
        type_str = kwargs.get("type")
        memory_type = None
        if type_str:
            try:
                memory_type = MemoryType(type_str)
            except ValueError:
                return f"Error: Invalid memory type '{type_str}'."

        query = MemoryQuery(query=query_text, memory_type=memory_type, top_k=limit)
        results = self.memory_store.retrieve(query)

        if not results:
            return "No relevant memories found."

        output = []
        for i, item in enumerate(results, 1):
            score = item.success_score  # Or leave out score if we don't have breakdown
            output.append(f"Result {i} (Type: {item.type}):\n{item.text}")

        return "\n\n".join(output)
