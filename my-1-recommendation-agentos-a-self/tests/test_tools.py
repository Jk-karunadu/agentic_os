import json
from uuid import uuid4

import pytest

from app.memory import MemoryStore
from app.models import MemoryItem, MemoryType
from app.tools.base import BaseTool, ToolRegistry
from app.tools.memory import MemoryQueryTool
from app.tools.search import WebSearchTool


class DummyTool(BaseTool):
    name = "dummy_tool"
    description = "A dummy tool for testing."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"}
        },
        "required": ["x", "y"]
    }

    def execute(self, **kwargs) -> str:
        return str(kwargs.get("x", 0) + kwargs.get("y", 0))


def test_tool_registry():
    registry = ToolRegistry()
    tool = DummyTool()
    registry.register(tool)

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "dummy_tool"

    # Test execution with dict
    result = registry.execute_tool("dummy_tool", {"x": 5, "y": 7})
    assert result == "12"

    # Test execution with json string
    result = registry.execute_tool("dummy_tool", '{"x": 10, "y": 20}')
    assert result == "30"

    # Test error handling
    assert "not found" in registry.execute_tool("unknown", {})
    assert "Invalid JSON" in registry.execute_tool("dummy_tool", "{invalid")


def test_memory_query_tool(tmp_path):
    db_path = tmp_path / "test.db"
    store = MemoryStore(str(db_path))
    store._initialise()

    # Add a memory
    store.store_memory(MemoryItem(
        type=MemoryType.EPISODIC,
        text="Agents can do work efficiently.",
        source_task_id=str(uuid4()),
        importance=0.8,
    ))

    tool = MemoryQueryTool(store)

    # Search for memory
    result = tool.execute(query="work efficiently")
    assert "Result 1" in result
    assert "Agents can do work" in result
    assert "Type: episodic" in result

    # Missing query
    assert "required" in tool.execute()


def test_web_search_tool():
    tool = WebSearchTool()
    
    # Missing query
    assert "required" in tool.execute()
    
    # We won't test the actual network call to Wikipedia here 
    # to avoid flaky network tests, but the schema is correct.
    schema = tool.get_schema()
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["required"]
