from unittest.mock import MagicMock
import pytest

from app.agent import ConversationalAgent
from app.config import Settings
from app.tools.base import BaseTool, ToolRegistry


class DummyDirTool(BaseTool):
    name = "list_directory"
    description = "List directory contents"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def execute(self, **kwargs) -> str:
        return f"Contents of {kwargs.get('path')}: [DIR] projects, [FILE] notes.txt"


def test_agent_give_up_detection():
    settings = Settings()
    llm = MagicMock()
    registry = ToolRegistry()
    agent = ConversationalAgent(settings, llm, registry)

    assert agent._is_giving_up("Could you please provide the exact path?")
    assert agent._is_giving_up("I couldn't find any files matching your request.")
    assert agent._is_giving_up("Please provide more details about where the project is.")
    # Permission / asking questions
    assert agent._is_giving_up("The files might be relevant. Would you like me to read the contents of any of these files?")
    assert agent._is_giving_up("Do you want me to run the project now?")
    assert not agent._is_giving_up("Here is the final comparison table of the files found.")


def test_agent_procrastination_detection():
    settings = Settings()
    llm = MagicMock()
    registry = ToolRegistry()
    agent = ConversationalAgent(settings, llm, registry)

    assert agent._is_procrastinating("I will search for the files now.")
    assert agent._is_procrastinating("Let me search your downloads.")
    assert not agent._is_procrastinating("The file contains 42 entries.")


def test_rethink_prompt_escalation():
    settings = Settings()
    llm = MagicMock()
    registry = ToolRegistry()
    agent = ConversationalAgent(settings, llm, registry)

    prompt1 = agent._build_rethink_prompt(1, "search", "error")
    assert "DIFFERENT parameters" in prompt1

    prompt2 = agent._build_rethink_prompt(2, "search", "error")
    assert "list_directory" in prompt2

    prompt3 = agent._build_rethink_prompt(3, "search", "error")
    assert "run_shell" in prompt3


def test_fallback_tool_call_extraction():
    settings = Settings()
    llm = MagicMock()
    registry = ToolRegistry()
    registry.register(DummyDirTool())
    agent = ConversationalAgent(settings, llm, registry)

    # Raw JSON dumped in content
    text = 'I will now list all the folders present in the C drive.\n\n{"name": "list_directory", "arguments": {"path": "C:\\\\"}}'
    calls, cleaned = agent._extract_fallback_tool_calls(text)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "list_directory"
    assert calls[0]["function"]["arguments"] == {"path": "C:\\"}
    assert "I will now list all the folders" in cleaned
    assert '{"name": "list_directory"' not in cleaned


def test_fallback_tool_call_executes_in_loop():
    settings = Settings()
    settings.max_agent_iterations = 5

    llm = MagicMock()
    registry = ToolRegistry()
    registry.register(DummyDirTool())
    agent = ConversationalAgent(settings, llm, registry)

    # Turn 0: LLM outputs raw JSON text in content without tool_calls
    # Turn 1: LLM receives tool output and provides final answer
    llm.chat.side_effect = [
        ({"content": 'I will list the folders.\n\n{"name": "list_directory", "arguments": {"path": "C:\\\\"}}', "tool_calls": []}, {"prompt_tokens": 10, "output_tokens": 10}),
        ({"content": "Here are the folders: projects and notes.txt", "tool_calls": []}, {"prompt_tokens": 20, "output_tokens": 20}),
    ]

    events = list(agent.run_stream("what all folders in c drive", []))
    
    # Verify tool was called
    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["tool"] == "list_directory"

    # Verify tool result was observed
    obs_events = [e for e in events if e["type"] == "tool_result"]
    assert len(obs_events) == 1
    assert "Contents of C:\\" in obs_events[0]["summary"]

    # Final event
    final_event = next(e for e in events if e["type"] == "final")
    assert "projects and notes.txt" in final_event["content"]
