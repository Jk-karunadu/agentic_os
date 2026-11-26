import sys
import pytest
from app.tools.run_shell import RunShellTool


def test_run_shell_basic_execution():
    tool = RunShellTool()
    cmd = 'Write-Output "agentic_test_ok"' if sys.platform == "win32" else 'echo "agentic_test_ok"'
    result = tool.execute(command=cmd)
    assert "agentic_test_ok" in result
    assert "Exit Code: 0" in result


def test_run_shell_blocked_commands():
    tool = RunShellTool()
    # Test blocked destructive patterns
    res1 = tool.execute(command="format c:")
    assert "blocked for safety" in res1.lower()

    res2 = tool.execute(command="del /s /q test.txt")
    assert "blocked for safety" in res2.lower()

    res3 = tool.execute(command="rm -rf /")
    assert "blocked for safety" in res3.lower()


def test_run_shell_missing_command():
    tool = RunShellTool()
    res = tool.execute()
    assert "required" in res.lower()


def test_run_shell_working_directory(tmp_path):
    tool = RunShellTool()
    (tmp_path / "unique_marker.txt").write_text("found_marker", encoding="utf-8")
    
    cmd = 'Get-ChildItem -Name' if sys.platform == "win32" else 'ls'
    result = tool.execute(command=cmd, working_directory=str(tmp_path))
    assert "unique_marker.txt" in result
