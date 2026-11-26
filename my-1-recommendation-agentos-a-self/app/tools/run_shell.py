"""Tool to execute shell commands for autonomous task execution."""
import os
import re
import subprocess
import sys
from app.tools.base import BaseTool


class RunShellTool(BaseTool):
    """Executes shell commands and returns the output."""

    name = "run_shell"
    description = (
        "Execute a shell command on the local computer and return the output. "
        "Use this to: run projects (python, npm, etc.), install dependencies, "
        "check running processes, search files with system commands (dir, where, findstr), "
        "navigate the filesystem, or perform any command-line operation. "
        "On Windows this runs via PowerShell. "
        "Use when you need to DO something (run code, install packages) not just find files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The shell command to execute. Examples: "
                    "'dir E:\\projects /s /b', "
                    "'python main.py', "
                    "'pip install -r requirements.txt', "
                    "'Get-ChildItem -Recurse -Filter *.py'"
                ),
            },
            "working_directory": {
                "type": "string",
                "description": "Optional working directory for the command. Defaults to user home.",
            },
        },
        "required": ["command"],
    }

    TIMEOUT_SECONDS = 60
    MAX_OUTPUT_CHARS = 5000

    # Dangerous command patterns to block (case-insensitive)
    _BLOCKED_PATTERNS = [
        r"\bformat\s+[a-z]:",
        r"\bdel\s+/s\s+/q\b",
        r"\brm\s+-(?:[a-zA-Z]*r[a-zA-Z]*f|[a-zA-Z]*f[a-zA-Z]*r)\s+/(?:\s|$)",
        r"\brm\s+-rf\b",
        r"\brmdir\s+/s\s+/q\s+[a-z]:\\?$",
        r"\bRemove-Item\s+.*-Recurse.*[a-z]:\\?$",
        r"\breg\s+delete\b",
        r"\bbcdedit\b",
        r"\bdiskpart\b",
        r"\bnet\s+user\b.*\/add\b",
        r"\bshutdown\b",
    ]

    def execute(self, **kwargs) -> str:
        command = (kwargs.get("command") or "").strip()
        cwd = (kwargs.get("working_directory") or "").strip()

        if not command:
            return "Error: command parameter is required."

        # Safety check
        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return (
                    f"Error: Command blocked for safety. The pattern '{command[:60]}' matches a "
                    "potentially destructive operation. Please use a safer alternative."
                )

        # Resolve working directory
        if cwd:
            if not os.path.isdir(cwd):
                return f"Error: Working directory '{cwd}' does not exist."
        else:
            cwd = str(os.path.expanduser("~"))

        # Determine shell
        if sys.platform == "win32":
            shell_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            shell_cmd = ["/bin/sh", "-c", command]

        try:
            result = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
                cwd=cwd,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {self.TIMEOUT_SECONDS} seconds. "
                "The command may be long-running. Try a more targeted command."
            )
        except FileNotFoundError:
            return "Error: Shell not found. Cannot execute command."
        except Exception as e:
            return f"Error executing command: {e}"

        # Build output
        output_parts = []

        if result.stdout:
            stdout = result.stdout.strip()
            if len(stdout) > self.MAX_OUTPUT_CHARS:
                stdout = stdout[:self.MAX_OUTPUT_CHARS] + f"\n\n[Output truncated at {self.MAX_OUTPUT_CHARS} chars]"
            output_parts.append(stdout)

        if result.stderr:
            stderr = result.stderr.strip()
            if len(stderr) > 1500:
                stderr = stderr[:1500] + "\n[stderr truncated]"
            if stderr:
                output_parts.append(f"[STDERR]\n{stderr}")

        exit_info = f"[Exit Code: {result.returncode}]"

        if not output_parts:
            if result.returncode == 0:
                return f"Command executed successfully (no output). {exit_info}"
            return f"Command failed with no output. {exit_info}"

        combined = "\n\n".join(output_parts)
        return f"{combined}\n\n{exit_info}"
