"""Tool to execute Python code in a sandboxed subprocess."""

import subprocess
import sys
import tempfile
import os
from app.tools.base import BaseTool


class PythonExecutorTool(BaseTool):
    """Executes Python code in a sandboxed subprocess and returns the output."""

    name = "run_python"
    description = (
        "Execute Python code and return the printed output. "
        "Use this for math calculations, data analysis, parsing CSV data, "
        "or any computation that requires precise results. "
        "The code runs in an isolated subprocess. Print your results to see them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Use print() to output results.",
            }
        },
        "required": ["code"],
    }

    def execute(self, **kwargs) -> str:
        code = kwargs.get("code")
        if not code:
            return "Error: code parameter is required."

        # Write code to a temp file and execute in a subprocess for isolation
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30,  # Hard 30-second timeout
                cwd=tempfile.gettempdir(),
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.returncode != 0 and result.stderr:
                output += f"\n[ERROR]\n{result.stderr}"

            # Clean up
            os.unlink(temp_path)

            if not output.strip():
                return "Code executed successfully but produced no output. Use print() to see results."

            # Truncate large outputs
            max_chars = 3000
            if len(output) > max_chars:
                output = output[:max_chars] + "\n\n[Output truncated at 3000 characters]"

            return output.strip()

        except subprocess.TimeoutExpired:
            os.unlink(temp_path)
            return "Error: Code execution timed out after 30 seconds."
        except Exception as e:
            return f"Error executing Python code: {e}"
