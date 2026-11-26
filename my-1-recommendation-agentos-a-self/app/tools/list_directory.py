"""Tool to list directory contents for filesystem exploration."""
import os
from fnmatch import fnmatch
from pathlib import Path
from app.tools.base import BaseTool


class ListDirectoryTool(BaseTool):
    """Lists files and subdirectories at a given path for autonomous filesystem exploration."""

    name = "list_directory"
    description = (
        "List all files and subdirectories in a directory or drive root (e.g. 'E:\\', 'C:\\', 'E:\\projects'). "
        "Use this whenever the user asks to see what folders or files exist in a drive or directory, "
        "or to explore and navigate the filesystem to find project folders and files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute path to list (e.g. 'E:\\', 'C:\\Users', 'E:\\projects'). "
                    "Use drive roots like 'C:\\' or 'D:\\' or 'E:\\' to explore entire drives."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, list contents recursively up to 3 levels deep. Default false (1 level only).",
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern to filter results (e.g. '*.py', 'Face*', '*recognition*'). Case-insensitive.",
            },
        },
        "required": ["path"],
    }

    # Directories to skip during exploration (case-insensitive)
    _SKIP_DIRS = {
        "$recycle.bin", "system volume information", "windows", "programdata",
        "recovery", "config.msi", "msocache", "$windows.~bt", "$windows.~ws",
        "windowsapps", "node_modules", ".git", "__pycache__", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "appdata", "site-packages",
        "dist-packages", "wpsystem", "wudownloadcache",
    }

    MAX_RESULTS = 50
    MAX_DEPTH = 3

    def execute(self, **kwargs) -> str:
        raw_path = (kwargs.get("path") or "").strip()
        recursive = kwargs.get("recursive", False)
        pattern = (kwargs.get("pattern") or "").strip()

        if not raw_path:
            return "Error: path parameter is required. Provide an absolute path like 'E:\\' or 'C:\\Users'."

        target = Path(raw_path)

        # Handle bare drive letters like "E:" → "E:\\"
        if len(raw_path) <= 3 and raw_path[0].isalpha():
            target = Path(raw_path.rstrip("\\/:") + ":\\")

        if not target.exists():
            return f"Error: Path '{target}' does not exist. Try a different path."
        if not target.is_dir():
            return f"'{target}' is a file, not a directory. Use read_file to read its contents."

        entries: list[dict] = []
        max_depth = self.MAX_DEPTH if recursive else 1

        try:
            self._collect(target, entries, pattern, depth=0, max_depth=max_depth)
        except PermissionError:
            return f"Error: Permission denied accessing '{target}'."

        if not entries:
            hint = f" matching '{pattern}'" if pattern else ""
            return (
                f"Directory '{target}' exists but contains no visible entries{hint}. "
                "Try a broader path or different pattern."
            )

        # Format output
        lines = [f"Contents of `{target}` ({len(entries)} entries):\n"]
        for e in entries:
            icon = "[DIR]" if e["type"] == "dir" else "[FILE]"
            size_str = f" ({self._human_size(e['size'])})" if e["type"] == "file" else ""
            indent = "  " * e.get("depth", 0)
            lines.append(f"{indent}{icon} {e['name']}{size_str}")
            if e.get("path") and e.get("depth", 0) > 0:
                lines.append(f"{indent}   Path: `{e['path']}`")

        if len(entries) >= self.MAX_RESULTS:
            lines.append(f"\n⚠️ Results capped at {self.MAX_RESULTS}. Use a more specific path or pattern to narrow down.")

        return "\n".join(lines)

    def _collect(
        self, directory: Path, entries: list[dict], pattern: str,
        depth: int, max_depth: int,
    ) -> None:
        """Recursively collect directory entries."""
        if depth >= max_depth or len(entries) >= self.MAX_RESULTS:
            return

        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except (PermissionError, OSError):
            return

        for child in children:
            if len(entries) >= self.MAX_RESULTS:
                return

            name = child.name
            name_lower = name.lower()

            # Skip hidden and system directories
            if name.startswith(".") and child.is_dir():
                continue
            if child.is_dir() and name_lower in self._SKIP_DIRS:
                continue

            # Apply pattern filter (case-insensitive)
            if pattern and not fnmatch(name_lower, pattern.lower()):
                if child.is_dir() and depth < max_depth - 1:
                    # Still recurse into non-matching dirs to find matching children
                    self._collect(child, entries, pattern, depth + 1, max_depth)
                continue

            entry = {
                "name": name,
                "type": "dir" if child.is_dir() else "file",
                "depth": depth,
                "path": str(child),
            }

            if child.is_file():
                try:
                    entry["size"] = child.stat().st_size
                except OSError:
                    entry["size"] = 0

            entries.append(entry)

            # Recurse into matching directories
            if child.is_dir() and depth < max_depth - 1:
                self._collect(child, entries, pattern, depth + 1, max_depth)

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size string."""
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
