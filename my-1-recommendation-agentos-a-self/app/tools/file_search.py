"""Tool to search local files and extract text from documents."""
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from app.tools.base import BaseTool


class FileSearchTool(BaseTool):
    """Searches the local filesystem for files matching a query."""

    name = "file_search"
    description = (
        "Search for files on the local computer and read their content. "
        "Searches Downloads, Documents, Desktop, and OneDrive folders. "
        "Supports .txt, .pdf, .docx, .pptx, .xlsx, .eml, .csv, .md files. "
        "Use when the user asks about local files, documents, resumes, presentations, or spreadsheets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search for in file names and content (e.g. 'resume', 'invoice', 'report').",
            },
            "file_type": {
                "type": "string",
                "description": "Optional: comma-separated file extensions without dot (e.g. 'pdf,docx,txt').",
            },
            "search_path": {
                "type": "string",
                "description": "Optional: folder name or path to search (e.g. 'Downloads', 'Documents', 'Desktop', or full path).",
            },
        },
        "required": ["query"],
    }

    SUPPORTED_EXTENSIONS = {
        ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".eml", ".csv", ".md", ".log",
        ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".bat",
        ".ps1", ".html", ".css", ".js", ".ts", ".sql"
    }

    @property
    def _default_paths(self) -> list[Path]:
        """Returns all common existing user folders including OneDrive."""
        home = Path.home()
        candidates = [
            home / "Downloads",
            home / "Documents",
            home / "Desktop",
            home / "OneDrive",
            home / "OneDrive" / "Documents",
            home / "OneDrive" / "Desktop",
            home / "OneDrive" / "Downloads",
        ]
        return [p for p in candidates if p.exists()]

    def _resolve_search_dirs(self, search_path_input: str) -> list[Path]:
        """Intelligently resolves search path aliases and comma-separated inputs."""
        if not search_path_input:
            return self._default_paths

        home = Path.home()
        resolved: list[Path] = []
        raw_parts = [p.strip() for p in re.split(r"[,;]", search_path_input) if p.strip()]

        for part in raw_parts:
            low = part.lower().replace("folder", "").replace("directory", "").strip()
            
            # Drive-root support: C:, D:\, E:\, etc.
            drive_match = re.match(r"^([a-zA-Z]):[\\\/]?$", part.strip())
            if drive_match:
                drive = Path(f"{drive_match.group(1)}:\\")
                if drive.exists() and drive not in resolved:
                    resolved.append(drive)
                continue

            # Common folder aliases
            if "download" in low:
                dl = home / "Downloads"
                if dl.exists() and dl not in resolved:
                    resolved.append(dl)
            elif "document" in low:
                for doc_cand in [home / "Documents", home / "OneDrive" / "Documents"]:
                    if doc_cand.exists() and doc_cand not in resolved:
                        resolved.append(doc_cand)
            elif "desktop" in low:
                for dsk_cand in [home / "Desktop", home / "OneDrive" / "Desktop"]:
                    if dsk_cand.exists() and dsk_cand not in resolved:
                        resolved.append(dsk_cand)
            elif "onedrive" in low:
                od = home / "OneDrive"
                if od.exists() and od not in resolved:
                    resolved.append(od)
            else:
                # Check as direct path or relative to home
                cand1 = Path(part).expanduser()
                cand2 = home / part
                if cand1.exists() and cand1 not in resolved:
                    resolved.append(cand1)
                elif cand2.exists() and cand2 not in resolved:
                    resolved.append(cand2)

        return resolved if resolved else self._default_paths

    @staticmethod
    def _fuzzy_match(query: str, name: str, threshold: float = 0.70) -> bool:
        """Case-insensitive fuzzy matching for filenames/folder names.
        
        Handles typos (Face_recognisation vs Face_Recognition) and
        format variations (face-recognition, facerecognition, etc.)
        without character-bag false positives.
        """
        import difflib
        q = query.lower().strip()
        n = name.lower().strip()
        if not q or not n:
            return False

        # 1. Exact match
        if q == n:
            return True

        # 2. Query is a substring of target name (e.g. 'face' in 'face_recognition')
        if q in n:
            return True

        # 3. Target is in query ONLY if target has significant length (>= 5 chars)
        # to prevent single-letter dirs like 'i', 'c', 'io' from matching
        if len(n) >= 5 and n in q:
            return True

        # 4. Normalized comparison (strip separators: spaces, underscores, hyphens, dots)
        q_norm = re.sub(r"[-_ .]", "", q)
        n_norm = re.sub(r"[-_ .]", "", n)
        if q_norm == n_norm:
            return True
        if q_norm in n_norm:
            return True
        if len(n_norm) >= 5 and n_norm in q_norm:
            return True

        # 5. Sequence similarity on normalized string
        if difflib.SequenceMatcher(None, q_norm, n_norm).ratio() >= threshold:
            return True

        # 6. Token-level matching: every significant word in query must align to a target token
        q_tokens = [t for t in re.split(r"[-_ .]+", q) if len(t) > 2]
        n_tokens = [t for t in re.split(r"[-_ .]+", n) if len(t) > 2]
        if q_tokens and n_tokens:
            matched = 0
            for qt in q_tokens:
                if any(qt == nt or (len(nt) >= 4 and difflib.SequenceMatcher(None, qt, nt).ratio() >= 0.80) for nt in n_tokens):
                    matched += 1
            if matched == len(q_tokens):
                return True

        return False

    # Directories to skip when doing drive-root searches
    _SKIP_DIRS = {
        "$recycle.bin", "system volume information", "windows", "programdata",
        "recovery", "config.msi", "msocache", "$windows.~bt", "$windows.~ws",
        "windowsapps", "node_modules", ".git", "__pycache__", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "appdata", "site-packages",
        "dist-packages", "wpsystem", "wudownloadcache",
    }

    def execute(self, **kwargs) -> str:
        query = (kwargs.get("query") or "").strip().lower()
        file_type_arg = (kwargs.get("file_type") or "").strip().lower()
        search_path_arg = (kwargs.get("search_path") or "").strip()

        if not query:
            return "Error: query parameter is required."

        # Parse requested extensions
        ext_filters: set[str] = set()
        if file_type_arg:
            for ext in re.split(r"[,;|\s]+", file_type_arg):
                clean_ext = ext.strip().lstrip(".")
                if clean_ext:
                    ext_filters.add(f".{clean_ext}")

        search_dirs = self._resolve_search_dirs(search_path_arg)
        if not search_dirs:
            return "No accessible folders found to search."

        matches = []
        dir_matches = []
        seen_paths: set[str] = set()

        for search_dir in search_dirs:
            try:
                for root, dirs, files in os.walk(str(search_dir)):
                    # Skip hidden, cache, and system folders
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d.lower() not in self._SKIP_DIRS
                    ]
                    
                    # Check directory names for fuzzy match (find project folders)
                    for d in list(dirs):
                        dpath = Path(root) / d
                        str_dpath = str(dpath)
                        if str_dpath not in seen_paths and self._fuzzy_match(query, d):
                            seen_paths.add(str_dpath)
                            dir_matches.append({
                                "path": str_dpath,
                                "name": d,
                                "type": "directory",
                            })

                    for fname in files:
                        try:
                            fpath = Path(root) / fname
                            suffix = fpath.suffix.lower()
                            
                            # If user specified extension filter (e.g. file_type="pdf"), enforce strictly
                            if ext_filters and suffix not in ext_filters:
                                continue

                            str_path = str(fpath)
                            if str_path in seen_paths:
                                continue

                            # Use fuzzy matching on filename
                            name_match = self._fuzzy_match(query, fname)
                            can_extract = suffix in self.SUPPORTED_EXTENSIONS
                            content = self._extract_text(fpath) if (can_extract and (not name_match or suffix in {".pdf", ".docx", ".txt", ".md"})) else ""
                            content_match = query in content.lower() if content else False

                            if name_match or content_match:
                                seen_paths.add(str_path)
                                snippet = self._get_snippet(content, query) if content else ""
                                matches.append({
                                    "path": str_path,
                                    "name": fname,
                                    "ext": suffix,
                                    "snippet": snippet,
                                })

                            if len(matches) >= 10:
                                break
                        except (PermissionError, OSError):
                            continue
                    if len(matches) >= 10 and len(dir_matches) >= 5:
                        break
            except (PermissionError, OSError):
                continue
            if len(matches) >= 10 and len(dir_matches) >= 5:
                break

        if not matches and not dir_matches:
            dirs_str = ", ".join(d.name for d in search_dirs)
            type_str = f" of type {file_type_arg}" if file_type_arg else ""
            return f"No files or folders found matching '{query}'{type_str} in: {dirs_str}."

        lines = []
        
        # Show directory matches first
        if dir_matches:
            lines.append(f"Found {len(dir_matches)} folder(s) matching '{query}':\n")
            for i, m in enumerate(dir_matches[:5], 1):
                lines.append(f"{i}. [DIR] **{m['name']}/**")
                lines.append(f"   Path: `{m['path']}`")
                lines.append("")

        # Show file matches
        if matches:
            lines.append(f"Found {len(matches)} file(s) matching '{query}':\n")
            for i, m in enumerate(matches, 1):
                lines.append(f"{i}. [FILE] **{m['name']}** ({m['ext']})")
                lines.append(f"   Path: `{m['path']}`")
                if m["snippet"]:
                    lines.append(f"   Excerpt: {m['snippet'][:250]}")
                lines.append("")
        
        return "\n".join(lines).strip()

    def _extract_text(self, path: Path) -> str:
        try:
            suffix = path.suffix.lower()
            if suffix in {
                ".txt", ".md", ".log", ".csv", ".eml", ".py", ".json", ".yaml",
                ".yml", ".toml", ".ini", ".cfg", ".sh", ".bat", ".ps1", ".html",
                ".css", ".js", ".ts", ".sql"
            }:
                return path.read_text(encoding="utf-8", errors="ignore")[:4000]
            elif suffix == ".docx":
                with zipfile.ZipFile(path) as z:
                    with z.open("word/document.xml") as f:
                        return " ".join(n.text for n in ET.parse(f).iter() if n.text)[:3000]
            elif suffix == ".pptx":
                texts = []
                with zipfile.ZipFile(path) as z:
                    for name in z.namelist():
                        if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                            with z.open(name) as f:
                                texts.extend(n.text for n in ET.parse(f).iter() if n.text)
                return " ".join(texts)[:3000]
            elif suffix == ".xlsx":
                with zipfile.ZipFile(path) as z:
                    if "xl/sharedStrings.xml" in z.namelist():
                        with z.open("xl/sharedStrings.xml") as f:
                            return " ".join(n.text for n in ET.parse(f).iter() if n.text)[:3000]
            elif suffix == ".pdf":
                data = path.read_bytes()
                text = re.sub(rb"[^\x20-\x7e\n]", b" ", data).decode("ascii", errors="ignore")
                return re.sub(r"\s+", " ", text).strip()[:3000]
        except Exception:
            pass
        return ""

    def _get_snippet(self, text: str, query: str) -> str:
        if not text:
            return ""
        idx = text.lower().find(query.lower())
        if idx == -1:
            return text[:200]
        start = max(0, idx - 80)
        end = min(len(text), idx + 180)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return prefix + text[start:end] + suffix
