"""Tool to read local files (TXT, CSV, PDF, DOCX, JSON, etc.)."""

import os
from app.tools.base import BaseTool


class FileReaderTool(BaseTool):
    """Reads the text content of a local file."""

    name = "read_file"
    description = (
        "Read the contents of a local file. Supports plain text files (.txt, .csv, .json, .md, .log), "
        "PDF files (.pdf), and Word documents (.docx). "
        "Returns the extracted text content (up to 5000 characters)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The absolute or relative file path to read.",
            }
        },
        "required": ["path"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path")
        if not path:
            return "Error: path parameter is required."

        if not os.path.exists(path):
            return f"Error: File not found at '{path}'."

        ext = os.path.splitext(path)[1].lower()
        max_chars = 5000

        try:
            if ext == ".pdf":
                text = self._read_pdf(path)
            elif ext == ".docx":
                text = self._read_docx(path)
            else:
                # Plain text fallback for .txt, .csv, .json, .md, .log, etc.
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()

            if not text.strip():
                return f"File '{os.path.basename(path)}' exists but contains no readable text."

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} characters]"

            return f"Contents of {os.path.basename(path)}:\n\n{text}"

        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def _read_pdf(path: str) -> str:
        """Extract text from a PDF file."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            # Fallback: try pdfplumber
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return "\n\n".join(text_parts)
            except ImportError:
                return (
                    "Error: No PDF library available. "
                    "Install PyMuPDF (pip install pymupdf) or pdfplumber (pip install pdfplumber)."
                )

    @staticmethod
    def _read_docx(path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError:
            return "Error: python-docx not installed. Install with: pip install python-docx"
