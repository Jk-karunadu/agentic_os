"""Tool to fetch and extract text content from any URL."""

import httpx
from app.tools.base import BaseTool


class URLReaderTool(BaseTool):
    """Fetches a webpage and extracts its main text content."""

    name = "read_url"
    description = (
        "Fetch the content of a webpage URL and return its text. "
        "Use this to read articles, documentation, or any web page found via search. "
        "Returns the extracted text content (up to 4000 characters)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the webpage to read.",
            }
        },
        "required": ["url"],
    }

    def execute(self, **kwargs) -> str:
        url = kwargs.get("url")
        if not url:
            return "Error: url parameter is required."

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()

            html = response.text

            # Strip HTML tags to get plain text
            import re
            # Remove script and style blocks entirely
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # Remove all HTML tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate to avoid blowing up the LLM context window
            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated at 4000 characters]"

            if not text:
                return f"No readable text content found at {url}."

            return f"Content from {url}:\n\n{text}"

        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} when fetching {url}."
        except Exception as e:
            return f"Error reading URL: {e}"
