"""Tool to fetch and read the full content of a web page."""
import re
import httpx
from app.tools.base import BaseTool


class URLScraperTool(BaseTool):
    """Fetches a URL and returns its readable text content."""

    name = "url_scrape"
    description = (
        "Fetch and read the full text content of a web page URL. "
        "Use after web_search when you need to read the actual content of a specific page, "
        "not just the search summary. Also use to verify information from specific sources."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the web page to read.",
            }
        },
        "required": ["url"],
    }

    def execute(self, **kwargs) -> str:
        url = kwargs.get("url", "").strip()
        if not url:
            return "Error: url parameter is required."
        if not url.startswith("http"):
            url = "https://" + url
        try:
            response = httpx.get(
                url,
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgentOS/1.0)"},
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text" not in content_type and "html" not in content_type:
                return f"URL returned non-text content ({content_type}). Cannot extract readable text."
            text = response.text
            text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            for ent, ch in {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ", "&quot;": '"'}.items():
                text = text.replace(ent, ch)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return f"No readable text found at {url}."
            if len(text) > 4000:
                text = text[:4000] + "\n\n[Content truncated at 4000 characters]"
            return f"Content from {url}:\n\n{text}"
        except httpx.TimeoutException:
            return f"Timeout while reading {url}."
        except httpx.HTTPStatusError as e:
            return f"HTTP {e.response.status_code} error reading {url}."
        except Exception as e:
            return f"Error reading {url}: {e}"
