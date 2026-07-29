from typing import Any
from ddgs import DDGS
from app.tools.base import BaseTool

class WebSearchTool(BaseTool):
    """Tool to search the web using DuckDuckGo."""

    name = "web_search"
    description = "Search the web for information. Returns a list of top results with URLs and summaries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"],
    }

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        if not query:
            return "Error: query parameter is required."

        try:
            results = DDGS().text(query, max_results=3, backend="lite")
            if not results:
                return f"No search results found for '{query}'."
            
            output = []
            for r in results:
                output.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSummary: {r.get('body')}\n")
            return "\n".join(output)
            
        except Exception as e:
            return f"Error executing web search: {e}"
