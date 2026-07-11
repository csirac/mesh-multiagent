
from typing import Optional, List, Dict, Any
import json

class ExaSearchClient:
    """
    Wrapper for Exa search API interactions.
    Handles search, full content fetching, and caching.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client: Optional[Any] = None
        self.last_results: List[Any] = []
        
        if api_key:
            self._initialize(api_key)
    
    def _initialize(self, api_key: str):
        """Initialize Exa client with API key."""
        try:
            from exa_py import Exa
            self.client = Exa(api_key)
        except Exception as e:
            self.client = None
            print( e )
    
    def is_available(self) -> bool:
        """Check if Exa client is initialized."""
        return self.client is not None
    
    def search(self, query: str, num_results: int = 8) -> str:
        """
        Perform lightweight Exa search.
        Returns formatted string with results.
        """
        if not self.is_available():
            return "Exa API not available (no EXA_API_KEY)."
        
        try:
            result = self.client.search(query=query, num_results=num_results)
        except Exception as e:
            return f"Exa search error: {e}"
        
        self.last_results = result.results or []
        
        if not self.last_results:
            return "No results found."
        
        return self._format_results(self.last_results)
    
    def fetch_full_content(self, index: int) -> str:
        """
        Fetch full content for result at given index.
        Index is 1-based (matching user-facing display).
        """
        if not self.is_available():
            return "Exa API not available (no EXA_API_KEY)."
        
        if not self.last_results:
            return "No cached Exa results. Run /exa QUERY first."
        
        if index < 1 or index > len(self.last_results):
            return f"Index {index} is out of range (1–{len(self.last_results)})."
        
        result = self.last_results[index - 1]
        
        try:
            resp = self.client.get_contents([result.url])
        except Exception as e:
            return f"Exa full-content error: {e}"
        
        if not resp.results:
            return "No full content available."
        
        page = resp.results[0]
        title = page.title or result.title or "(no title)"
        body = page.text.strip()
         
        return f"{title}\n{page.url}\n\n{body}"
    
    def fetch_full_content_by_url(self, url: str) -> str:
        """Fetch full content for a specific URL (for tool calls)."""
        if not self.is_available():
            return "Exa API not available (no EXA_API_KEY)."
        
        try:
            resp = self.client.get_contents([url])
        except Exception as e:
            return f"Exa full-content error: {e}"
        
        if not resp.results:
            return "No full content available."
        
        page = resp.results[0]
        title = page.title or "(no title)"
        body = (page.text or "").strip()
        
        return f"{title}\n{url}\n\n{body}"
    
    @staticmethod
    def _format_results(results: List[Any]) -> str:
        """Format search results as human-readable text."""
        lines = []
        for i, r in enumerate(results, start=1):
            snippet = (r.text or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"{i}. {r.title}\n   {r.url}\n   {snippet}\n")
        return "\n".join(lines)
    
    def dispatch_tool_call(self, tool_call: Any) -> tuple[Dict[str, str], bool]:
        """
        Handle Exa-related tool calls from LLM.
        
        Returns:
            (tool_response_message, is_error)
        """
        try:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_id = tool_call.id
        except AttributeError:
            try:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                call_id = tool_call.get("id", "")
            except Exception:
                return ({
                    "role": "tool",
                    "tool_call_id": "",
                    "content": "Malformed tool call: missing fields."
                }, True)
        
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception as e:
            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Invalid JSON in tool arguments: {e}. Arguments: {raw_args}"
            }, True)
        
        if name == "exa_search":
            query = args.get("query", "")
            num_results_raw = args.get("num_results", 8)
            try:
                num_results_int = int(num_results_raw)
            except (TypeError, ValueError):
                num_results_int = 8  # or raise/log a clear error

            num_results = min(num_results_int, 12)
            output = self.search(query, num_results)
        
        elif name == "exa_fetch_full":
            url = args.get("url", "")
            if not url:
                output = "Missing required parameter: url"
            else:
                output = self.fetch_full_content_by_url(url)
        
        else:
            output = f"Unknown tool: {name}"
        
        return ({
            "role": "tool",
            "tool_call_id": call_id,
            "content": output
        }, False)
