# web_search.py
import requests
import time
import logging
from typing import Dict, List, Optional, Any
from config import TAVILY_API_KEY, DEFAULT_SEARCH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TavilySearchEngine:
    def __init__(self, api_key: str = TAVILY_API_KEY):
        if not api_key:
            raise ValueError("Tavily API key is required")
        
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"
        self.session = requests.Session()
        
        # Correct authentication header format
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform Tavily search with comprehensive parameter support
        """
        # Merge default config with provided kwargs
        search_params = {**DEFAULT_SEARCH_CONFIG, **kwargs}
        
        # Build payload - NO api_key in body!
        payload = {
            "query": query,
            "search_depth": search_params["search_depth"],
            "topic": search_params["topic"],
            "max_results": search_params["max_results"],
            "include_answer": search_params["include_answer"],
            "include_raw_content": search_params["include_raw_content"],
            "include_images": search_params["include_images"],
            "include_image_descriptions": search_params["include_image_descriptions"]
        }
        
        # Add optional parameters if specified
        if search_params["topic"] == "news" and search_params.get("days"):
            payload["days"] = search_params["days"]
        
        if search_params["include_domains"]:
            payload["include_domains"] = search_params["include_domains"]
        
        if search_params["exclude_domains"]:
            payload["exclude_domains"] = search_params["exclude_domains"]
        
        return self._make_request("/search", payload, search_params["timeout"])
    
    def _make_request(self, endpoint: str, payload: Dict, timeout: int = 30) -> Dict[str, Any]:
        """
        Make API request with proper error handling and retry logic
        """
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1}/{max_retries})")
                response = self.session.post(url, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Request successful. Response time: {data.get('response_time', 'N/A')}s")
                    return data
                elif response.status_code == 401:
                    logger.error("Authentication failed. Check your API key.")
                    raise ValueError("Invalid Tavily API key")
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return {"results": [], "query": payload.get("query", ""), "error": "Max retries exceeded"}
    
    def get_page_content(self, result: Dict[str, Any]) -> str:
        """
        Extract content from search result
        """
        # Try raw_content first, then content
        content = result.get("raw_content", "") or result.get("content", "")
        return content
    
    def extract_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Use Tavily's extract endpoint for specific URLs
        """
        payload = {"urls": urls}
        return self._make_request("/extract", payload)
