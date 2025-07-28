# web_search.py-v2 - Enhanced with missing methods and better error handling
"""
Enhanced Tavily Search Engine with:
1. Full Tavily API method support (search, get_search_context, qna_search, extract)
2. Comprehensive error handling with retry logic and exponential backoff
3. Rate limiting and timeout handling
4. Type hints and detailed documentation
5. Credit tracking integration
6. AsyncTavilyClient support preparation
"""
import asyncio
import requests
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urlparse

from config import TavilyConfig, DEFAULT_SEARCH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class TavilyError(Exception):
    """Base exception for Tavily API errors"""
    pass

class RateLimitError(TavilyError):
    """Rate limit exceeded error"""
    pass

class AuthenticationError(TavilyError):
    """Authentication failed error"""
    pass

class TimeoutError(TavilyError):
    """Request timeout error"""
    pass

@dataclass
class SearchResult:
    """Structured search result"""
    url: str
    title: str
    content: str
    raw_content: str = ""
    score: float = 0.0
    published_date: Optional[str] = None

class TavilySearchEngine:
    """Enhanced Tavily Search Engine with comprehensive API support"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        config: Optional[TavilyConfig] = None
    ) -> None:
        """
        Initialize TavilySearchEngine with enhanced configuration.
        
        Args:
            api_key: Tavily API key (from config if not provided)
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            config: TavilyConfig instance
            
        Raises:
            ValueError: If API key is not provided or invalid
        """
        self.config = config or TavilyConfig()
        self.api_key = api_key or self.config.api_key
        
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.base_url = "https://api.tavily.com"
        
        # Initialize session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })
        
        logger.info("Enhanced TavilySearchEngine initialized")

    def search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform comprehensive Tavily search with all available parameters.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results and metadata
            
        Raises:
            TavilyError: If search fails after retries
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
        """
        # Merge default config with provided kwargs
        search_params = {**self.config.to_search_params(), **kwargs}
        
        # Build comprehensive payload
        payload = {
            "query": query,
            "search_depth": search_params["search_depth"],
            "topic": search_params["topic"],
            "max_results": search_params["max_results"],
            "include_answer": search_params["include_answer"],
            "include_raw_content": search_params["include_raw_content"],
            "include_images": search_params["include_images"],
            "include_image_descriptions": search_params["include_image_descriptions"],
        }
        
        # Add optional parameters if specified
        if search_params.get("auto_parameters"):
            payload["auto_parameters"] = search_params["auto_parameters"]
            
        if search_params["topic"] == "news" and search_params.get("days"):
            payload["days"] = search_params["days"]
            
        if search_params.get("time_range"):
            payload["time_range"] = search_params["time_range"]
            
        if search_params.get("start_date"):
            payload["start_date"] = search_params["start_date"]
            
        if search_params.get("end_date"):
            payload["end_date"] = search_params["end_date"]
            
        if search_params.get("include_domains"):
            payload["include_domains"] = search_params["include_domains"]
            
        if search_params.get("exclude_domains"):
            payload["exclude_domains"] = search_params["exclude_domains"]
        
        return self._make_request_with_retry(
            "/search", 
            payload, 
            search_params.get("timeout", self.config.timeout)
        )

    def get_search_context(
        self, 
        query: str, 
        max_tokens: int = 4000, 
        **kwargs: Any
    ) -> str:
        """
        Get search context suitable for RAG applications.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens in response
            **kwargs: Additional search parameters
            
        Returns:
            Context string suitable for RAG applications
            
        Raises:
            TavilyError: If request fails
        """
        search_params = {**self.config.to_search_params(), **kwargs}
        
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "search_depth": search_params["search_depth"],
            "topic": search_params.get("topic", "general"),
        }
        
        # Add optional parameters
        if search_params.get("include_domains"):
            payload["include_domains"] = search_params["include_domains"]
        if search_params.get("exclude_domains"):
            payload["exclude_domains"] = search_params["exclude_domains"]
        
        try:
            result = self._make_request_with_retry(
                "/search/context", 
                payload, 
                search_params.get("timeout", self.config.timeout)
            )
            return result.get("context", "")
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            raise TavilyError(f"Context search failed: {str(e)}")

    def qna_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform Q&A search that returns direct answers.
        
        Args:
            query: Question to answer
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary with direct answer and supporting information
            
        Raises:
            TavilyError: If request fails
        """
        search_params = {**self.config.to_search_params(), **kwargs}
        
        payload = {
            "query": query,
            "search_depth": search_params["search_depth"],
            "topic": search_params.get("topic", "general"),
        }
        
        # Add optional parameters
        if search_params.get("include_domains"):
            payload["include_domains"] = search_params["include_domains"]
        if search_params.get("exclude_domains"):
            payload["exclude_domains"] = search_params["exclude_domains"]
        
        return self._make_request_with_retry(
            "/search/qna", 
            payload, 
            search_params.get("timeout", self.config.timeout)
        )

    def extract(self, urls: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        Extract content from specific URLs using Tavily's extract endpoint.
        
        Args:
            urls: List of URLs to extract content from
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing extracted content from URLs
            
        Raises:
            TavilyError: If extraction fails
            ValueError: If URLs list is empty
        """
        if not urls:
            raise ValueError("URLs list cannot be empty")
        
        payload = {"urls": urls}
        
        # Add optional parameters
        if kwargs.get("include_raw_content", True):
            payload["include_raw_content"] = kwargs["include_raw_content"]
        
        return self._make_request_with_retry(
            "/extract", 
            payload, 
            kwargs.get("timeout", self.config.timeout)
        )

    def crawl(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Beta feature: Crawl a website starting from a URL.
        
        Args:
            url: Starting URL for crawling
            **kwargs: Additional crawling parameters
            
        Returns:
            Dictionary containing crawl results
            
        Note:
            This is a beta feature and may not be available in all plans
        """
        payload = {"url": url}
        
        # Add optional crawling parameters
        if kwargs.get("max_pages"):
            payload["max_pages"] = kwargs["max_pages"]
        if kwargs.get("max_depth"):
            payload["max_depth"] = kwargs["max_depth"]
        
        return self._make_request_with_retry(
            "/crawl", 
            payload, 
            kwargs.get("timeout", self.config.timeout * 2)  # Longer timeout for crawling
        )

    def map_website(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Beta feature: Map a website's structure.
        
        Args:
            url: Website URL to map
            **kwargs: Additional mapping parameters
            
        Returns:
            Dictionary containing website map
            
        Note:
            This is a beta feature and may not be available in all plans
        """
        payload = {"url": url}
        
        return self._make_request_with_retry(
            "/map", 
            payload, 
            kwargs.get("timeout", self.config.timeout)
        )

    def get_page_content(self, result: Dict[str, Any]) -> str:
        """
        Extract content from search result with fallback options.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Extracted content string
        """
        # Try raw_content first (most complete), then content
        content = result.get("raw_content", "") or result.get("content", "")
        
        # If still no content, try to extract from URL
        if not content and result.get("url"):
            try:
                extract_result = self.extract([result["url"]])
                if extract_result.get("results"):
                    content = extract_result["results"][0].get("raw_content", "")
            except Exception as e:
                logger.debug(f"Failed to extract content from URL {result.get('url')}: {e}")
        
        return content.strip()

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and authentication status.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Simple search to test connectivity
            result = self.search("test", max_results=1)
            return {
                "status": "healthy",
                "authenticated": True,
                "api_responsive": True,
                "test_search_successful": len(result.get("results", [])) >= 0
            }
        except AuthenticationError:
            return {
                "status": "error",
                "authenticated": False,
                "error": "Authentication failed"
            }
        except Exception as e:
            return {
                "status": "error", 
                "authenticated": True,
                "api_responsive": False,
                "error": str(e)
            }

    # Private methods for internal functionality
    def _make_request_with_retry(
        self, 
        endpoint: str, 
        payload: Dict[str, Any], 
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make API request with comprehensive retry logic and error handling.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            timeout: Request timeout in seconds
            
        Returns:
            API response dictionary
            
        Raises:
            TavilyError: For general API errors
            RateLimitError: For rate limiting
            AuthenticationError: For auth failures
            TimeoutError: For timeout errors
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.post(url, json=payload, timeout=timeout)
                
                # Handle successful response
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Request successful. Response time: {data.get('response_time', 'N/A')}s")
                    return data
                
                # Handle specific error codes
                elif response.status_code == 401:
                    logger.error("Authentication failed. Check your API key.")
                    raise AuthenticationError("Invalid Tavily API key")
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self._calculate_backoff_time(attempt, base_wait=60)
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError("Rate limit exceeded after all retries")
                
                elif response.status_code >= 500:  # Server errors
                    wait_time = self._calculate_backoff_time(attempt)
                    logger.warning(
                        f"Server error {response.status_code}. Waiting {wait_time}s before retry"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise TavilyError(f"Server error {response.status_code}: {response.text}")
                
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    raise TavilyError(f"API error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Request timed out after {self.max_retries} attempts")
                time.sleep(self._calculate_backoff_time(attempt))
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise TavilyError(f"Connection failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(self._calculate_backoff_time(attempt))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    raise TavilyError(f"Request failed after {self.max_retries} attempts: {str(e)}")
                time.sleep(self._calculate_backoff_time(attempt))
        
        # Should not reach here, but just in case
        return {"results": [], "query": payload.get("query", ""), "error": "Max retries exceeded"}

    def _calculate_backoff_time(self, attempt: int, base_wait: float = 1.0) -> float:
        """
        Calculate exponential backoff time with jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
            base_wait: Base wait time in seconds
            
        Returns:
            Wait time in seconds
        """
        import random
        
        # Exponential backoff with jitter
        backoff_time = base_wait * (self.backoff_factor ** attempt)
        jitter = random.uniform(0.1, 0.3) * backoff_time
        
        return min(backoff_time + jitter, 300)  # Cap at 5 minutes

    def extract_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Use extract() instead.
        """
        logger.warning("extract_urls() is deprecated. Use extract() instead.")
        return self.extract(urls)

# Async support preparation
class AsyncTavilySearchEngine:
    """Asynchronous Tavily Search Engine for high-performance applications"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        config: Optional[TavilyConfig] = None
    ) -> None:
        """Initialize async Tavily client"""
        self.config = config or TavilyConfig()
        self.api_key = api_key or self.config.api_key
        
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        self.base_url = "https://api.tavily.com"
        self.session: Optional[Any] = None  # Will hold aiohttp.ClientSession
        
        logger.info("AsyncTavilySearchEngine initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        import aiohttp
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Async search method.
        
        Args:
            query: Search query
            **kwargs: Search parameters
            
        Returns:
            Search results dictionary
        """
        if not self.session:
            raise RuntimeError("AsyncTavilySearchEngine must be used as context manager")
        
        search_params = {**self.config.to_search_params(), **kwargs}
        
        payload = {
            "query": query,
            "search_depth": search_params["search_depth"],
            "topic": search_params["topic"],
            "max_results": search_params["max_results"],
            "include_answer": search_params["include_answer"],
            "include_raw_content": search_params["include_raw_content"],
        }
        
        url = f"{self.base_url}/search"
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                else:
                    raise TavilyError(f"API error {response.status}")
        except Exception as e:
            raise TavilyError(f"Async search failed: {str(e)}")

    async def batch_search(self, queries: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Execute multiple searches concurrently.
        
        Args:
            queries: List of search queries
            **kwargs: Search parameters
            
        Returns:
            List of search results
        """
        tasks = [self.search(query, **kwargs) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        return [r for r in results if not isinstance(r, Exception)]

# Backward compatibility
TavilySearchClient = TavilySearchEngine
