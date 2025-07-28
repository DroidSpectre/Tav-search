# search_agent.py-v3-enhanced-async
"""
Enhanced SearchAgent with:
1. Asynchronous capabilities and parallel search execution
2. Advanced error handling with retry logic
3. Type hints and comprehensive documentation
4. Enhanced search features with auto_parameters support
5. Credit management integration
6. Improved source summarization with debugging
7. Better conversation context handling
8. Modern Tavily API feature support
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass

from web_search import TavilySearchEngine, TavilyError, RateLimitError
from llm_interface import LLMInterface
from config import (
    MAX_RECURSIVE_SEARCHES,
    MAX_CONVERSATION_HISTORY,
    DEFAULT_SEARCH_CONFIG,
    TavilyConfig,
    LLMConfig
)
from report_saver import ReportSaver
from debug_notes import NotesDebugger
from credit_manager import CreditManager

MAX_PARALLEL_QUERIES: int = 3
MAX_CONTEXT_CHARS: int = 6000
CONTENT_PREVIEW_CHARS: int = 2000
COMPLETENESS_THRESHOLD: int = 7
SIMILARITY_THRESHOLD: float = 0.8

CREDIBLE_DOMAINS: Tuple[str, ...] = (
    # Current domains
    "arxiv.org", "wikipedia.org", "nytimes.com", "nature.com", "acm.org", "ieee.org",
    # Major news wire services
    "reuters.com", "ap.org", "bloomberg.com",
    # International quality news
    "bbc.com", "theguardian.com", "ft.com", "economist.com",
    # U.S. national papers
    "wsj.com", "washingtonpost.com", "usatoday.com",
    # Public media
    "npr.org", "pbs.org", "c-span.org",
    # Quality journalism
    "propublica.org", "theatlantic.com", "newyorker.com",
    # Government sources
    "nih.gov", "cdc.gov", "nasa.gov", "noaa.gov",
    # Academic domains
    "mit.edu", "harvard.edu", "stanford.edu", "berkeley.edu", "ox.ac.uk", "cam.ac.uk",
    # International organizations
    "un.org", "who.int", "worldbank.org", "imf.org", "europa.eu",
    # Scientific journals
    "science.org", "nejm.org", "cell.com", "plos.org"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_word_re = re.compile(r"[A-Za-z0-9]+")

@dataclass
class SearchResult:
    """Enhanced search result container"""
    question: str
    answer: str
    context: str
    source_notes: List[str]
    search_iterations: List[Dict[str, Any]]
    sources_used: int
    total_iterations: int
    credits_used: int
    saved_files: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None

def _tokenize(text: str) -> Set[str]:
    """Extract tokens from text for similarity comparison"""
    return set(_word_re.findall(text.lower()))

def _jaccard(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _credibility_boost(url: str) -> float:
    """Calculate credibility boost for URL based on domain"""
    dom = _domain(url)
    return 0.1 if any(dom.endswith(cd) for cd in CREDIBLE_DOMAINS) else 0.0

class SearchAgent:
    """Enhanced search agent with async capabilities and improved error handling"""
    
    def __init__(
        self,
        tavily_config: Optional[TavilyConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        search_engine: Optional[TavilySearchEngine] = None,
        llm: Optional[LLMInterface] = None,
    ) -> None:
        """
        Initialize SearchAgent with enhanced configuration support.
        
        Args:
            tavily_config: Tavily API configuration
            llm_config: LLM configuration
            search_engine: Custom search engine instance
            llm: Custom LLM interface instance
        """
        self.tavily_config = tavily_config or TavilyConfig()
        self.llm_config = llm_config or LLMConfig()
        
        self.search_engine = search_engine or TavilySearchEngine(
            api_key=self.tavily_config.api_key,
            max_retries=3
        )
        self.llm = llm or LLMInterface(
            model_path=self.llm_config.model_path,
            config=self.llm_config
        )
        
        self.report_saver = ReportSaver()
        self.notes_debugger = NotesDebugger()
        self.credit_manager = CreditManager()
        
        self.conversation_history: List[Dict[str, str]] = []
        self.used_urls: Set[str] = set()
        self.latest_question: str = ""
        self._debug_sources: List[Dict[str, Any]] = []
        
        logger.info("Enhanced SearchAgent initialized")

    def search_and_synthesize(
        self, 
        question: str, 
        use_advanced: bool = False,
        **search_kwargs: Any
    ) -> SearchResult:
        """
        Perform comprehensive search with enhanced features and error handling.
        
        Args:
            question: Search query
            use_advanced: Use advanced search depth (2 credits)
            **search_kwargs: Additional search parameters
            
        Returns:
            SearchResult object containing all search information
            
        Raises:
            TavilyError: If search fails after retries
            RateLimitError: If rate limit is exceeded
        """
        logger.info("Question received: %s", question)
        self.latest_question = question
        
        # Check credit availability before starting
        search_params = self._prepare_search_params(use_advanced, **search_kwargs)
        if not self.credit_manager.can_make_request(search_params):
            error_msg = "Insufficient credits for this search"
            logger.error(error_msg)
            return SearchResult(
                question=question,
                answer=error_msg,
                context="",
                source_notes=[],
                search_iterations=[],
                sources_used=0,
                total_iterations=0,
                credits_used=0,
                error_info={"type": "credit_limit", "message": error_msg}
            )
        
        total_credits = 0
        all_iterations: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        all_notes: List[str] = []
        error_info: Optional[Dict[str, Any]] = None

        try:
            current_queries = [question]
            for iteration in range(MAX_RECURSIVE_SEARCHES):
                logger.info("Iteration %d", iteration + 1)
                start_t = time.time()
                
                search_payloads = current_queries[:MAX_PARALLEL_QUERIES]
                
                try:
                    results = self._parallel_search(search_payloads, **search_params)
                    iteration_credits = self.credit_manager.track_search(search_params)
                    total_credits += iteration_credits
                    
                    all_iterations.extend(results["meta"])
                    
                    # Get both context and notes from results
                    added_context, iteration_notes = self._select_and_format_results(
                        results["results"]
                    )
                    
                    if added_context:
                        context_parts.extend(added_context)
                    if iteration_notes:
                        all_notes.extend(iteration_notes)

                except (TavilyError, RateLimitError) as e:
                    logger.error(f"Search error in iteration {iteration + 1}: {e}")
                    error_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "iteration": iteration + 1
                    }
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in iteration {iteration + 1}: {e}")
                    error_info = {
                        "type": "UnexpectedError",
                        "message": str(e),
                        "iteration": iteration + 1
                    }
                    break

                # Generate answer using notes
                if all_notes:
                    answer = self._synthesize_from_notes(question, all_notes)
                else:
                    # Fallback to traditional method if no notes
                    full_context = self._trim_context("\n\n".join(context_parts))
                    prompt = self.llm.create_search_prompt(
                        question, full_context, self.conversation_history
                    )
                    answer = self.llm.generate_response(prompt)

                # Assess answer quality
                notes_context = "\n\n".join(all_notes) if all_notes else "\n\n".join(context_parts)
                assessment = self.llm.assess_answer_quality(question, answer, notes_context)
                logger.info("Assessment: %s", assessment)

                if (
                    not assessment["needs_more_info"]
                    or assessment["completeness"] >= COMPLETENESS_THRESHOLD
                ):
                    break

                current_queries = self._expand_queries(
                    base_query=assessment["suggested_query"] or question
                )

                logger.debug("Next queries: %s", current_queries)
                if not current_queries:
                    break

                logger.info(
                    "Iteration %d completed in %.2fs", iteration + 1, time.time() - start_t
                )

            # Generate final answer
            if all_notes:
                final_answer = self._synthesize_from_notes(question, all_notes)
            else:
                final_context = self._trim_context("\n\n".join(context_parts))
                final_prompt = self.llm.create_search_prompt(
                    question, final_context, self.conversation_history
                )
                final_answer = self.llm.generate_response(final_prompt)

            self._update_history(question, final_answer)

        except Exception as e:
            logger.error(f"Critical error in search_and_synthesize: {e}")
            final_answer = "I apologize, but I encountered a critical error during the search process."
            error_info = {
                "type": "CriticalError",
                "message": str(e)
            }

        # Prepare result
        final_context = self._trim_context("\n\n".join(context_parts))
        result = SearchResult(
            question=question,
            answer=final_answer,
            context=final_context,
            source_notes=all_notes,
            search_iterations=all_iterations,
            sources_used=len(self.used_urls),
            total_iterations=len(all_iterations),
            credits_used=total_credits,
            error_info=error_info
        )

        # Save debugging information
        self._save_debug_info(question)

        # Save reports
        try:
            saved_files = self.report_saver.save_report_complete(result.__dict__)
            result.saved_files = saved_files
            logger.info(f"Reports saved: {saved_files}")
        except Exception as e:
            logger.error(f"Failed to save reports: {e}")
            result.saved_files = {"error": str(e)}

        return result

    def adaptive_search(self, query: str) -> SearchResult:
        """
        Use Tavily's auto_parameters for intelligent search configuration.
        
        Args:
            query: Search query
            
        Returns:
            SearchResult with optimized parameters
        """
        search_params = {
            "auto_parameters": True,
            "search_depth": self.tavily_config.search_depth,
            "max_results": self.tavily_config.max_results,
            "include_answer": self.tavily_config.include_answer,
            "include_raw_content": self.tavily_config.include_raw_content
        }
        
        return self.search_and_synthesize(query, **search_params)

    def news_search(self, query: str, days: int = 3) -> SearchResult:
        """
        Specialized news search with optimized parameters.
        
        Args:
            query: News search query
            days: Number of days to search back
            
        Returns:
            SearchResult with news-specific configuration
        """
        search_params = {
            "topic": "news",
            "days": days,
            "search_depth": "advanced",  # Better for news
            "include_answer": True
        }
        
        return self.search_and_synthesize(query, use_advanced=True, **search_params)

    def context_search(self, query: str, max_tokens: int = 4000) -> str:
        """
        Get context for RAG applications using Tavily's context endpoint.
        
        Args:
            query: Search query
            max_tokens: Maximum context tokens
            
        Returns:
            Context string suitable for RAG
        """
        try:
            return self.search_engine.get_search_context(
                query=query,
                max_tokens=max_tokens,
                search_depth=self.tavily_config.search_depth
            )
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return ""

    async def async_search_and_synthesize(
        self, 
        question: str, 
        **search_kwargs: Any
    ) -> SearchResult:
        """
        Asynchronous version of search_and_synthesize.
        
        Args:
            question: Search query
            **search_kwargs: Additional search parameters
            
        Returns:
            SearchResult object
        """
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.search_and_synthesize, 
            question, 
            **search_kwargs
        )

    async def batch_search(self, queries: List[str], **search_kwargs: Any) -> List[SearchResult]:
        """
        Execute multiple searches concurrently.
        
        Args:
            queries: List of search queries
            **search_kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        tasks = [
            self.async_search_and_synthesize(query, **search_kwargs) 
            for query in queries
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def multi_iteration_search(
        self, 
        initial_query: str, 
        max_iterations: int = 3
    ) -> SearchResult:
        """
        Implement intelligent multi-iteration search with gap analysis.
        
        Args:
            initial_query: Initial search query
            max_iterations: Maximum search iterations
            
        Returns:
            SearchResult with synthesized information from all iterations
        """
        results = []
        current_query = initial_query
        
        for iteration in range(max_iterations):
            logger.info(f"Multi-iteration search {iteration + 1}/{max_iterations}")
            
            # Use advanced search for better results
            search_result = self.search_and_synthesize(
                current_query,
                use_advanced=True
            )
            
            results.append(search_result)
            
            # Check if we have sufficient information
            if self._is_search_complete(search_result, initial_query):
                break
                
            # Generate follow-up query based on gaps
            current_query = self._generate_followup_query(search_result, initial_query)
            if not current_query:
                break
        
        return self._synthesize_multi_iteration_results(results, initial_query)

    def clear_session(self) -> None:
        """Clear all session data including conversation history and debug info"""
        self.conversation_history.clear()
        self.used_urls.clear()
        self.latest_question = ""
        self._debug_sources.clear()
        logger.info("Session cleared")

    def get_credit_usage_report(self) -> Dict[str, Any]:
        """Get detailed credit usage report"""
        return self.credit_manager.get_usage_report()

    # Private methods (implementation details)
    def _prepare_search_params(self, use_advanced: bool, **kwargs: Any) -> Dict[str, Any]:
        """Prepare search parameters with validation"""
        params = {
            "search_depth": "advanced" if use_advanced else self.tavily_config.search_depth,
            "topic": kwargs.get("topic", self.tavily_config.topic),
            "max_results": kwargs.get("max_results", self.tavily_config.max_results),
            "include_answer": kwargs.get("include_answer", self.tavily_config.include_answer),
            "include_raw_content": kwargs.get("include_raw_content", self.tavily_config.include_raw_content),
            "timeout": kwargs.get("timeout", self.tavily_config.timeout),
        }
        
        # Add optional parameters
        if kwargs.get("days"):
            params["days"] = kwargs["days"]
        if kwargs.get("auto_parameters"):
            params["auto_parameters"] = kwargs["auto_parameters"]
        if kwargs.get("include_domains"):
            params["include_domains"] = kwargs["include_domains"]
        if kwargs.get("exclude_domains"):
            params["exclude_domains"] = kwargs["exclude_domains"]
            
        return params

    def _synthesize_from_notes(self, question: str, notes: List[str]) -> str:
        """Generate answer using only the collected source notes."""
        if not notes:
            return "I apologize, but I couldn't gather sufficient information to answer your question."

        joined_notes = "\n\n".join(notes)
        prompt = f"""You are an expert research assistant. Based on the following notes extracted from various sources, provide a comprehensive answer to the question.

Question: {question}

Source Notes:
{joined_notes}

Instructions:
1. Use ONLY the information provided in the notes above
2. Synthesize the information into a coherent, well-structured answer
3. Do NOT add information not present in the notes
4. If the notes don't provide enough information, clearly state what's missing
5. Be factual and avoid speculation
6. Organize the answer logically with clear explanations

Answer:"""

        try:
            response = self.llm.generate_response(prompt, max_tokens=800, temperature=0.7)
            
            # Save synthesis debug info
            try:
                self.notes_debugger.save_synthesis_debug(
                    question=question,
                    notes=notes,
                    final_answer=response.strip(),
                    synthesis_prompt=prompt
                )
            except Exception as debug_e:
                logger.error(f"Failed to save synthesis debug: {debug_e}")
            
            logger.info("Answer synthesized from notes successfully")
            return response.strip()
        except Exception as e:
            logger.error(f"Note synthesis failed: {e}")
            return "I apologize, but I encountered an error while synthesizing the information from my notes."

    def _summarize_source(self, question: str, title: str, url: str, content: str) -> str:
        """Summarize a single source into key points relevant to the question."""
        prompt = f"""You are a research assistant. Extract and summarize the key points from this source that are relevant to answering the question.

Question: {question}

Source: {title}
URL: {url}
Content: {content[:1800]}

Instructions:
1. Focus only on information that helps answer the question
2. Extract 2-5 key points or facts
3. Be concise but informative
4. Use bullet points for clarity
5. If the source doesn't contain relevant information, say "No relevant information found"

Key Points:
-"""

        # Capture debug information
        debug_info = {
            'prompt': prompt,
            'response': '',
            'source_title': title,
            'source_url': url,
            'content_length': len(content),
            'content_preview': content[:500],
            'question': question
        }

        try:
            response = self.llm.generate_response(prompt, max_tokens=200, temperature=0.5)
            debug_info['response'] = response
            logger.info(f"Source note created for: {title}")
            
            # Store debugging info
            debug_info['note'] = f"Source: {title}\n{response.strip()}"
            self._debug_sources.append(debug_info)
            
            return f"Source: {title}\n{response.strip()}"
        except Exception as e:
            debug_info['response'] = f"ERROR: {str(e)}"
            debug_info['note'] = ""
            logger.error(f"Source summarization failed for {title}: {e}")
            
            # Still save debug info even on error
            self._debug_sources.append(debug_info)
            
            return ""

    def _parallel_search(
        self, queries: List[str], **search_kwargs: Any
    ) -> Dict[str, Any]:
        """Execute multiple searches in parallel with proper error handling"""
        combined_results: List[Dict[str, Any]] = []
        meta_info: List[Dict[str, Any]] = []
        credits = 0

        def _worker(q: str) -> Tuple[str, Dict[str, Any]]:
            try:
                res = self.search_engine.search(q, **search_kwargs)
                return q, res
            except Exception as e:
                logger.error(f"Search failed for query '{q}': {e}")
                return q, {"results": [], "error": str(e)}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(queries), MAX_PARALLEL_QUERIES)
        ) as executor:
            futures = {executor.submit(_worker, q): q for q in queries}
            for future in concurrent.futures.as_completed(futures):
                try:
                    query, res = future.result()
                    depth = search_kwargs.get("search_depth", "basic")
                    iteration_credits = 2 if depth == "advanced" else 1
                    credits += iteration_credits

                    if "error" not in res:
                        combined_results.extend(res.get("results", []))
                    
                    meta_info.append({
                        "query": query,
                        "results_count": len(res.get("results", [])),
                        "tavily_answer": res.get("answer", ""),
                        "response_time": res.get("response_time", 0),
                        "credits_used": iteration_credits,
                        "error": res.get("error")
                    })
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")

        # Deduplicate results
        unique_results = []
        titles_seen: List[str] = []
        for r in combined_results:
            url = r.get("url", "")
            title = r.get("title", "")
            if url in self.used_urls:
                continue
            if any(_jaccard(title, t) >= SIMILARITY_THRESHOLD for t in titles_seen):
                continue
            self.used_urls.add(url)
            titles_seen.append(title)
            unique_results.append(r)

        return {"results": unique_results, "meta": meta_info, "credits": credits}

    def _select_and_format_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Process results to create both context chunks and source notes."""
        ranked = sorted(
            results,
            key=lambda r: -(r.get("score", 0.0) + _credibility_boost(r.get("url", ""))),
        )
        context_chunks: List[str] = []
        notes: List[str] = []

        for r in ranked:
            url = r.get("url", "")
            title = r.get("title", "Untitled")
            content = self.search_engine.get_page_content(r)
            if not content:
                continue

            content = content.strip()
            if not content:
                continue

            # Create context chunk for logging/saving
            content_preview = (
                content[:CONTENT_PREVIEW_CHARS] + "..."
                if len(content) > CONTENT_PREVIEW_CHARS
                else content
            )

            chunk = (
                f"Source: {title} ({url}) "
                f"[Relevance: {r.get('score', 0.0):.2f}]\n"
                f"Content: {content_preview}\n"
            )
            context_chunks.append(chunk)

            # Create source note
            note = self._summarize_source(self.latest_question, title, url, content)
            if note:
                notes.append(note)

        return context_chunks, notes

    def _trim_context(self, context: str) -> str:
        """Trim context to fit within limits"""
        if len(context) <= MAX_CONTEXT_CHARS:
            return context

        parts = context.split("\n\n")
        while parts and len("\n\n".join(parts)) > MAX_CONTEXT_CHARS:
            parts.pop(0)
        return "\n\n".join(parts)

    def _expand_queries(self, base_query: str) -> List[str]:
        """Generate expanded queries based on conversation history and LLM suggestions"""
        expansions: Set[str] = {base_query}

        # Add from conversation history
        for msg in self.conversation_history[-4:]:
            if msg["role"] == "user":
                expansions.add(msg["content"])

        try:
            prompt = (
                "Generate TWO concise, distinct search queries closely related "
                f"to the question: \"{base_query}\".\n"
                "Return them on separate lines without numbering or quotes."
            )
            raw = self.llm.generate_response(prompt, max_tokens=50)
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    expansions.add(line)
        except Exception as e:
            logger.debug("Expansion LLM call failed: %s", e)

        return list(expansions)[:MAX_PARALLEL_QUERIES]

    def _update_history(self, question: str, answer: str) -> None:
        """Update conversation history with size management"""
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        max_len = MAX_CONVERSATION_HISTORY * 2
        if len(self.conversation_history) > max_len:
            self.conversation_history = self.conversation_history[-max_len:]

    def _save_debug_info(self, question: str) -> None:
        """Save debugging information if available"""
        if self._debug_sources:
            try:
                self.notes_debugger.save_source_notes_debug(question, self._debug_sources)
                logger.info("Debug notes saved successfully")
                self._debug_sources.clear()
            except Exception as e:
                logger.error(f"Failed to save debug notes: {e}")

    def _is_search_complete(self, search_result: SearchResult, original_query: str) -> bool:
        """Determine if search has gathered sufficient information"""
        # Simple heuristic - can be enhanced
        return (
            len(search_result.source_notes) >= 3 and
            len(search_result.answer) > 200 and
            search_result.error_info is None
        )

    def _generate_followup_query(self, search_result: SearchResult, original_query: str) -> str:
        """Generate follow-up query based on gaps in current results"""
        try:
            prompt = f"""Based on the search results for "{original_query}", identify information gaps and suggest a specific follow-up search query.

Current answer: {search_result.answer[:500]}...

Generate ONE specific search query to fill the most important information gap. Be concise:"""
            
            return self.llm.generate_response(prompt, max_tokens=30).strip()
        except Exception as e:
            logger.error(f"Follow-up query generation failed: {e}")
            return ""

    def _synthesize_multi_iteration_results(
        self, 
        results: List[SearchResult], 
        original_query: str
    ) -> SearchResult:
        """Synthesize results from multiple search iterations"""
        # Combine all notes and context
        all_notes = []
        all_context_parts = []
        total_credits = 0
        all_iterations = []
        
        for result in results:
            all_notes.extend(result.source_notes)
            if result.context:
                all_context_parts.append(result.context)
            total_credits += result.credits_used
            all_iterations.extend(result.search_iterations)
        
        # Generate final synthesized answer
        if all_notes:
            final_answer = self._synthesize_from_notes(original_query, all_notes)
        else:
            final_answer = "Unable to synthesize information from multiple search iterations."
        
        return SearchResult(
            question=original_query,
            answer=final_answer,
            context="\n\n".join(all_context_parts),
            source_notes=all_notes,
            search_iterations=all_iterations,
            sources_used=len(self.used_urls),
            total_iterations=len(results),
            credits_used=total_credits
        )
