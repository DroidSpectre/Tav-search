# search_agent.py-v2c-reporter-notepad-debug
"""
Upgraded SearchAgent with:
1. Parallel search execution
2. Simple semantic-similarity deduplication
3. Dynamic context window management
4. Credibility-biased ranking
5. Automatic query expansion
6. Conversation-aware search refinement
7. Saves response reports and Tavily Quick answers as txt files
8. Per-source notepad summarization to reduce hallucination
9. Debug output for troubleshooting note generation

Designed to run out-of-the-box in Termux (Android). No heavyweight
dependencies are required; only the Python standard library and the
existing project modules are used.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any, Set, Tuple

from web_search import TavilySearchEngine
from llm_interface import LLMInterface
from config import (
    MAX_RECURSIVE_SEARCHES,
    MAX_CONVERSATION_HISTORY,
    DEFAULT_SEARCH_CONFIG,
)
from report_saver import ReportSaver
from debug_notes import NotesDebugger  # <-- NEW IMPORT

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_PARALLEL_QUERIES: int = 3
MAX_CONTEXT_CHARS: int = 6000
CONTENT_PREVIEW_CHARS: int = 2000
COMPLETENESS_THRESHOLD: int = 7
SIMILARITY_THRESHOLD: float = 0.8
CREDIBLE_DOMAINS: Tuple[str, ...] = (
    "arxiv.org",
    "wikipedia.org",
    "nytimes.com",
    "nature.com",
    "acm.org",
    "ieee.org",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_word_re = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    return set(_word_re.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _credibility_boost(url: str) -> float:
    dom = _domain(url)
    return 0.1 if any(dom.endswith(cd) for cd in CREDIBLE_DOMAINS) else 0.0


class SearchAgent:
    def __init__(
        self,
        search_engine: Optional[TavilySearchEngine] = None,
        llm: Optional[LLMInterface] = None,
    ):
        self.search_engine = search_engine or TavilySearchEngine()
        self.llm = llm or LLMInterface()
        self.report_saver = ReportSaver()
        self.notes_debugger = NotesDebugger()  # <-- NEW LINE
        self.conversation_history: List[Dict[str, str]] = []
        self.used_urls: Set[str] = set()
        self.latest_question: str = ""  # Track current question for notepad
        logger.info("SearchAgent initialized")

    def search_and_synthesize(self, question: str, **search_kwargs) -> Dict[str, Any]:
        logger.info("Question received: %s", question)
        self.latest_question = question  # Store for notepad usage
        
        total_credits = 0
        all_iterations: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        all_notes: List[str] = []  # Collect all source notes

        current_queries = [question]
        for iteration in range(MAX_RECURSIVE_SEARCHES):
            logger.info("Iteration %d", iteration + 1)
            start_t = time.time()
            search_payloads = current_queries[:MAX_PARALLEL_QUERIES]
            results = self._parallel_search(search_payloads, **search_kwargs)

            all_iterations.extend(results["meta"])
            total_credits += results["credits"]

            # Get both context and notes from results
            added_context, iteration_notes = self._select_and_format_results(results["results"])
            if added_context:
                context_parts.extend(added_context)
            if iteration_notes:
                all_notes.extend(iteration_notes)

            # Use notes for answer generation instead of raw context
            if all_notes:
                answer = self._synthesize_from_notes(question, all_notes)
            else:
                # Fallback to traditional method if no notes
                full_context = self._trim_context("\n\n".join(context_parts))
                prompt = self.llm.create_search_prompt(
                    question, full_context, self.conversation_history
                )
                answer = self.llm.generate_response(prompt)

            # Assess answer quality based on notes
            notes_context = "\n\n".join(all_notes) if all_notes else "\n\n".join(context_parts)
            assessment = self.llm.assess_answer_quality(
                question, answer, notes_context
            )
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

        # Generate final answer using accumulated notes
        if all_notes:
            final_answer = self._synthesize_from_notes(question, all_notes)
        else:
            # Fallback to traditional method
            final_context = self._trim_context("\n\n".join(context_parts))
            final_prompt = self.llm.create_search_prompt(
                question, final_context, self.conversation_history
            )
            final_answer = self.llm.generate_response(final_prompt)

        self._update_history(question, final_answer)

        # Prepare result with both context and notes
        final_context = self._trim_context("\n\n".join(context_parts))
        result = {
            "question": question,
            "answer": final_answer,
            "context": final_context,
            "source_notes": all_notes,  # Add notes to result
            "search_iterations": all_iterations,
            "sources_used": len(self.used_urls),
            "total_iterations": len(all_iterations),
            "credits_used": total_credits,
        }

        # <-- NEW: Save debugging information
        if hasattr(self, '_debug_sources') and self._debug_sources:
            try:
                self.notes_debugger.save_source_notes_debug(question, self._debug_sources)
                logger.info("Debug notes saved successfully")
                # Clear for next search
                delattr(self, '_debug_sources')
            except Exception as e:
                logger.error(f"Failed to save debug notes: {e}")

        try:
            saved_files = self.report_saver.save_report_complete(result)
            result["saved_files"] = saved_files
            logger.info(f"Reports saved: {saved_files}")
        except Exception as e:
            logger.error(f"Failed to save reports: {e}")
            result["saved_files"] = {"error": str(e)}

        return result

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
            
            # <-- NEW: Debug synthesis details
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

        # <-- NEW: Capture debug information
        debug_info = {
            'prompt': prompt,
            'response': '',
            'source_title': title,
            'source_url': url,
            'content_length': len(content),
            'content_preview': content[:500],  # First 500 chars for debugging
            'question': question
        }

        try:
            response = self.llm.generate_response(prompt, max_tokens=200, temperature=0.5)
            debug_info['response'] = response
            logger.info(f"Source note created for: {title}")
            
            # <-- NEW: Store debugging info
            if not hasattr(self, '_debug_sources'):
                self._debug_sources = []
            
            # Add the generated note to debug info
            debug_info['note'] = f"Source: {title}\n{response.strip()}"
            self._debug_sources.append(debug_info)
            
            return f"Source: {title}\n{response.strip()}"
        except Exception as e:
            debug_info['response'] = f"ERROR: {str(e)}"
            debug_info['note'] = ""
            logger.error(f"Source summarization failed for {title}: {e}")
            
            # <-- NEW: Still save debug info even on error
            if not hasattr(self, '_debug_sources'):
                self._debug_sources = []
            self._debug_sources.append(debug_info)
            
            return ""

    def clear_session(self) -> None:
        self.conversation_history.clear()
        self.used_urls.clear()
        self.latest_question = ""
        # <-- NEW: Clear debug data
        if hasattr(self, '_debug_sources'):
            delattr(self, '_debug_sources')
        logger.info("Session cleared")

    def _parallel_search(
        self, queries: List[str], **search_kwargs
    ) -> Dict[str, Any]:
        combined_results: List[Dict[str, Any]] = []
        meta_info: List[Dict[str, Any]] = []
        credits = 0

        def _worker(q: str) -> Tuple[str, Dict[str, Any]]:
            res = self.search_engine.search(q, **search_kwargs)
            return q, res

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(queries), MAX_PARALLEL_QUERIES)
        ) as executor:
            futures = {executor.submit(_worker, q): q for q in queries}
            for future in concurrent.futures.as_completed(futures):
                query, res = future.result()
                depth = search_kwargs.get("search_depth", "basic")
                credits += 2 if depth == "advanced" else 1

                combined_results.extend(res.get("results", []))
                meta_info.append(
                    {
                        "query": query,
                        "results_count": len(res.get("results", [])),
                        "tavily_answer": res.get("answer", ""),
                        "response_time": res.get("response_time", 0),
                        "credits_used": 2 if depth == "advanced" else 1,
                    }
                )

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

    def _select_and_format_results(self, results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
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
        if len(context) <= MAX_CONTEXT_CHARS:
            return context

        parts = context.split("\n\n")
        while parts and len("\n\n".join(parts)) > MAX_CONTEXT_CHARS:
            parts.pop(0)
        return "\n\n".join(parts)

    def _expand_queries(self, base_query: str) -> List[str]:
        expansions: Set[str] = {base_query}

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
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        max_len = MAX_CONVERSATION_HISTORY * 2
        if len(self.conversation_history) > max_len:
            self.conversation_history = self.conversation_history[-max_len:]

