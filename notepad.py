# notepad.py
"""
Source Notepad Module

Part of the LLM-powered search agent.
Performs per-source note summarization and final answer synthesis
based only on the collected source notes.

Designed to work with existing llm_interface.py and search_agent.py.
"""

import logging
from typing import List, Dict, Optional
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Notepad:
    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm or LLMInterface()
        logger.info("Notepad initialized")

    def summarize_source(self, question: str, source_title: str, source_url: str, source_content: str) -> str:
        """
        Summarize a single source for a given question and return the note.
        """
        prompt = f"""
You are an assistant tasked with examining a source and summarizing relevant key points.
Carefully read the source content and summarize ONLY the information that directly helps answer the user's question.

Question:
{question}

Source:
Title: {source_title}
URL: {source_url}
Content:
{source_content[:2000]}  # Limit to avoid overload

Instructions:
- Summarize relevant points clearly and objectively.
- Include any key facts, code tips, or caveats.
- Do NOT include unrelated information or speculation.
- Be concise. Include 2–5 bullet points max.

Answer format:
- [Bullet point 1]
- [Bullet point 2]
...
"""
        try:
            response = self.llm.generate_response(prompt, max_tokens=300, temperature=0.5)
            logger.info("Generated note for source: %s", source_title)
            return response.strip()
        except Exception as e:
            logger.error(f"Source summarization failed for {source_title}: {e}")
            return ""

    def synthesize_answer(self, question: str, summary_notes: List[str], history: Optional[List[Dict]] = None) -> str:
        """
        Generate a final answer to the question using the list of notes only.
        """
        joined_notes = "\n".join(summary_notes)
        prompt = """
You are an expert assistant. Write a clear and fact-based answer to the following question.
Use only the notes provided, which are bullet-point summaries extracted from sources. Do NOT add information that is not evident in the notes.

Question:
{question}

Notes:
{notes}

Instructions:
- Write a coherent and complete answer to the question.
- Use formal, professional tone.
- Cite or attribute high-confidence facts, if needed.
- Do not extract quotes, just synthesize.

Final Answer:""".format(question=question, notes=joined_notes)

        try:
            response = self.llm.generate_response(prompt, max_tokens=600, temperature=0.7)
            logger.info("Synthesis generated successfully.")
            return response.strip()
        except Exception as e:
            logger.error(f"Synthesis generation failed: {e}")
            return "I could not synthesize an answer at this time."
