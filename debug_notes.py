# debug_notes.py
"""
Debugging module to output LLM notes and responses as JSON/TXT files
for troubleshooting the notepad functionality.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NotesDebugger:
    def __init__(self, debug_dir: Optional[str] = None):
        """Initialize the notes debugger with a debug directory."""
        if debug_dir is None:
            # Use Android download folder structure like your reports
            self.debug_dir = "/storage/emulated/0/Download/Debug_Notes"
        else:
            self.debug_dir = debug_dir
        
        # Create debug directory
        os.makedirs(self.debug_dir, exist_ok=True)
        logger.info(f"Debug directory: {self.debug_dir}")

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitize text for use in filename."""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
        sanitized = re.sub(r'[\s\-\.]+', '_', sanitized)
        sanitized = sanitized.strip('_')
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        if not sanitized:
            sanitized = "untitled"
        return sanitized

    def save_source_notes_debug(self, question: str, sources_and_notes: List[Dict[str, Any]]) -> str:
        """Save individual source notes with their original content for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._sanitize_filename(question)}_source_notes_{timestamp}.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        debug_data = {
            "question": question,
            "timestamp": timestamp,
            "sources_debug": sources_and_notes,
            "total_sources": len(sources_and_notes),
            "notes_with_content": len([s for s in sources_and_notes if s.get('note', '').strip()])
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Source notes debug saved: {filepath}")
        return filepath

    def save_synthesis_debug(self, question: str, notes: List[str], final_answer: str, 
                           synthesis_prompt: str) -> str:
        """Save synthesis debugging information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._sanitize_filename(question)}_synthesis_debug_{timestamp}.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        debug_data = {
            "question": question,
            "timestamp": timestamp,
            "input_notes": notes,
            "synthesis_prompt": synthesis_prompt,
            "final_answer": final_answer,
            "notes_count": len(notes),
            "notes_total_chars": sum(len(note) for note in notes),
            "answer_chars": len(final_answer)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Synthesis debug saved: {filepath}")
        return filepath

    def save_raw_sources_debug(self, question: str, raw_sources: List[Dict[str, Any]]) -> str:
        """Save raw source content before note generation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._sanitize_filename(question)}_raw_sources_{timestamp}.json"
        filepath = os.path.join(self.debug_dir, filename)
        
        debug_data = {
            "question": question,
            "timestamp": timestamp,
            "raw_sources": raw_sources,
            "source_count": len(raw_sources)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Raw sources debug saved: {filepath}")
        return filepath

    def save_prompt_debug(self, question: str, prompts_and_responses: List[Dict[str, str]]) -> str:
        """Save all LLM prompts and responses for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._sanitize_filename(question)}_prompts_debug_{timestamp}.txt"
        filepath = os.path.join(self.debug_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"PROMPT DEBUG SESSION\n")
            f.write(f"Question: {question}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, entry in enumerate(prompts_and_responses, 1):
                f.write(f"PROMPT {i}:\n")
                f.write("-" * 40 + "\n")
                f.write(entry.get('prompt', 'No prompt recorded') + "\n\n")
                
                f.write(f"RESPONSE {i}:\n")
                f.write("-" * 40 + "\n")
                f.write(entry.get('response', 'No response recorded') + "\n\n")
                f.write("=" * 80 + "\n\n")
        
        logger.info(f"Prompt debug saved: {filepath}")
        return filepath

