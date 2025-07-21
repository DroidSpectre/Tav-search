# report_saver.py
"""
Report Saver Module for LLM-Powered Search App

This module handles saving search reports and Tavily Quick Answers as text files
in the device's download folder under "Search Reports" subdirectory.

Compatible with Termux on Android.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ReportSaver:
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the ReportSaver with a base directory.
        
        Args:
            base_path: Custom base path. If None, uses Android download folder.
        """
        if base_path is None:
            # Default Android download folder path in Termux
            self.base_path = "/storage/emulated/0/Download/Search Reports"
        else:
            self.base_path = base_path
        
        # Create the directory if it doesn't exist
        self._ensure_directory_exists()
        logger.info(f"ReportSaver initialized with path: {self.base_path}")
    
    def _ensure_directory_exists(self):
        """Create the Search Reports directory if it doesn't exist."""
        try:
            os.makedirs(self.base_path, exist_ok=True)
            logger.info(f"Directory ensured: {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {self.base_path}: {e}")
            raise
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Convert text to a safe filename by removing/replacing invalid characters.
        
        Args:
            text: The text to convert to filename
            max_length: Maximum length of the filename
            
        Returns:
            Sanitized filename string
        """
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r'[\s\-\.]+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        # Ensure it's not empty
        if not sanitized:
            sanitized = "untitled"
        
        return sanitized
    
    def _generate_filename(self, question: str, is_tavily_answer: bool = False) -> str:
        """
        Generate a filename based on the question/topic.
        
        Args:
            question: The original question/query
            is_tavily_answer: Whether this is a Tavily Quick Answer
            
        Returns:
            Generated filename with timestamp
        """
        # Create base name from question
        base_name = self._sanitize_filename(question)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add appropriate tag
        if is_tavily_answer:
            filename = f"{base_name}_TQA_{timestamp}.txt"
        else:
            filename = f"{base_name}_{timestamp}.txt"
        
        return filename
    
    def save_full_report(self, report_data: Dict[str, Any]) -> str:
        """
        Save a full search report to a text file.
        
        Args:
            report_data: Dictionary containing report information
            
        Returns:
            Full path to the saved file
        """
        try:
            question = report_data.get('question', 'Unknown Query')
            answer = report_data.get('answer', 'No answer available')
            context = report_data.get('context', '')
            iterations = report_data.get('search_iterations', [])
            stats = {
                'sources_used': report_data.get('sources_used', 0),
                'total_iterations': report_data.get('total_iterations', 0),
                'credits_used': report_data.get('credits_used', 0)
            }
            
            # Generate filename
            filename = self._generate_filename(question)
            filepath = os.path.join(self.base_path, filename)
            
            # Create report content
            report_content = self._format_full_report(
                question, answer, context, iterations, stats
            )
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Full report saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save full report: {e}")
            raise
    
    def save_tavily_answers(self, report_data: Dict[str, Any]) -> List[str]:
        """
        Save Tavily Quick Answers from search iterations to separate text files.
        
        Args:
            report_data: Dictionary containing report information
            
        Returns:
            List of full paths to saved Tavily answer files
        """
        saved_files = []
        
        try:
            question = report_data.get('question', 'Unknown Query')
            iterations = report_data.get('search_iterations', [])
            
            for i, iteration in enumerate(iterations):
                tavily_answer = iteration.get('tavily_answer', '')
                if tavily_answer.strip():  # Only save if there's actual content
                    
                    # Generate filename with iteration number
                    base_question = f"{question}_iter{i+1}"
                    filename = self._generate_filename(base_question, is_tavily_answer=True)
                    filepath = os.path.join(self.base_path, filename)
                    
                    # Create content
                    content = self._format_tavily_answer(
                        question, iteration.get('query', ''), tavily_answer, i+1
                    )
                    
                    # Save to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    saved_files.append(filepath)
                    logger.info(f"Tavily answer saved: {filepath}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save Tavily answers: {e}")
            raise
    
    def _format_full_report(self, question: str, answer: str, context: str, 
                           iterations: List[Dict], stats: Dict) -> str:
        """Format the full report content."""
        report_lines = [
            "=" * 60,
            "SEARCH REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "QUESTION:",
            "-" * 20,
            question,
            "",
            "ANSWER:",
            "-" * 20,
            answer,
            "",
            "SEARCH STATISTICS:",
            "-" * 20,
            f"Sources consulted: {stats['sources_used']}",
            f"Search iterations: {stats['total_iterations']}",
            f"Credits used: {stats['credits_used']}",
            "",
            "SEARCH PROCESS:",
            "-" * 20,
        ]
        
        for i, iteration in enumerate(iterations, 1):
            report_lines.extend([
                f"{i}. Query: {iteration.get('query', 'N/A')}",
                f"   Results: {iteration.get('results_count', 0)}",
                f"   Response time: {iteration.get('response_time', 0)}s",
                f"   Credits: {iteration.get('credits_used', 1)}",
                ""
            ])
        
        if context:
            report_lines.extend([
                "CONTEXT/SOURCES:",
                "-" * 20,
                context,
                ""
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def _format_tavily_answer(self, original_question: str, search_query: str, 
                             tavily_answer: str, iteration: int) -> str:
        """Format Tavily Quick Answer content."""
        content_lines = [
            "=" * 50,
            "TAVILY QUICK ANSWER",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Iteration: {iteration}",
            "",
            "ORIGINAL QUESTION:",
            "-" * 20,
            original_question,
            "",
            "SEARCH QUERY:",
            "-" * 20,
            search_query,
            "",
            "TAVILY QUICK ANSWER:",
            "-" * 20,
            tavily_answer,
            "",
            "=" * 50
        ]
        
        return "\n".join(content_lines)
    
    def save_report_complete(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save both full report and Tavily answers, return file paths.
        
        Args:
            report_data: Dictionary containing report information
            
        Returns:
            Dictionary with saved file paths
        """
        try:
            # Save full report
            full_report_path = self.save_full_report(report_data)
            
            # Save Tavily answers
            tavily_files = self.save_tavily_answers(report_data)
            
            result = {
                'full_report': full_report_path,
                'tavily_answers': tavily_files,
                'total_files': 1 + len(tavily_files)
            }
            
            logger.info(f"Report saving complete. Total files: {result['total_files']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to save complete report: {e}")
            raise
