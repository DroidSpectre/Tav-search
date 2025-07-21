# llm_interface.py
import logging
from typing import Dict, List, Optional, Any
from llama_cpp import Llama
from config import MODEL_PATH, LLM_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the LLM model with error handling
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=LLM_CONFIG["n_ctx"],
                n_threads=LLM_CONFIG["n_threads"],
                verbose=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model from {self.model_path}: {e}")
    
    print("🧠 Generating answer with LLM. Please wait...")
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate LLM response with proper error handling
        """
        if not self.llm:
            raise RuntimeError("Model not loaded")
        
        try:
            response = self.llm.create_completion(
                prompt,
                max_tokens=kwargs.get("max_tokens", LLM_CONFIG["max_tokens"]),
                temperature=kwargs.get("temperature", LLM_CONFIG["temperature"]),
                top_p=kwargs.get("top_p", 0.9),
                stop=kwargs.get("stop", None)
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def create_search_prompt(self, question: str, context: str, history: Optional[List[Dict]] = None) -> str:
        """
        Create optimized prompt for search synthesis
        """
        prompt = "You are an expert research assistant. Analyze the provided context and answer the question accurately.\n\n"
        
        if history:
            prompt += "Previous conversation:\n"
            for msg in history[-5:]:  # Last 5 exchanges
                prompt += f"{msg['role'].title()}: {msg['content']}\n"
            prompt += "\n"
        
        prompt += f"Question: {question}\n\n"
        prompt += f"Context from web search:\n{context}\n\n"
        prompt += "Instructions:\n"
        prompt += "1. Provide a comprehensive answer based on the context\n"
        prompt += "2. If information is insufficient, clearly state what's missing\n"
        prompt += "3. Include relevant details and cite sources when possible\n"
        prompt += "4. Be factual and avoid speculation\n\n"
        prompt += "Answer:"
        
        return prompt
    
    def assess_answer_quality(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Assess answer quality and determine if more information is needed
        """
        assessment_prompt = f"""
Evaluate this answer:

Question: {question}
Answer: {answer}
Context: {context}

Rate completeness (1-10): How well does the answer address the question?
Rate confidence (1-10): How confident are you in the answer's accuracy?
Needs more info (yes/no): Does this question need additional information?
New search query: If more info needed, suggest a specific search query, otherwise say "none"

Format your response exactly as:
Completeness: [number]
Confidence: [number] 
Needs_more_info: [yes/no]
Suggested_query: [query or "none"]
"""
        
        response = self.generate_response(assessment_prompt, max_tokens=200)
        
        # Parse the response
        assessment = {
            "completeness": 7,  # Default to moderate scores
            "confidence": 7,
            "needs_more_info": False,
            "suggested_query": None
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Completeness:'):
                try:
                    assessment["completeness"] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Confidence:'):
                try:
                    assessment["confidence"] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Needs_more_info:'):
                assessment["needs_more_info"] = 'yes' in line.lower()
            elif line.startswith('Suggested_query:'):
                query = line.split(':', 1)[1].strip()
                if query.lower() != "none":
                    assessment["suggested_query"] = query
        
        return assessment
