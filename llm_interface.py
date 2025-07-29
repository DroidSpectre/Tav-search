# llm_interface.py-v2 - Enhanced with configuration support and type hints
"""
Enhanced LLM Interface with:
1. LLMConfig dataclass support for comprehensive configuration
2. Backward compatibility with existing code
3. Type hints and comprehensive documentation
4. Enhanced error handling and validation
5. Support for all llama-cpp-python parameters
6. Improved generation parameters handling
"""
import logging
from typing import Dict, List, Optional, Any, Union
from llama_cpp import Llama

# Import both legacy and new configuration styles
try:
    from config import MODEL_PATH, LLM_CONFIG, LLMConfig
except ImportError:
    # Fallback for basic configuration
    MODEL_PATH = "path/to/your/model.gguf"
    LLM_CONFIG = {
        "n_ctx": 6144,
        "n_threads": 4,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    LLMConfig = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    """Enhanced LLM Interface with comprehensive configuration support"""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        config: Optional['LLMConfig'] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize LLM Interface with enhanced configuration support.
        
        Args:
            model_path: Path to GGUF model file (legacy compatibility)
            config: LLMConfig instance with comprehensive settings
            n_ctx: Context window size (legacy compatibility)
            n_threads: Number of threads (legacy compatibility)
            temperature: Generation temperature (legacy compatibility)
            max_tokens: Maximum tokens to generate (legacy compatibility)
            **kwargs: Additional parameters for llama-cpp-python
            
        The constructor supports multiple initialization patterns:
        1. New style: LLMInterface(model_path="path", config=LLMConfig())
        2. Legacy style: LLMInterface(model_path="path")
        3. Mixed style: LLMInterface(model_path="path", n_ctx=4096)
        """
        # Initialize configuration
        if config is not None:
            # Use provided LLMConfig instance
            self.config = config
            self.model_path = model_path or config.model_path
        else:
            # Create config from parameters or use defaults
            if LLMConfig is not None:
                # Use LLMConfig class if available
                self.config = LLMConfig(
                    model_path=model_path or MODEL_PATH,
                    n_ctx=n_ctx or LLM_CONFIG.get("n_ctx", 6144),
                    n_threads=n_threads or LLM_CONFIG.get("n_threads", 4),
                    temperature=temperature or LLM_CONFIG.get("temperature", 0.7),
                    max_tokens=max_tokens or LLM_CONFIG.get("max_tokens", 2048),
                )
                self.model_path = self.config.model_path
            else:
                # Fallback to legacy configuration
                self.model_path = model_path or MODEL_PATH
                self.config = None
        
        # Store additional kwargs for llama-cpp-python
        self.llama_kwargs = kwargs
        
        # Initialize model instance
        self.llm: Optional[Llama] = None
        
        # Load the model
        self._load_model()
        
        logger.info("Enhanced LLMInterface initialized")

    def _load_model(self) -> None:
        """
        Load the LLM model with comprehensive error handling and configuration support.
        
        Raises:
            RuntimeError: If model loading fails
            FileNotFoundError: If model file doesn't exist
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Prepare llama-cpp-python parameters
            llama_params = self._get_llama_params()
            
            # Load model with comprehensive parameters
            self.llm = Llama(**llama_params)
            
            logger.info("Model loaded successfully")
            print("🧠 LLM model loaded and ready")
            
        except FileNotFoundError:
            error_msg = f"Model file not found: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load model from {self.model_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _get_llama_params(self) -> Dict[str, Any]:
        """
        Prepare parameters for llama-cpp-python initialization.
        
        Returns:
            Dictionary of parameters for Llama constructor
        """
        if self.config is not None:
            # Use LLMConfig parameters
            params = self.config.to_llama_params()
        else:
            # Use legacy configuration
            params = {
                "model_path": self.model_path,
                "n_ctx": LLM_CONFIG.get("n_ctx", 6144),
                "n_threads": LLM_CONFIG.get("n_threads", 4),
                "verbose": False,
            }
        
        # Add any additional kwargs
        params.update(self.llama_kwargs)
        
        # Ensure model_path is always set correctly
        params["model_path"] = self.model_path
        
        return params

    def _get_generation_params(self, **override_kwargs: Any) -> Dict[str, Any]:
        """
        Prepare parameters for text generation.
        
        Args:
            **override_kwargs: Parameters to override defaults
            
        Returns:
            Dictionary of generation parameters
        """
        if self.config is not None:
            # Use LLMConfig generation parameters
            params = self.config.to_generation_params()
        else:
            # Use legacy configuration
            params = {
                "max_tokens": LLM_CONFIG.get("max_tokens", 2048),
                "temperature": LLM_CONFIG.get("temperature", 0.7),
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.1,
            }
        
        # Apply overrides
        params.update(override_kwargs)
        
        return params

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate LLM response with comprehensive parameter support and error handling.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: Stop sequences (string or list of strings)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self.llm:
            raise RuntimeError("Model not loaded. Cannot generate response.")
        
        try:
            # Prepare generation parameters
            gen_params = self._get_generation_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                **kwargs
            )
            
            # Remove None values
            gen_params = {k: v for k, v in gen_params.items() if v is not None}
            
            logger.debug(f"Generating response with params: {gen_params}")
            
            # Generate response
            response = self.llm.create_completion(prompt, **gen_params)
            
            # Extract text from response
            generated_text = response["choices"][0]["text"].strip()
            
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            error_msg = f"LLM generation failed: {str(e)}"
            logger.error(error_msg)
            return "I apologize, but I encountered an error generating a response."

    def create_search_prompt(
        self, 
        question: str, 
        context: str, 
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Create optimized prompt for search synthesis with enhanced formatting.
        
        Args:
            question: User's question
            context: Search context from web results
            history: Conversation history (optional)
            
        Returns:
            Formatted prompt string for LLM generation
        """
        prompt_parts = [
            "You are an expert research assistant. Analyze the provided context and answer the question accurately and comprehensively.\n"
        ]
        
        # Add conversation history if available
        if history:
            prompt_parts.append("Previous conversation:")
            for msg in history[-5:]:  # Last 5 exchanges for context
                role = msg.get('role', 'unknown').title()
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")  # Empty line for separation
        
        # Add main question and context
        prompt_parts.extend([
            f"Question: {question}\n",
            f"Context from web search:\n{context}\n",
            "Instructions:",
            "1. Provide a comprehensive answer based ONLY on the context provided",
            "2. If information is insufficient, clearly state what's missing",
            "3. Include relevant details and cite sources when possible",
            "4. Be factual and avoid speculation or hallucination",
            "5. Organize the answer logically with clear explanations",
            "6. If the context is empty or irrelevant, state that clearly\n",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)

    def assess_answer_quality(
        self, 
        question: str, 
        answer: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Assess answer quality and determine if more information is needed.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Source context used for generation
            
        Returns:
            Dictionary containing quality assessment metrics
        """
        assessment_prompt = f"""Evaluate this research answer for completeness and quality:

Question: {question}

Answer: {answer}

Context: {context}

Please evaluate the answer and provide ratings:

Rate completeness (1-10): How well does the answer address all aspects of the question?
Rate confidence (1-10): How confident are you in the answer's accuracy based on the context?
Needs more info (yes/no): Does this question need additional information to be fully answered?
New search query: If more info needed, suggest ONE specific search query, otherwise say "none"

Format your response exactly as follows:
Completeness: [number]
Confidence: [number]
Needs_more_info: [yes/no]
Suggested_query: [query or "none"]"""

        try:
            response = self.generate_response(assessment_prompt, max_tokens=200, temperature=0.3)
            
            # Parse the structured response
            assessment = self._parse_assessment_response(response)
            
            logger.debug(f"Assessment result: {assessment}")
            return assessment
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            # Return conservative default assessment
            return {
                "completeness": 5,
                "confidence": 5,
                "needs_more_info": True,
                "suggested_query": None
            }

    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured assessment response from LLM.
        
        Args:
            response: Raw LLM response containing assessment
            
        Returns:
            Parsed assessment dictionary
        """
        # Default assessment values
        assessment = {
            "completeness": 7,
            "confidence": 7,
            "needs_more_info": False,
            "suggested_query": None
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                
                if line.startswith('Completeness:'):
                    try:
                        score = int(line.split(':', 1)[1].strip())
                        assessment["completeness"] = max(1, min(10, score))
                    except (ValueError, IndexError):
                        pass
                        
                elif line.startswith('Confidence:'):
                    try:
                        score = int(line.split(':', 1)[1].strip())
                        assessment["confidence"] = max(1, min(10, score))
                    except (ValueError, IndexError):
                        pass
                        
                elif line.startswith('Needs_more_info:'):
                    needs_more = line.split(':', 1)[1].strip().lower()
                    assessment["needs_more_info"] = 'yes' in needs_more
                    
                elif line.startswith('Suggested_query:'):
                    query = line.split(':', 1)[1].strip()
                    if query.lower() not in ["none", "n/a", ""]:
                        assessment["suggested_query"] = query
                        
        except Exception as e:
            logger.debug(f"Assessment parsing error: {e}")
        
        return assessment

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and configuration.
        
        Returns:
            Dictionary containing model and configuration information
        """
        info = {
            "model_path": self.model_path,
            "model_loaded": self.llm is not None,
        }
        
        if self.config is not None:
            info.update({
                "config_type": "LLMConfig",
                "n_ctx": self.config.n_ctx,
                "n_threads": self.config.n_threads,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
            })
        else:
            info.update({
                "config_type": "Legacy",
                "n_ctx": LLM_CONFIG.get("n_ctx", "unknown"),
                "n_threads": LLM_CONFIG.get("n_threads", "unknown"),
                "temperature": LLM_CONFIG.get("temperature", "unknown"),
                "max_tokens": LLM_CONFIG.get("max_tokens", "unknown"),
            })
        
        return info

    def validate_configuration(self) -> bool:
        """
        Validate the current configuration and model setup.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if model file exists
            import os
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Check if model is loaded
            if not self.llm:
                logger.error("Model not loaded")
                return False
            
            # Test basic generation
            test_response = self.generate_response("Test", max_tokens=1)
            if not test_response:
                logger.error("Model generation test failed")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def __del__(self):
        """Cleanup method to ensure proper resource management"""
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # llama-cpp-python handles cleanup automatically
                logger.debug("LLM interface cleanup completed")
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")

