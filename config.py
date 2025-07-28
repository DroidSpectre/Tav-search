# config.py-v2 - Enhanced with parameter validation and type hints
"""
Enhanced configuration module with parameter validation, type hints, and dataclass support.
Provides comprehensive configuration for Tavily API and LLM settings with validation.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TavilyConfig:
    """Enhanced Tavily API configuration with validation"""
    api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    search_depth: str = "basic"  # "basic" or "advanced"
    topic: str = "general"  # "general", "news", "finance"
    max_results: int = 7
    timeout: int = 30
    include_answer: bool = True
    include_raw_content: bool = True
    include_images: bool = False
    include_image_descriptions: bool = False
    auto_parameters: bool = False
    days: Optional[int] = None  # For news searches
    time_range: Optional[str] = None  # "day", "week", "month", "year"
    start_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    end_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        
        if self.search_depth not in ["basic", "advanced"]:
            raise ValueError("search_depth must be 'basic' or 'advanced'")
        
        if self.topic not in ["general", "news", "finance"]:
            raise ValueError("topic must be 'general', 'news', or 'finance'")
        
        if not 0 <= self.max_results <= 20:
            raise ValueError("max_results must be between 0 and 20")
        
        if self.timeout < 1:
            raise ValueError("timeout must be positive")
        
        if self.days is not None and self.days < 1:
            raise ValueError("days must be positive")
        
        if self.time_range and self.time_range not in ["day", "week", "month", "year"]:
            raise ValueError("time_range must be one of 'day', 'week', 'month', 'year'")
        
        # Validate date formats if provided
        if self.start_date:
            self._validate_date_format(self.start_date, "start_date")
        if self.end_date:
            self._validate_date_format(self.end_date, "end_date")
    
    def _validate_date_format(self, date_str: str, field_name: str) -> None:
        """Validate ISO date format (YYYY-MM-DD)"""
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            raise ValueError(f"{field_name} must be in ISO format (YYYY-MM-DD)")
    
    def to_search_params(self) -> Dict[str, Any]:
        """Convert config to search parameters dictionary"""
        params = {
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": self.max_results,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_images": self.include_images,
            "include_image_descriptions": self.include_image_descriptions,
            "timeout": self.timeout,
        }
        
        # Add optional parameters only if they have values
        if self.auto_parameters:
            params["auto_parameters"] = self.auto_parameters
        if self.days is not None:
            params["days"] = self.days
        if self.time_range:
            params["time_range"] = self.time_range
        if self.start_date:
            params["start_date"] = self.start_date
        if self.end_date:
            params["end_date"] = self.end_date
        if self.include_domains:
            params["include_domains"] = self.include_domains
        if self.exclude_domains:
            params["exclude_domains"] = self.exclude_domains
        
        return params

@dataclass
class LLMConfig:
    """Enhanced LLM configuration with validation"""
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "path/to/your/model.gguf"))
    n_ctx: int = 6144  # Context window size
    n_threads: int = 4  # Number of threads
    temperature: float = 0.7  # Response creativity (0.0-2.0)
    max_tokens: int = 2048  # Maximum response length
    timeout: int = 120  # Generation timeout in seconds
    top_p: float = 0.9  # Nucleus sampling parameter
    top_k: int = 50  # Top-k sampling parameter
    repeat_penalty: float = 1.1  # Repetition penalty
    verbose: bool = False  # Enable verbose logging
    
    def __post_init__(self) -> None:
        """Validate LLM configuration parameters"""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model path {self.model_path} does not exist. Please update MODEL_PATH in .env file")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        
        if self.n_ctx < 512:
            raise ValueError("n_ctx must be at least 512")
        
        if self.n_threads < 1:
            raise ValueError("n_threads must be positive")
        
        if self.timeout < 1:
            raise ValueError("timeout must be positive")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        
        if self.repeat_penalty < 0.1:
            raise ValueError("repeat_penalty must be at least 0.1")
    
    def to_llama_params(self) -> Dict[str, Any]:
        """Convert config to llama-cpp-python parameters"""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "verbose": self.verbose,
        }
    
    def to_generation_params(self) -> Dict[str, Any]:
        """Convert config to generation parameters"""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }

# Legacy compatibility - create instances with default values
try:
    # Create default configurations
    DEFAULT_TAVILY_CONFIG = TavilyConfig()
    DEFAULT_LLM_CONFIG = LLMConfig()
    
    # Legacy variables for backward compatibility
    TAVILY_API_KEY = DEFAULT_TAVILY_CONFIG.api_key
    MODEL_PATH = DEFAULT_LLM_CONFIG.model_path
    
    # Legacy LLM configuration
    LLM_CONFIG = {
        "n_ctx": DEFAULT_LLM_CONFIG.n_ctx,
        "n_threads": DEFAULT_LLM_CONFIG.n_threads,
        "temperature": DEFAULT_LLM_CONFIG.temperature,
        "max_tokens": DEFAULT_LLM_CONFIG.max_tokens,
    }
    
    # Legacy search configuration  
    DEFAULT_SEARCH_CONFIG = {
        "max_results": DEFAULT_TAVILY_CONFIG.max_results,
        "search_depth": DEFAULT_TAVILY_CONFIG.search_depth,
        "topic": DEFAULT_TAVILY_CONFIG.topic,
        "include_answer": DEFAULT_TAVILY_CONFIG.include_answer,
        "include_raw_content": DEFAULT_TAVILY_CONFIG.include_raw_content,
        "include_images": DEFAULT_TAVILY_CONFIG.include_images,
        "include_image_descriptions": DEFAULT_TAVILY_CONFIG.include_image_descriptions,
        "days": 3,  # Default for news searches
        "include_domains": [],
        "exclude_domains": [],
        "timeout": DEFAULT_TAVILY_CONFIG.timeout,
    }
    
except Exception as e:
    print(f"Configuration error: {e}")
    print("Please check your .env file and ensure all required variables are set.")
    raise

# Application configuration with validation
MAX_RECURSIVE_SEARCHES = 3
MAX_CONVERSATION_HISTORY = 10

# Validate application constants
if MAX_RECURSIVE_SEARCHES < 1:
    raise ValueError("MAX_RECURSIVE_SEARCHES must be positive")
if MAX_CONVERSATION_HISTORY < 1:
    raise ValueError("MAX_CONVERSATION_HISTORY must be positive")

def validate_environment() -> bool:
    """
    Validate that all required environment variables and configurations are properly set.
    
    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Test Tavily configuration
        tavily_config = TavilyConfig()
        if not tavily_config.api_key:
            print("❌ TAVILY_API_KEY not found in environment")
            return False
        
        # Test LLM configuration
        llm_config = LLMConfig()
        if not os.path.exists(llm_config.model_path):
            print(f"❌ Model file not found: {llm_config.model_path}")
            return False
        
        print("✅ Environment validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration settings.
    
    Returns:
        Dictionary containing configuration summary
    """
    try:
        tavily_config = TavilyConfig()
        llm_config = LLMConfig()
        
        return {
            "tavily": {
                "api_key_set": bool(tavily_config.api_key),
                "search_depth": tavily_config.search_depth,
                "topic": tavily_config.topic,
                "max_results": tavily_config.max_results,
                "timeout": tavily_config.timeout,
            },
            "llm": {
                "model_path": llm_config.model_path,
                "model_exists": os.path.exists(llm_config.model_path),
                "n_ctx": llm_config.n_ctx,
                "n_threads": llm_config.n_threads,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
            },
            "application": {
                "max_recursive_searches": MAX_RECURSIVE_SEARCHES,
                "max_conversation_history": MAX_CONVERSATION_HISTORY,
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

# Optional: Print configuration summary on import (for debugging)
if __name__ == "__main__":
    import json
    print("Configuration Summary:")
    print(json.dumps(get_config_summary(), indent=2))
