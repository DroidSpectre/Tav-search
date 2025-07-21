# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "path/to/your/model.gguf")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable is required")

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model path {MODEL_PATH} does not exist. Please update MODEL_PATH in .env file")

# LLM Configuration
LLM_CONFIG = {
    "n_ctx": 6144,
    "n_threads": 4,
    "temperature": 0.7,
    "max_tokens": 2048
}

# Search Configuration
DEFAULT_SEARCH_CONFIG = {
    "max_results": 7,
    "search_depth": "basic",  # "basic" or "advanced"
    "topic": "general",       # "general" or "news"
    "include_answer": True,
    "include_raw_content": True,
    "include_images": False,
    "include_image_descriptions": False,
    "days": 3,               # for news topic
    "include_domains": [],
    "exclude_domains": [],
    "timeout": 30
}

# Application Configuration
MAX_RECURSIVE_SEARCHES = 3
MAX_CONVERSATION_HISTORY = 10
