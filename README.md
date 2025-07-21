
# Tav-search

On-device web search using Tavily API and llama-cpp-python for intelligent search synthesis.

## Overview

Tav-search is an advanced search engine that combines the power of web search through Tavily API with local Large Language Model (LLM) processing. It performs intelligent search result synthesis, generates comprehensive answers from multiple sources, and provides detailed search analytics with automatic report generation.

## Key Features

- **Intelligent Search Synthesis**: Uses local LLM to synthesize search results into comprehensive answers
- **Multi-iteration Search**: Automatically performs follow-up searches based on answer quality assessment
- **Source Summarization**: Creates concise notes from each source to reduce hallucination
- **Parallel Search Execution**: Performs multiple search queries simultaneously for faster results
- **Automatic Report Generation**: Saves detailed search reports and Tavily Quick Answers as text files
- **Conversation History**: Maintains context across search sessions
- **Credit Tracking**: Monitors API usage and provides detailed statistics
- **Debug Mode**: Comprehensive debugging tools for troubleshooting search processes
- **Android/Termux Compatible**: Designed to run seamlessly in Termux environment

## Project Structure

```
Tav-search/
├── main.py              # Interactive command-line interface
├── search_agent.py      # Core search orchestration with parallel processing
├── web_search.py        # Tavily API integration
├── llm_interface.py     # Local LLM integration using llama-cpp-python
├── config.py           # Configuration settings and environment variables
├── report_saver.py     # Automatic report generation and file management
├── notepad.py          # Source summarization module
├── debug_notes.py      # Debugging and troubleshooting utilities
└── .gitignore          # Git ignore file for environment variables
```

## Prerequisites

- Python 3.7 or higher
- Tavily API key
- Compatible GGUF model file for llama-cpp-python

### Dependencies

- `llama-cpp-python` - For local LLM processing
- `requests` - For API communication
- `python-dotenv` - For environment variable management

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DroidSpectre/Tav-search.git
   cd Tav-search
   ```

2. **Install dependencies:**
   ```bash
   pip install llama-cpp-python requests python-dotenv
   ```

3. **Create environment file:**
   Create a `.env` file in the project root:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   MODEL_PATH=/path/to/your/model.gguf
   ```

4. **Download a compatible GGUF model:**
   - Download a suitable GGUF model file (e.g., from Hugging Face)
   - Update the `MODEL_PATH` in your `.env` file

## Usage

### Basic Usage

Run the interactive search interface:

```bash
python main.py
```

### Search Commands

- **Basic search:** Simply type your question
- **Advanced search:** Add `--advanced` for more thorough results (uses 2 credits per search)
- **News search:** Add `--news` to search recent news
- **Recent news:** Add `--recent` for news from the last 7 days
- **Help:** Type `help` for available options
- **Clear session:** Type `clear` to reset conversation history
- **Exit:** Type `quit` to exit

### Examples

```
🤔 Your question: What is quantum computing? --advanced
🤔 Your question: Latest AI developments --news
🤔 Your question: Recent climate change news --recent
```

## Configuration

The `config.py` file contains all configuration settings:

### LLM Configuration
```python
LLM_CONFIG = {
    "n_ctx": 6144,          # Context window size
    "n_threads": 4,         # Number of threads
    "temperature": 0.7,     # Response creativity
    "max_tokens": 2048      # Maximum response length
}
```

### Search Configuration
```python
DEFAULT_SEARCH_CONFIG = {
    "max_results": 7,                    # Results per search
    "search_depth": "basic",             # "basic" or "advanced"
    "topic": "general",                  # "general" or "news"
    "include_answer": True,              # Include Tavily Quick Answers
    "include_raw_content": True,         # Include full content
    "days": 3,                          # For news searches
    "timeout": 30                       # Request timeout
}
```

## Output Files

The application automatically saves reports to `/storage/emulated/0/Download/Search Reports/` (Android) or a custom directory:

- **Full Search Reports**: Complete analysis with sources, statistics, and search process
- **Tavily Quick Answers**: Individual quick answers from each search iteration
- **Debug Files**: Detailed debugging information (when debug mode is enabled)

## Features in Detail

### Intelligent Search Process
1. **Initial Search**: Performs web search using Tavily API
2. **Source Analysis**: Summarizes each source into key points
3. **Answer Synthesis**: Uses LLM to create comprehensive answer from source notes
4. **Quality Assessment**: Evaluates answer completeness and accuracy
5. **Iterative Refinement**: Performs additional searches if needed

### Anti-Hallucination Measures
- **Source-based Notes**: Creates concise summaries from each source
- **Content Limiting**: Prevents information overload in LLM context
- **Credibility Scoring**: Prioritizes reputable domains
- **Deduplication**: Removes similar sources using semantic similarity

### Credit Management
- **Basic Search**: 1 credit per search
- **Advanced Search**: 2 credits per search
- **Usage Tracking**: Detailed credit consumption reporting

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `MODEL_PATH` points to a valid GGUF file
2. **API Authentication**: Verify `TAVILY_API_KEY` is correct and valid
3. **Permission Errors**: Ensure write permissions for report directory
4. **Memory Issues**: Reduce `n_ctx` in LLM configuration for lower memory usage

### Debug Mode

Enable comprehensive debugging by checking the debug files generated in `/storage/emulated/0/Download/Debug_Notes/`:
- Source processing details
- LLM prompt and response logs
- Synthesis process information

## Android/Termux Support

This application is specifically designed to work seamlessly in Termux on Android devices:
- Uses Android-compatible file paths
- Optimized for mobile hardware constraints
- No heavyweight dependencies required
- Efficient memory usage patterns

## API Credits

Tav-search uses the Tavily API which operates on a credit system:
- Basic searches consume 1 credit each
- Advanced searches consume 2 credits each
- The application provides detailed credit usage tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the Apache License 2.0. See the LICENSE file for details.

## Support

If you encounter any issues or have questions:
1. Check the debug files in the Debug_Notes directory
2. Review the configuration settings
3. Ensure all dependencies are properly installed
4. Open an issue on GitHub with detailed error information

**Note**: Make sure to keep your `.env` file secure and never commit it to version control. The `.gitignore` file is already configured to prevent this.
