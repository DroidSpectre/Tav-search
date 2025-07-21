# main.py
import sys
import logging
from typing import Optional
from search_agent import SearchAgent
from config import DEFAULT_SEARCH_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchInterface:
    def __init__(self):
        try:
            self.agent = SearchAgent()
            logger.info("Search interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search interface: {e}")
            print(f"❌ Initialization failed: {e}")
            print("Please check your configuration:")
            print("1. Ensure TAVILY_API_KEY is set in .env file")
            print("2. Ensure MODEL_PATH points to a valid GGUF model file")
            sys.exit(1)
    
    def interactive_search(self):
        """
        Interactive command-line search interface
        """
        print("🔍 Advanced LLM Search Engine")
        print("=" * 50)
        print("Commands:")
        print("  - Type your question to search")
        print("  - 'help' for search options")
        print("  - 'clear' to start new session")
        print("  - 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.agent.clear_session()
                    print("🧹 Session cleared")
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Check for search parameters in input
                search_params = self._parse_search_params(user_input)
                question = search_params.pop('question', user_input)
                
                print(f"\n🔄 Searching for: {question}")
                if search_params.get('search_depth') == 'advanced':
                    print("⚡ Using advanced search (2 credits per search)")
                
                # Perform search
                result = self.agent.search_and_synthesize(question, **search_params)
                
                # Display results
                self._display_results(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
    
    def _parse_search_params(self, user_input: str) -> dict:
        """
        Parse search parameters from user input
        """
        params = DEFAULT_SEARCH_CONFIG.copy()
        
        # Simple parameter parsing
        if '--news' in user_input:
            params['topic'] = 'news'
            user_input = user_input.replace('--news', '')
        
        if '--advanced' in user_input:
            params['search_depth'] = 'advanced'
            user_input = user_input.replace('--advanced', '')
        
        if '--recent' in user_input:
            params['topic'] = 'news'
            params['days'] = 7
            user_input = user_input.replace('--recent', '')
        
        params['question'] = user_input.strip()
        return params
    
    def _display_results(self, result: dict):
        """
        Display search results in a formatted way
        """
        print(f"\n📄 Answer:")
        print("-" * 40)
        print(result['answer'])
        
        print(f"\n📊 Search Statistics:")
        print(f"  • Search iterations: {result['total_iterations']}")
        print(f"  • Sources consulted: {result['sources_used']}")
        print(f"  • Credits used: {result['credits_used']}")
        
        if result.get('search_iterations'):
            print(f"\n🔍 Search Process:")
            for i, iteration in enumerate(result['search_iterations'], 1):
                print(f"  {i}. Query: {iteration['query']}")
                print(f"     Results: {iteration['results_count']}")
                print(f"     Response time: {iteration.get('response_time', 'N/A')}s")
                print(f"     Credits: {iteration.get('credits_used', 1)}")
                if iteration.get('tavily_answer'):
                    print(f"     Quick answer: {iteration['tavily_answer'][:100]}...")
    
    def _show_help(self):
        """
        Show help information
        """
        print("\n📚 Search Options:")
        print("  --news      Search recent news")
        print("  --advanced  Use advanced search (2 credits, more thorough)")
        print("  --recent    Search recent news (last 7 days)")
        print("\n💡 Examples:")
        print("  What is quantum computing? --advanced")
        print("  Latest AI developments --news")
        print("  Recent climate change news --recent")

def main():
    """
    Main entry point
    """
    interface = SearchInterface()
    interface.interactive_search()

if __name__ == "__main__":
    main()
