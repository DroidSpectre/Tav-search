# credit_manager.py
from datetime import datetime, timedelta
from typing import Dict, Any
import json

class CreditManager:
    """Track and manage Tavily API credit usage"""
    
    def __init__(self, monthly_limit: int = 1000):
        self.monthly_limit = monthly_limit
        self.usage_history = []
        
    def track_search(self, search_params: Dict[str, Any]) -> int:
        """Calculate and track credit usage for a search"""
        credits_used = self._calculate_credits(search_params)
        
        self.usage_history.append({
            'timestamp': datetime.now(),
            'credits': credits_used,
            'search_depth': search_params.get('search_depth', 'basic'),
            'max_results': search_params.get('max_results', 5)
        })
        
        return credits_used
    
    def _calculate_credits(self, params: Dict[str, Any]) -> int:
        """Calculate credits based on search parameters"""
        base_credits = 1
        if params.get('search_depth') == 'advanced':
            base_credits = 2
        return base_credits
    
    def get_monthly_usage(self) -> int:
        """Get current month's credit usage"""
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_month_usage = [
            entry for entry in self.usage_history 
            if entry['timestamp'] >= month_start
        ]
        return sum(entry['credits'] for entry in current_month_usage)
    
    def can_make_request(self, search_params: Dict[str, Any]) -> bool:
        """Check if request can be made within credit limits"""
        projected_credits = self._calculate_credits(search_params)
        current_usage = self.get_monthly_usage()
        return (current_usage + projected_credits) <= self.monthly_limit
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Generate detailed usage report"""
        monthly_usage = self.get_monthly_usage()
        return {
            'monthly_usage': monthly_usage,
            'monthly_limit': self.monthly_limit,
            'remaining_credits': self.monthly_limit - monthly_usage,
            'usage_percentage': (monthly_usage / self.monthly_limit) * 100
        }
