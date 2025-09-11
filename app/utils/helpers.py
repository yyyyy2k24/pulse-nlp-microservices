import re
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone, time
import pytz

logger = logging.getLogger(__name__)

def flatten_json(nested_json: Dict[str, Any], separator: str = "_") -> Dict[str, Any]:
    """
    Flatten nested JSON object to linear structure

    Input: {"market_impact": {"overall_impact": "high", "sectors": ["crypto", "tech"]}}
    Output: {"market_impact_overall_impact": "high", "market_impact_sectors_0": "crypto", "market_impact_sectors_1": "tech"}
    """
    def _flatten(obj: Any, parent_key: str = "") -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                items.extend(_flatten(item, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return _flatten(nested_json)

def remove_excluded_fields(flat_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove specified fields from flattened JSON response
    
    Args:
        flat_json: Flattened dictionary to clean
        
    Returns:
        Cleaned dictionary with excluded fields removed
    """
    # Define fields to exclude from output
    excluded_keys = [
        "format",
        "analysis_timestamp", 
        "market_impact_catalyst_type",
        "risk_assessment_key_risks_0",
        "risk_assessment_key_risks_1", 
        "risk_assessment_mitigation_strategies_0",
        "risk_assessment_mitigation_strategies_1",
        "processing_time_ms"
    ]
    
    # Create a copy to avoid modifying original dict
    cleaned_dict = flat_json.copy()
    
    # Remove excluded keys if they exist
    for key in excluded_keys:
        cleaned_dict.pop(key, None)
    
    return cleaned_dict

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amounts for display"""
    try:
        if amount >= 1_000_000_000:
            return f"{currency} {amount/1_000_000_000:.2f}B"
        elif amount >= 1_000_000:
            return f"{currency} {amount/1_000_000:.2f}M"
        elif amount >= 1_000:
            return f"{currency} {amount/1_000:.2f}K"
        else:
            return f"{currency} {amount:.2f}"
    except (ValueError, TypeError):
        return f"{currency} 0.00"

def sanitize_text(text: str) -> str:
    """Clean and sanitize text input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove or replace potentially harmful characters
    text = re.sub(r'[^\w\s\$\#\@\.\,\!\?\-\(\)\/\:]', '', text)
    
    # Limit length
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return text

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from social media text"""
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
    return [tag.lower() for tag in hashtags]

def extract_cashtags(text: str) -> List[str]:
    """Extract cashtags ($SYMBOL) from text"""
    cashtag_pattern = r'\$([A-Z]{1,5})'
    cashtags = re.findall(cashtag_pattern, text.upper())
    return cashtags

def calculate_engagement_rate(
    likes: int = 0, 
    shares: int = 0, 
    comments: int = 0, 
    followers: int = 1
) -> float:
    """Calculate social media engagement rate"""
    total_engagement = likes + shares + comments
    if followers <= 0:
        return 0.0
    
    engagement_rate = (total_engagement / followers) * 100
    return round(engagement_rate, 2)

def determine_market_hours(timezone_str: str = "US/Eastern") -> Dict[str, Any]:
    """Determine if markets are currently open"""
    try:
        tz = pytz.timezone(timezone_str)
        current_time = datetime.now(tz)
        current_day = current_time.weekday()  # 0=Monday, 6=Sunday
        current_hour = current_time.hour
        
        # US market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        is_weekday = current_day < 5  # Monday-Friday
        is_market_hours = (current_hour >= 9 and current_hour < 16) or \
                         (current_hour == 9 and current_time.minute >= 30)
        
        market_open = is_weekday and is_market_hours
        
        return {
            "is_market_open": market_open,
            "current_time": current_time.isoformat(),
            "timezone": timezone_str,
            "trading_session": _get_trading_session(current_time)
        }
        
    except Exception as e:
        logger.error(f"Error determining market hours: {e}")
        return {
            "is_market_open": False,
            "error": str(e)
        }

def _get_trading_session(current_time: datetime) -> str:
    """Determine current trading session"""
    hour = current_time.hour
    
    if 4 <= hour < 9:
        return "pre_market"
    elif 9 <= hour < 16:
        return "regular_hours" 
    elif 16 <= hour < 20:
        return "after_hours"
    else:
        return "closed"

def format_timeframe(timeframe: str) -> str:
    """Format timeframe strings for display"""
    timeframe_map = {
        "immediate": "0-24 hours",
        "short_term": "1-30 days", 
        "medium_term": "1-6 months",
        "long_term": "6-12 months"
    }
    return timeframe_map.get(timeframe.lower(), timeframe)

def convert_to_linear_json(data: Union[Dict, List, Any]) -> Dict[str, Any]:
    """Convert nested structure to linear JSON format"""
    if isinstance(data, dict):
        return flatten_json(data)
    elif isinstance(data, list):
        # Convert list to dict with indices as keys
        indexed_dict = {str(i): item for i, item in enumerate(data)}
        return flatten_json(indexed_dict)
    else:
        return {"value": data}

def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from social media text"""
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text, re.IGNORECASE)
    return [mention.lower() for mention in mentions]

def get_crypto_asset_info(symbol: str) -> Dict[str, Any]:
    """Get basic crypto asset information"""
    crypto_map = {
        "BTC": {"name": "Bitcoin", "type": "cryptocurrency", "category": "store_of_value"},
        "ETH": {"name": "Ethereum", "type": "cryptocurrency", "category": "smart_contracts"},
        "DOGE": {"name": "Dogecoin", "type": "cryptocurrency", "category": "meme"},
        "ADA": {"name": "Cardano", "type": "cryptocurrency", "category": "smart_contracts"},
        "SOL": {"name": "Solana", "type": "cryptocurrency", "category": "smart_contracts"},
        "MATIC": {"name": "Polygon", "type": "cryptocurrency", "category": "scaling"},
        "LINK": {"name": "Chainlink", "type": "cryptocurrency", "category": "oracle"},
        "UNI": {"name": "Uniswap", "type": "cryptocurrency", "category": "defi"}
    }
    
    return crypto_map.get(symbol.upper(), {
        "name": symbol,
        "type": "unknown", 
        "category": "other"
    })

def get_stock_asset_info(symbol: str) -> Dict[str, Any]:
    """Get basic stock asset information"""
    stock_map = {
        "TSLA": {"name": "Tesla Inc", "type": "stock", "sector": "automotive"},
        "AAPL": {"name": "Apple Inc", "type": "stock", "sector": "technology"},
        "GOOGL": {"name": "Alphabet Inc", "type": "stock", "sector": "technology"},
        "MSFT": {"name": "Microsoft Corp", "type": "stock", "sector": "technology"},
        "META": {"name": "Meta Platforms", "type": "stock", "sector": "technology"},
        "NVDA": {"name": "NVIDIA Corp", "type": "stock", "sector": "technology"},
        "AMZN": {"name": "Amazon.com Inc", "type": "stock", "sector": "e-commerce"}
    }
    
    return stock_map.get(symbol.upper(), {
        "name": symbol,
        "type": "stock",
        "sector": "unknown"
    })
