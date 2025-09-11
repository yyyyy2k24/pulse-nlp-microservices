import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def validate_social_media_username(username: str) -> Tuple[bool, Optional[str]]:
    """Validate social media username format"""
    if not username:
        return False, "Username cannot be empty"
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Check length
    if len(username) < 1 or len(username) > 50:
        return False, "Username must be 1-50 characters"
    
    # Check format (alphanumeric + underscores, no spaces)
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    
    return True, None

def validate_follower_count(followers: int) -> Tuple[bool, Optional[str]]:
    """Validate follower count"""
    if not isinstance(followers, int):
        return False, "Follower count must be an integer"
    
    if followers < 0:
        return False, "Follower count cannot be negative"
    
    if followers > 1_000_000_000:  # 1 billion max
        return False, "Follower count seems unrealistic (max 1B)"
    
    return True, None

def validate_platform_metrics(metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate platform engagement metrics"""
    required_numeric_fields = ["likes", "shares", "comments"]
    optional_numeric_fields = ["views", "impressions", "retweets"]
    
    for field in required_numeric_fields:
        if field in metrics:
            if not isinstance(metrics[field], int) or metrics[field] < 0:
                return False, f"{field} must be a non-negative integer"
    
    for field in optional_numeric_fields:
        if field in metrics:
            if not isinstance(metrics[field], int) or metrics[field] < 0:
                return False, f"{field} must be a non-negative integer"
    
    # Validate engagement rate if present
    if "engagement_rate" in metrics:
        rate = metrics["engagement_rate"]
        if not isinstance(rate, (int, float)) or rate < 0 or rate > 100:
            return False, "Engagement rate must be between 0 and 100"
    
    return True, None

def validate_market_data(market_context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate market context data"""
    if not isinstance(market_context, dict):
        return False, "Market context must be a dictionary"
    
    # Validate price fields
    price_fields = ["btc_price", "eth_price", "spy_price"]
    for field in price_fields:
        if field in market_context:
            price = market_context[field]
            if not isinstance(price, (int, float)) or price <= 0:
                return False, f"{field} must be a positive number"
    
    # Validate fear & greed index
    if "fear_greed_index" in market_context:
        index = market_context["fear_greed_index"]
        if not isinstance(index, (int, float)) or index < 0 or index > 100:
            return False, "Fear & greed index must be between 0 and 100"
    
    # Validate trending assets
    if "trending_assets" in market_context:
        assets = market_context["trending_assets"]
        if not isinstance(assets, list):
            return False, "Trending assets must be a list"
        
        for asset in assets:
            if not isinstance(asset, str) or not asset.strip():
                return False, "Trending assets must be non-empty strings"
    
    return True, None

def validate_analysis_request(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Comprehensive validation of analysis request"""
    
    # Check required fields
    required_fields = ["statement", "author", "source"]
    for field in required_fields:
        if field not in request_data:
            return False, f"Missing required field: {field}"
    
    # Validate statement
    statement = request_data["statement"]
    if not isinstance(statement, str) or len(statement.strip()) < 3:
        return False, "Statement must be at least 3 characters long"
    
    if len(statement) > 2000:
        return False, "Statement cannot exceed 2000 characters"
    
    # Validate author
    author = request_data["author"]
    if not isinstance(author, dict):
        return False, "Author must be a dictionary"
    
    # Validate username
    if "username" not in author:
        return False, "Author must include username"
    
    username_valid, username_error = validate_social_media_username(author["username"])
    if not username_valid:
        return False, f"Invalid username: {username_error}"
    
    # Validate followers
    if "followers" not in author:
        return False, "Author must include followers count"
    
    followers_valid, followers_error = validate_follower_count(author["followers"])
    if not followers_valid:
        return False, f"Invalid followers: {followers_error}"
    
    # Validate source
    valid_sources = ["twitter", "linkedin", "reddit", "news", "blog", "forum", "telegram", "discord"]
    if request_data["source"].lower() not in valid_sources:
        return False, f"Invalid source. Must be one of: {', '.join(valid_sources)}"
    
    # Validate platform metrics if present
    if "platform_metrics" in request_data:
        metrics_valid, metrics_error = validate_platform_metrics(request_data["platform_metrics"])
        if not metrics_valid:
            return False, f"Invalid platform metrics: {metrics_error}"
    
    # Validate market context if present
    if "market_context" in request_data:
        market_valid, market_error = validate_market_data(request_data["market_context"])
        if not market_valid:
            return False, f"Invalid market context: {market_error}"
    
    return True, None

def is_valid_asset_symbol(symbol: str) -> bool:
    """Check if asset symbol format is valid"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-10 characters, letters/numbers only
    if not re.match(r'^[A-Z0-9]{1,10}$', symbol.upper()):
        return False
    
    return True

def validate_timeframe(timeframe: str) -> Tuple[bool, Optional[str]]:
    """Validate timeframe parameter"""
    valid_timeframes = ["immediate", "short_term", "medium_term", "long_term"]
    
    if timeframe.lower() not in valid_timeframes:
        return False, f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
    
    return True, None

def validate_sentiment_score(score: float) -> Tuple[bool, Optional[str]]:
    """Validate sentiment score"""
    if not isinstance(score, (int, float)):
        return False, "Sentiment score must be a number"
    
    if score < 1 or score > 10:
        return False, "Sentiment score must be between 1 and 10"
    
    return True, None

def validate_confidence_level(confidence: str) -> Tuple[bool, Optional[str]]:
    """Validate confidence level"""
    valid_levels = ["high", "medium", "low", "very_low"]
    
    if confidence.lower() not in valid_levels:
        return False, f"Invalid confidence level. Must be one of: {', '.join(valid_levels)}"
    
    return True, None

def sanitize_and_validate_text(text: str, max_length: int = 2000) -> Tuple[str, bool, Optional[str]]:
    """Sanitize and validate text input"""
    if not text:
        return "", False, "Text cannot be empty"
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Check length
    if len(text) > max_length:
        return text[:max_length], False, f"Text exceeds maximum length of {max_length}"
    
    # Basic content validation (no just special characters)
    if not re.search(r'[a-zA-Z0-9]', text):
        return text, False, "Text must contain alphanumeric characters"
    
    return text, True, None

def validate_bulk_request(statements: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate bulk analysis request"""
    if not isinstance(statements, list):
        return False, "Statements must be a list"
    
    if len(statements) == 0:
        return False, "At least one statement is required"
    
    if len(statements) > 50:
        return False, "Maximum 50 statements allowed per bulk request"
    
    # Validate each statement
    for i, statement in enumerate(statements):
        valid, error = validate_analysis_request(statement)
        if not valid:
            return False, f"Statement {i+1}: {error}"
    
    return True, None

def validate_api_keys(gemini_key: str, openai_key: str) -> Tuple[bool, List[str]]:
    """Validate API key formats"""
    errors = []
    
    # Validate Gemini API key
    if not gemini_key:
        errors.append("GEMINI_API_KEY is required")
    elif len(gemini_key) < 10:
        errors.append("GEMINI_API_KEY appears to be invalid (too short)")
    
    # Validate OpenAI API key  
    if not openai_key:
        errors.append("OPENAI_API_KEY is required")
    elif not openai_key.startswith("sk-"):
        errors.append("OPENAI_API_KEY must start with 'sk-'")
    elif len(openai_key) < 20:
        errors.append("OPENAI_API_KEY appears to be invalid (too short)")
    
    return len(errors) == 0, errors
