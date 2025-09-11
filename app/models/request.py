from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class SourceType(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    REDDIT = "reddit"
    NEWS = "news"
    BLOG = "blog"
    FORUM = "forum"
    TELEGRAM = "telegram"
    DISCORD = "discord"

class UserTier(str, Enum):
    RETAIL = "retail"           # < 1K followers
    MICRO = "micro"             # 1K - 10K followers  
    INFLUENCER = "influencer"   # 10K - 100K followers
    MACRO = "macro"             # 100K - 1M followers
    MEGA = "mega"               # 1M+ followers

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"  
    AGGRESSIVE = "aggressive"

class TimeHorizon(str, Enum):
    SCALP = "scalp"         # Minutes to hours
    DAY = "day"             # Hours to days
    SWING = "swing"         # Days to weeks  
    POSITION = "position"   # Weeks to months
    LONG_TERM = "long_term" # Months to years

class AnalysisRequest(BaseModel):
    # Core Statement Data
    statement: str = Field(..., min_length=3, max_length=2000, description="The financial statement to analyze")
    
    # Author Context  
    author: Dict[str, Any] = Field(..., description="Information about the statement author")
    
    # Platform Context
    source: SourceType = Field(..., description="Platform where statement was posted")
    platform_metrics: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific engagement metrics")
    
    # Market Context (Optional)
    market_context: Optional[Dict[str, Any]] = Field(default=None, description="Current market conditions")
    
    # User Preferences (Optional)
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="User's trading/investment profile")
    
    # Analysis Options
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis configuration options")

    @field_validator("author")
    @classmethod
    def validate_author(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate author information contains required fields"""
        required_fields = ["username", "followers", "verified"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Author must include '{field}' field")
        
        if not isinstance(v["followers"], int) or v["followers"] < 0:
            raise ValueError("Followers must be a non-negative integer")
        
        return v

    @field_validator("platform_metrics")
    @classmethod
    def validate_platform_metrics(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate platform engagement metrics are non-negative integers"""
        # Ensure numeric values are non-negative
        numeric_fields = ["likes", "shares", "comments", "views", "retweets"]
        for field in numeric_fields:
            if field in v and (not isinstance(v[field], int) or v[field] < 0):
                raise ValueError(f"{field} must be a non-negative integer")
        return v

    class Config:
        json_schema_extra = {  
            "example": {
                "statement": "Bitcoin breaking through $70k resistance with massive institutional buying. This could trigger the next bull run phase!",
                "author": {
                    "username": "crypto_analyst_pro",
                    "followers": 250000,
                    "verified": True,
                    "account_age_days": 1200,
                    "bio": "Professional crypto analyst | 10+ years experience",
                    "tier": "macro"
                },
                "source": "twitter",
                "platform_metrics": {
                    "likes": 1500,
                    "retweets": 450,
                    "comments": 120,
                    "impressions": 45000,
                    "engagement_rate": 4.2
                },
                "market_context": {
                    "btc_price": 69800,
                    "market_cap": "1.2T",
                    "fear_greed_index": 75,
                    "trending_assets": ["BTC", "ETH", "SOL"]
                },
                "user_profile": {
                    "risk_tolerance": "moderate",
                    "time_horizon": "swing",
                    "portfolio_focus": ["crypto", "tech_stocks"],
                    "experience_level": "intermediate"
                },
                "options": {
                    "include_technical_analysis": True,
                    "priority": "speed",
                    "confidence_threshold": 0.75
                }
            }
        }

class BulkAnalysisRequest(BaseModel):
    statements: List[AnalysisRequest] = Field(..., min_items=1, max_items=50, description="Batch of statements to analyze")
    
    class Config:
        json_schema_extra = {  
            "example": {
                "statements": [
                    # Include 2-3 example requests here
                ]
            }
        }
