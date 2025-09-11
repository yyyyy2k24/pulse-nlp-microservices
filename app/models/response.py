from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class SentimentType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class ImpactLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ConfidenceLevel(str, Enum):
    HIGH = "high"      # 80-100%
    MEDIUM = "medium"  # 60-80%
    LOW = "low"        # 40-60%
    VERY_LOW = "very_low"  # <40%

class TimeframeAnalysis(BaseModel):
    immediate: str = Field(..., description="0-24 hour outlook")
    short_term: str = Field(..., description="1-30 day outlook")
    long_term: str = Field(..., description="3-12 month outlook")

class AssetMention(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol")
    name: str = Field(..., description="Full asset name")
    type: str = Field(..., description="Asset type (crypto, stock, commodity, etc.)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in asset extraction")
    context: str = Field(..., description="How the asset was mentioned in the statement")

class MarketImpactAnalysis(BaseModel):
    overall_impact: ImpactLevel = Field(..., description="Overall market impact level")
    affected_sectors: List[str] = Field(default_factory=list, description="Market sectors likely to be affected")
    catalyst_type: str = Field(..., description="Type of market catalyst (regulatory, technical, fundamental, etc.)")
    impact_timeline: str = Field(..., description="Expected timeline for impact to materialize")

class RiskAssessment(BaseModel):
    risk_level: str = Field(..., description="Overall risk level for traders/investors")
    key_risks: List[str] = Field(default_factory=list, description="Primary risk factors to consider")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Suggested risk mitigation approaches")

class StatementAnalysis(BaseModel):
    # Original Input
    statement: str = Field(..., description="Original statement analyzed")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When analysis was performed")
    
    # Core Analysis Results
    sentiment: SentimentType = Field(..., description="Overall sentiment direction")
    sentiment_score: float = Field(..., ge=1, le=10, description="Sentiment intensity (1=very bearish, 10=very bullish)")
    confidence: ConfidenceLevel = Field(..., description="Analysis confidence level")
    
    # Impact Analysis
    market_impact: MarketImpactAnalysis = Field(..., description="Market impact assessment")
    
    # Asset Analysis
    mentioned_assets: List[AssetMention] = Field(default_factory=list, description="Assets mentioned in statement")
    
    # AI Reasoning
    ai_reasoning: str = Field(..., description="Detailed explanation of why this analysis matters")
    key_insights: List[str] = Field(default_factory=list, description="Key takeaways for investors/traders")
    
    # Timeframe Analysis
    timeframe_analysis: TimeframeAnalysis = Field(..., description="Multi-timeframe market outlook")
    
    # Risk Assessment
    risk_assessment: RiskAssessment = Field(..., description="Risk analysis for market participants")
    
    # Analysis Metadata
    processing_time_ms: int = Field(..., description="Analysis processing time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether result came from cache")
    similarity_score: Optional[float] = Field(default=None, description="Similarity to cached result (if applicable)")
    
    class Config:
        json_schema_extra = {  
            "example": {
                "statement": "Bitcoin breaking through $70k resistance with massive institutional buying.",
                "analysis_timestamp": "2025-09-05T14:05:00Z",
                "sentiment": "bullish",
                "sentiment_score": 8.5,
                "confidence": "high",
                "market_impact": {
                    "overall_impact": "high",
                    "affected_sectors": ["cryptocurrency", "fintech", "payments"],
                    "catalyst_type": "technical_institutional",
                    "impact_timeline": "immediate_to_short_term"
                },
                "mentioned_assets": [
                    {
                        "symbol": "BTC",
                        "name": "Bitcoin", 
                        "type": "cryptocurrency",
                        "confidence": 0.95,
                        "context": "primary subject breaking resistance"
                    }
                ],
                "ai_reasoning": "This statement indicates a significant technical breakout for Bitcoin with institutional support, suggesting strong bullish momentum that could drive broader crypto market sentiment.",
                "key_insights": [
                    "Technical resistance break suggests upward price momentum",
                    "Institutional buying provides fundamental support",
                    "Could trigger FOMO buying in broader crypto market"
                ],
                "timeframe_analysis": {
                    "immediate": "Expect continued upward price action with increased volatility",
                    "short_term": "Bullish trend likely to continue if $70k holds as support",
                    "long_term": "Institutional adoption trend supports sustained higher valuations"
                },
                "risk_assessment": {
                    "risk_level": "moderate",
                    "key_risks": ["Profit taking at psychological levels", "Regulatory uncertainty", "Market manipulation concerns"],
                    "mitigation_strategies": ["Dollar-cost averaging", "Stop-loss orders", "Portfolio diversification"]
                },
                "processing_time_ms": 1250,
                "cache_hit": False,
                "similarity_score": None
            }
        }

class AnalysisResponse(BaseModel):
    success: bool = Field(..., description="Whether analysis completed successfully")
    analysis: StatementAnalysis = Field(..., description="Detailed analysis results")
    
class BulkAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Whether bulk analysis completed successfully") 
    total_processed: int = Field(..., description="Number of statements processed")
    results: List[StatementAnalysis] = Field(..., description="Analysis results for each statement")
    processing_summary: Dict[str, Any] = Field(..., description="Summary of processing metrics")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    components: Dict[str, Any] = Field(..., description="Component health status")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

class ServiceStatsResponse(BaseModel):
    total_analyses: int = Field(..., description="Total analyses performed")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    vector_db_size: int = Field(..., description="Number of cached analyses in vector DB")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
