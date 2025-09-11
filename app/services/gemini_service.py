import time
import json
import logging
from typing import Dict, Any, Tuple, List
from google import genai
from google.genai import types
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class GeminiAnalysisService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Configure client with new SDK
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        logger.info("✅ Gemini service initialized with new SDK")
    
    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            # Simple test with new SDK
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Test connection",
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                    temperature=0.1
                )
            )
            return True
        except Exception as e:
            logger.error(f"❌ Gemini connection test failed: {e}")
            return False
    
    async def analyze_statement(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Analyze financial statement using Gemini"""
        start_time = time.time()
        
        try:
            statement = request_data["statement"]
            author = request_data["author"]
            source = request_data["source"]
            platform_metrics = request_data.get("platform_metrics", {})
            market_context = request_data.get("market_context", {})
            user_profile = request_data.get("user_profile", {})
            
            # Build comprehensive analysis prompt
            prompt = self._build_analysis_prompt(
                statement, author, source, platform_metrics, market_context, user_profile
            )
            
            # Generate analysis with new SDK
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            # Parse response
            try:
                result_text = response.text if hasattr(response, 'text') else str(response)
                result = json.loads(result_text)
                
                # Validate and enhance result
                result = self._validate_and_enhance_result(result, request_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parse error: {e}")
                result = self._create_fallback_response(statement, f"JSON parsing failed: {e}")
            
            processing_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time
            result["analysis_timestamp"] = datetime.utcnow()
            
            logger.info(f"✨ Analysis completed in {processing_time}ms")
            
            return result, processing_time
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"❌ Gemini analysis failed: {e}")
            return self._create_fallback_response(request_data["statement"], str(e)), processing_time
    
    def _build_analysis_prompt(
        self,
        statement: str,
        author: Dict[str, Any],
        source: str,
        platform_metrics: Dict[str, Any],
        market_context: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> str:
        """Build comprehensive analysis prompt"""
        
        # Determine author influence level
        followers = author.get("followers", 0)
        if followers >= 1_000_000:
            influence_level = "MEGA INFLUENCER (1M+ followers) - Market-moving potential"
        elif followers >= 100_000:
            influence_level = "MACRO INFLUENCER (100K+ followers) - Significant reach"
        elif followers >= 10_000:
            influence_level = "MICRO INFLUENCER (10K+ followers) - Moderate influence"
        else:
            influence_level = "RETAIL USER (<10K followers) - Limited reach"
        
        prompt = f"""
You are an expert financial analyst specializing in social media sentiment analysis and market psychology. 

ANALYZE THIS FINANCIAL STATEMENT:
"{statement}"

CONTEXT INFORMATION:
Author: @{author.get('username', 'unknown')} 
- Followers: {followers:,}
- Verified: {author.get('verified', False)}
- Influence Level: {influence_level}
- Account Age: {author.get('account_age_days', 'unknown')} days

Platform: {source.upper()}
Engagement: {platform_metrics.get('likes', 0)} likes, {platform_metrics.get('shares', 0)} shares, {platform_metrics.get('comments', 0)} comments

Market Context: {json.dumps(market_context) if market_context else 'Not provided'}

Return ONLY a JSON object with this exact structure:
{{
  "sentiment": "bullish|bearish|neutral",
  "sentiment_score": 5.5,
  "confidence": "high|medium|low|very_low",
  "market_impact": {{
    "overall_impact": "high|medium|low",
    "affected_sectors": ["cryptocurrency", "fintech"],
    "catalyst_type": "technical_institutional",
    "impact_timeline": "immediate_to_short_term"
  }},
  "mentioned_assets": [
    {{
      "symbol": "BTC",
      "name": "Bitcoin",
      "type": "cryptocurrency",
      "confidence": 0.95,
      "context": "breaking through resistance"
    }}
  ],
  "ai_reasoning": "Detailed explanation of analysis...",
  "key_insights": [
    "Technical resistance break indicates momentum",
    "Institutional buying provides support"
  ],
  "timeframe_analysis": {{
    "immediate": "0-24h market reaction analysis",
    "short_term": "1-30d outlook",
    "long_term": "3-12m perspective"
  }},
  "risk_assessment": {{
    "risk_level": "moderate",
    "key_risks": ["Risk factor 1", "Risk factor 2"],
    "mitigation_strategies": ["Strategy 1", "Strategy 2"]
  }}
}}

ANALYSIS GUIDELINES:
1. Sentiment score 1-10: 1-3 (Very Bearish), 4-6 (Neutral), 7-10 (Very Bullish)
2. Impact: HIGH (100K+ followers + strong sentiment), MEDIUM (10K+ OR moderate sentiment), LOW (<10K followers)  
3. Extract ALL financial assets, convert to standard symbols
4. Provide actionable insights for traders/investors
5. Consider author influence, timing, and market context
"""
        return prompt
    
    def _validate_and_enhance_result(self, result: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance analysis result"""
        
        # Ensure statement is included
        result["statement"] = request_data["statement"]
        
        # Validate enums
        result["sentiment"] = result.get("sentiment", "neutral").lower()
        if result["sentiment"] not in ["bullish", "bearish", "neutral"]:
            result["sentiment"] = "neutral"
        
        # Validate sentiment score
        score = result.get("sentiment_score", 5.0)
        result["sentiment_score"] = max(1.0, min(10.0, float(score)))
        
        # Validate confidence
        confidence = result.get("confidence", "medium").lower()
        if confidence not in ["high", "medium", "low", "very_low"]:
            result["confidence"] = "medium"
        
        # Validate impact level
        impact = result.get("market_impact", {})
        overall_impact = impact.get("overall_impact", "low").lower()
        if overall_impact not in ["high", "medium", "low"]:
            impact["overall_impact"] = "low"
        
        # Ensure required arrays exist
        if not result.get("mentioned_assets"):
            result["mentioned_assets"] = []
        
        if not result.get("key_insights"):
            result["key_insights"] = ["General market sentiment detected"]
        
        # Ensure risk assessment exists
        if not result.get("risk_assessment"):
            result["risk_assessment"] = {
                "risk_level": "moderate",
                "key_risks": ["Market volatility", "Timing uncertainty"],
                "mitigation_strategies": ["Diversification", "Risk management"]
            }
        
        return result
    
    def _create_fallback_response(self, statement: str, error_reason: str) -> Dict[str, Any]:
        """Create fallback response when analysis fails"""
        return {
            "statement": statement,
            "sentiment": "neutral",
            "sentiment_score": 5.0,
            "confidence": "very_low",
            "market_impact": {
                "overall_impact": "low",
                "affected_sectors": [],
                "catalyst_type": "unknown",
                "impact_timeline": "uncertain"
            },
            "mentioned_assets": [],
            "ai_reasoning": f"Analysis failed due to: {error_reason}. Defaulting to neutral assessment.",
            "key_insights": ["Unable to complete detailed analysis"],
            "timeframe_analysis": {
                "immediate": "Unable to assess immediate impact",
                "short_term": "Analysis unavailable", 
                "long_term": "Long-term outlook uncertain"
            },
            "risk_assessment": {
                "risk_level": "unknown",
                "key_risks": ["Analysis uncertainty"],
                "mitigation_strategies": ["Seek additional analysis sources"]
            },
            "analysis_timestamp": datetime.utcnow(),
            "processing_time_ms": 0,
            "cache_hit": False
        }

# Global service instance
_gemini_service = None

def get_gemini_service() -> GeminiAnalysisService:
    """Get singleton Gemini service instance"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiAnalysisService()
    return _gemini_service
