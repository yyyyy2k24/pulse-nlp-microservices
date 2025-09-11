import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from ..config import settings
from .vector_service import get_vector_service
from .gemini_service import get_gemini_service
from ..models.request import AnalysisRequest, BulkAnalysisRequest
from ..models.response import StatementAnalysis

logger = logging.getLogger(__name__)

class FinancialAnalysisService:
    def __init__(self):
        self.vector_service = get_vector_service()
        self.gemini_service = get_gemini_service()
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "processing_times": []
        }
    
    async def analyze_single_statement(self, request: AnalysisRequest) -> StatementAnalysis:
        """Analyze a single financial statement"""
        self.stats["total_analyses"] += 1
        
        try:
            # Convert request to dict for processing using Pydantic v2 method
            request_data = request.model_dump()  # ✅ Updated from .dict() to .model_dump()
            statement = request_data["statement"]
            
            # Check vector database for similar analysis
            cached_result, similarity = await self.vector_service.find_similar_analysis(
                statement, request_data
            )
            
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(f"📋 Cache hit - similarity: {similarity:.3f}")
                return StatementAnalysis(**cached_result)
            
            # Perform new analysis with Gemini
            logger.info("🔬 Performing new Gemini analysis...")
            analysis_result, processing_time = await self.gemini_service.analyze_statement(request_data)
            
            # Store result in vector database for future use
            stored = await self.vector_service.store_analysis(
                statement, request_data, analysis_result
            )
            
            if stored:
                logger.info("💾 Analysis cached for future queries")
            
            # Track processing time
            self.stats["processing_times"].append(processing_time)
            if len(self.stats["processing_times"]) > 1000:
                self.stats["processing_times"] = self.stats["processing_times"][-500:]
            
            return StatementAnalysis(**analysis_result)
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
    
    async def analyze_bulk_statements(self, request: BulkAnalysisRequest) -> List[StatementAnalysis]:
        """Analyze multiple statements in batch"""
        results = []
        cache_hits = 0
        processing_times = []
        
        # Process statements concurrently (with rate limiting)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
        
        async def process_statement(stmt_request):
            async with semaphore:
                return await self.analyze_single_statement(stmt_request)
        
        # Create tasks for all statements
        tasks = [process_statement(stmt) for stmt in request.statements]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and collect stats
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to analyze statement {i}: {result}")
                # Create error response
                error_result = StatementAnalysis(
                    statement=request.statements[i].statement,
                    analysis_timestamp=datetime.utcnow(),
                    sentiment="neutral",
                    sentiment_score=5.0,
                    confidence="very_low",
                    market_impact={
                        "overall_impact": "low",
                        "affected_sectors": [],
                        "catalyst_type": "error",
                        "impact_timeline": "unknown"
                    },
                    mentioned_assets=[],
                    ai_reasoning=f"Analysis failed: {str(result)}",
                    key_insights=["Analysis error occurred"],
                    timeframe_analysis={
                        "immediate": "Analysis unavailable",
                        "short_term": "Analysis unavailable",
                        "long_term": "Analysis unavailable"
                    },
                    risk_assessment={
                        "risk_level": "unknown",
                        "key_risks": ["Analysis failure"],
                        "mitigation_strategies": ["Retry analysis"]
                    },
                    processing_time_ms=0,
                    cache_hit=False
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
                if result.cache_hit:
                    cache_hits += 1
                processing_times.append(result.processing_time_ms)
        
        logger.info(f"📊 Bulk analysis complete: {len(valid_results)} statements, {cache_hits} cache hits")
        return valid_results
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["total_analyses"], 1)) * 100
        
        avg_processing_time = 0
        if self.stats["processing_times"]:
            avg_processing_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        vector_stats = self.vector_service.get_statistics()
        
        return {
            "total_analyses_performed": self.stats["total_analyses"],
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "vector_database": vector_stats,
            "service_health": "healthy" if vector_stats.get("connected") else "degraded"
        }

    def serialize_analysis_result(self, result: StatementAnalysis) -> str:
        """
        Serialize analysis result to JSON string
        Using Pydantic v2 model_dump_json() method
        """
        return result.model_dump_json()  # ✅ Updated from .json() to .model_dump_json()
    
    def get_analysis_dict(self, result: StatementAnalysis) -> Dict[str, Any]:
        """
        Convert analysis result to dictionary
        Using Pydantic v2 model_dump() method
        """
        return result.model_dump()  # ✅ Updated from .dict() to .model_dump()

# Global service instance
_analysis_service = None

def get_analysis_service() -> FinancialAnalysisService:
    """Get singleton analysis service instance"""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = FinancialAnalysisService()
    return _analysis_service
