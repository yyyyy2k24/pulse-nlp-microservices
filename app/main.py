import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models.request import AnalysisRequest, BulkAnalysisRequest
from .models.response import (
    AnalysisResponse, BulkAnalysisResponse, HealthResponse, 
    ServiceStatsResponse, StatementAnalysis
)
from .services.analysis_service import get_analysis_service
from .services.vector_service import get_vector_service
from .services.gemini_service import get_gemini_service
from .utils.helpers import flatten_json, remove_excluded_fields

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if settings.is_production else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Service startup time
SERVICE_START_TIME = time.time()

# Global services (will be initialized in lifespan)
analysis_service = None
vector_service = None
gemini_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application
    Handles startup and shutdown events in a single place
    """
    global analysis_service, vector_service, gemini_service
    
    # 🚀 STARTUP
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info("📡 Using SentenceTransformers for embeddings (no OpenAI cost)")
    
    try:
        # Validate configuration
        if not settings.api_keys_configured:
            raise ValueError("Missing required API key (GEMINI_API_KEY)")
        
        # Initialize services
        logger.info("🔧 Initializing services...")
        vector_service = get_vector_service()
        gemini_service = get_gemini_service()
        analysis_service = get_analysis_service()
        
        # Test connections
        if gemini_service.test_connection():
            logger.info("✅ Gemini API connection verified")
        else:
            logger.warning("⚠️ Gemini API connection test failed")
        
        vector_stats = vector_service.get_statistics()
        if vector_stats.get("connected"):
            logger.info(f"✅ ChromaDB connected: {vector_stats['total_documents']} cached analyses")
            logger.info(f"🤖 Embedding model: {vector_stats.get('embedding_model', 'unknown')}")
        else:
            logger.warning("⚠️ ChromaDB connection issues")
        
        logger.info("✅ All services initialized successfully")
        logger.info("🌐 API server ready to handle requests")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    # Yield control to FastAPI (app runs here)
    yield
    
    # 🛑 SHUTDOWN
    logger.info("🛑 Shutting down application...")
    
    try:
        # Cleanup resources
        if vector_service:
            # Optional: Close vector database connections if needed
            logger.info("💾 Closing vector database connections")
        
        if analysis_service:
            # Optional: Save final statistics
            stats = analysis_service.get_service_statistics()
            logger.info(f"📊 Final stats: {stats['total_analyses_performed']} analyses, "
                       f"{stats['cache_hit_rate_percent']:.1f}% cache hit rate")
        
        uptime = int(time.time() - SERVICE_START_TIME)
        logger.info(f"⏱️ Total uptime: {uptime} seconds")
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="Advanced Financial NLP Service with ChromaDB Vector Storage and Gemini AI - SentenceTransformers Edition",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan  # ✅ NEW: Use lifespan instead of on_event
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "description": "Advanced Financial NLP Analysis with Vector Caching",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "cost_optimized": "No OpenAI API costs - uses local embeddings",
        "output_format": "Linear JSON (flattened structure)",
        "status": "operational",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "bulk_analyze": "/api/v1/analyze/bulk", 
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "features": [
            "AI-powered sentiment analysis via Gemini",
            "Cost-free semantic caching with SentenceTransformers",
            "Multi-timeframe market analysis",
            "Risk assessment and insights",
            "Bulk processing support",
            "Author influence weighting",
            "Filtered linear JSON output format"
        ]
    }

@app.post("/api/v1/analyze")
async def analyze_statement(request: AnalysisRequest):
    """
    Analyze a single financial statement with filtered linear JSON output
    """
    try:
        # Validate service initialization
        if not analysis_service:
            logger.error("Analysis service not initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        logger.info(f"🔍 Processing analysis request for statement: '{request.statement[:50]}...'")
        
        # Perform analysis through service layer
        result = await analysis_service.analyze_single_statement(request)
        
        # Convert Pydantic model to dictionary (Pydantic v2 compatible)
        result_dict = result.model_dump()
        
        # Flatten nested JSON structure to linear format
        linear_result = flatten_json(result_dict)
        
        # Remove excluded fields as per user requirements
        filtered_result = remove_excluded_fields(linear_result)
        
        logger.info(f"✅ Analysis completed successfully - Sentiment: {result.sentiment}, Score: {result.sentiment_score}")
        
        return {
            "success": True,
            "analysis": filtered_result
        }
        
    except ValueError as e:
        logger.warning(f"⚠️ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/analyze/bulk")
async def analyze_statements_bulk(request: BulkAnalysisRequest):
    """
    Analyze multiple financial statements in batch with filtered linear JSON output
    """
    try:
        # Validate service initialization
        if not analysis_service:
            logger.error("Analysis service not initialized")
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        logger.info(f"🔍 Processing bulk analysis request for {len(request.statements)} statements")
        
        start_time = time.time()
        
        # Perform bulk analysis through service layer
        results = await analysis_service.analyze_bulk_statements(request)
        
        # Process each result: flatten and filter
        filtered_results = []
        cache_hits = 0
        
        for i, result in enumerate(results):
            try:
                # Convert Pydantic model to dictionary (Pydantic v2 compatible)
                result_dict = result.model_dump()
                
                # Flatten nested JSON structure to linear format
                linear_result = flatten_json(result_dict)
                
                # Remove excluded fields as per user requirements
                filtered_result = remove_excluded_fields(linear_result)
                
                filtered_results.append(filtered_result)
                
                # Track cache hits for metrics
                if result.cache_hit:
                    cache_hits += 1
                    
            except Exception as result_error:
                logger.error(f"❌ Failed to process result {i}: {result_error}")
                # Add error result for this statement
                filtered_results.append({
                    "success": False,
                    "error": f"Failed to process statement {i}: {str(result_error)}"
                })
        
        processing_time = time.time() - start_time
        
        # Generate processing summary (without excluded fields)
        processing_summary = {
            "total_processing_time_seconds": round(processing_time, 2),
            "average_time_per_statement_ms": round((processing_time * 1000) / len(results), 2),
            "cache_hit_rate_percent": round((cache_hits / len(results)) * 100, 2),
            "statements_processed": len(results),
            "cache_hits": cache_hits,
            "new_analyses": len(results) - cache_hits
        }
        
        logger.info(f"✅ Bulk analysis completed: {len(results)} statements, {cache_hits} cache hits, {processing_time:.2f}s")
        
        return {
            "success": True,
            "total_processed": len(results),
            "results": filtered_results,
            "processing_summary": processing_summary
        }
        
    except ValueError as e:
        logger.warning(f"⚠️ Validation error in bulk analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Bulk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Service health check endpoint with filtered linear JSON output"""
    try:
        gemini_connected = gemini_service.test_connection() if gemini_service else False
        vector_stats = vector_service.get_statistics() if vector_service else {"connected": False}
        
        # Determine overall status
        if gemini_connected and vector_stats.get("connected"):
            status = "healthy"
        elif gemini_connected or vector_stats.get("connected"):
            status = "degraded"
        else:
            status = "unhealthy"
        
        components = {
            "gemini_ai_status": "connected" if gemini_connected else "disconnected",
            "gemini_ai_description": "Gemini AI API for analysis",
            "vector_database_status": "connected" if vector_stats.get("connected") else "disconnected",
            "vector_database_description": "ChromaDB vector storage with SentenceTransformers",
            "vector_database_documents_count": vector_stats.get("total_documents", 0),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "configuration_status": "valid" if settings.api_keys_configured else "invalid",
            "configuration_description": "API keys and configuration"
        }
        
        performance_metrics = {}
        if analysis_service:
            stats = analysis_service.get_service_statistics()
            performance_metrics = {
                "total_analyses": stats.get("total_analyses_performed", 0),
                "cache_hit_rate": stats.get("cache_hit_rate_percent", 0),
                "avg_processing_time_ms": stats.get("average_processing_time_ms", 0),
                "uptime_seconds": int(time.time() - SERVICE_START_TIME)
            }
        
        # Return flattened health response (without excluded fields)
        health_response = {
            "status": status,
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Flatten the nested components and performance metrics
        health_response.update(flatten_json({"components": components}))
        health_response.update(flatten_json({"performance_metrics": performance_metrics}))
        
        # Remove excluded fields from health response
        filtered_health = remove_excluded_fields(health_response)
        
        return filtered_health
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/stats")
async def service_statistics():
    """Get detailed service statistics with filtered linear JSON output"""
    try:
        if not analysis_service:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        stats = analysis_service.get_service_statistics()
        vector_stats = vector_service.get_statistics()
        
        stats_response = {
            "total_analyses": stats.get("total_analyses_performed", 0),
            "cache_hit_rate": stats.get("cache_hit_rate_percent", 0.0),
            "average_processing_time_ms": stats.get("average_processing_time_ms", 0.0),
            "vector_db_size": vector_stats.get("total_documents", 0),
            "uptime_seconds": int(time.time() - SERVICE_START_TIME),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "cost_optimization": "Using free local embeddings"
        }
        
        # Add vector DB stats if available
        if vector_stats.get("sentiment_distribution"):
            stats_response.update(flatten_json({"vector_db": vector_stats}))
        
        # Remove excluded fields from stats response
        filtered_stats = remove_excluded_fields(stats_response)
        
        return filtered_stats
        
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats unavailable: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "available_endpoints": ["/api/v1/analyze", "/health", "/stats"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please try again later"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=not settings.is_production,
        log_level=settings.LOG_LEVEL.lower()
    )
