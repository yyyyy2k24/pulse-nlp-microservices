import os
import json
import logging
import hashlib
from typing import Optional, List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

from ..config import settings

logger = logging.getLogger(__name__)

class VectorStorageService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and SentenceTransformer embedding model"""
        try:
            # Ensure persist directory exists
            os.makedirs(settings.CHROMADB_PERSIST_DIR, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMADB_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize SentenceTransformer for embeddings
            logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer model loaded successfully")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                metadata={
                    "description": "Financial statement analysis cache with SentenceTransformers",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "version": settings.SERVICE_VERSION,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            )
            
            logger.info(f"✅ ChromaDB initialized: {settings.CHROMADB_PERSIST_DIR}")
            logger.info(f"📊 Collection size: {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def _generate_document_id(self, statement: str, author_info: Dict[str, Any]) -> str:
        """Generate consistent document ID for statement"""
        content = f"{statement}_{author_info.get('username', '')}_{author_info.get('followers', 0)}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _extract_metadata(self, request_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for storage"""
        author = request_data.get('author', {})
        platform_metrics = request_data.get('platform_metrics', {})
        
        metadata = {
            # Author information
            "username": author.get('username', ''),
            "followers": author.get('followers', 0),
            "verified": author.get('verified', False),
            "user_tier": self._determine_user_tier(author.get('followers', 0)),
            
            # Platform information
            "source": request_data.get('source', ''),
            "likes": platform_metrics.get('likes', 0),
            "shares": platform_metrics.get('shares', 0),
            "engagement_rate": platform_metrics.get('engagement_rate', 0.0),
            
            # Analysis results
            "sentiment": analysis_result.get('sentiment', ''),
            "sentiment_score": analysis_result.get('sentiment_score', 5.0),
            "impact_level": analysis_result.get('market_impact', {}).get('overall_impact', 'low'),
            "confidence": analysis_result.get('confidence', 'medium'),
            "assets_count": len(analysis_result.get('mentioned_assets', [])),
            
            # Storage metadata
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": analysis_result.get('processing_time_ms', 0),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            
            # Full analysis result (JSON string)
            "analysis_result": json.dumps(analysis_result, default=str)
        }
        
        return metadata
    
    def _determine_user_tier(self, followers: int) -> str:
        """Determine user tier based on followers"""
        if followers >= 1_000_000:
            return "mega"
        elif followers >= 100_000:
            return "macro"
        elif followers >= 10_000:
            return "influencer"
        elif followers >= 1_000:
            return "micro"
        else:
            return "retail"
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using SentenceTransformer"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            # Generate embedding with normalization
            embedding = self.embedding_model.encode([text], normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def store_analysis(
        self, 
        statement: str,
        request_data: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> bool:
        """Store analysis result in vector database"""
        if not self.collection or not self.embedding_model:
            logger.warning("Collection or embedding model not initialized")
            return False
        
        try:
            # Generate document ID
            doc_id = self._generate_document_id(statement, request_data.get('author', {}))
            
            # Generate embedding using SentenceTransformer
            embedding = self._embed_text(statement)
            
            # Prepare metadata
            metadata = self._extract_metadata(request_data, analysis_result)
            
            # Store in ChromaDB
            self.collection.add(
                ids=[doc_id],
                documents=[statement],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            logger.info(f"💾 Stored analysis: '{statement[:50]}...' (ID: {doc_id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store analysis: {e}")
            return False
    
    async def find_similar_analysis(
        self,
        statement: str,
        request_data: Dict[str, Any],
        similarity_threshold: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Find similar analysis from vector database"""
        if not self.collection or not self.embedding_model:
            return None, None
        
        threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        try:
            # Check for exact match first
            author = request_data.get('author', {})
            exact_doc_id = self._generate_document_id(statement, author)
            
            try:
                exact_result = self.collection.get(ids=[exact_doc_id])
                if exact_result["documents"]:
                    cached_data = json.loads(exact_result["metadatas"][0]["analysis_result"])
                    logger.info(f"🎯 Exact match found: '{statement[:50]}...'")
                    return cached_data, 1.0
            except:
                pass
            
            # Semantic similarity search using SentenceTransformer embedding
            query_embedding = self._embed_text(statement)
            
            # Build filter for similar user tiers
            user_tier = self._determine_user_tier(author.get('followers', 0))
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=settings.MAX_CACHE_RESULTS,
                include=["documents", "metadatas", "distances"],
                where={"user_tier": user_tier}
            )
            
            if results["documents"] and results["documents"][0]:
                # Calculate similarity (ChromaDB returns cosine distance)
                distance = results["distances"][0][0] if results["distances"] else 1.0
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    cached_data = json.loads(results["metadatas"][0][0]["analysis_result"])
                    # Update cache hit metadata
                    cached_data["cache_hit"] = True
                    cached_data["similarity_score"] = round(similarity, 3)
                    
                    logger.info(f"🔍 Similar analysis found: {similarity:.3f} similarity")
                    return cached_data, similarity
            
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Failed to search vector DB: {e}")
            return None, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        if not self.collection:
            return {"connected": False, "error": "Collection not initialized"}
        
        try:
            count = self.collection.count()
            
            # Get sample metadata for statistics
            sample = self.collection.get(limit=100, include=["metadatas"])
            
            # Calculate statistics from sample
            stats = {
                "connected": True,
                "total_documents": count,
                "persist_directory": settings.CHROMADB_PERSIST_DIR,
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimensions": 384  # all-MiniLM-L6-v2 output size
            }
            
            if sample["metadatas"]:
                # Calculate distribution statistics
                sentiments = [m.get("sentiment", "neutral") for m in sample["metadatas"]]
                user_tiers = [m.get("user_tier", "retail") for m in sample["metadatas"]]
                
                stats.update({
                    "sample_size": len(sample["metadatas"]),
                    "sentiment_distribution": {
                        "bullish": sentiments.count("bullish"),
                        "bearish": sentiments.count("bearish"), 
                        "neutral": sentiments.count("neutral")
                    },
                    "user_tier_distribution": {
                        "retail": user_tiers.count("retail"),
                        "micro": user_tiers.count("micro"),
                        "influencer": user_tiers.count("influencer"),
                        "macro": user_tiers.count("macro"),
                        "mega": user_tiers.count("mega")
                    }
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {"connected": False, "error": str(e)}

# Global service instance
_vector_service = None

def get_vector_service() -> VectorStorageService:
    """Get singleton vector service instance"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorStorageService()
    return _vector_service
