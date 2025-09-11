import os
from typing import Optional
from pydantic_settings import BaseSettings  

class Settings(BaseSettings):
    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "Pulse-nlp-microservices")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "1.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API Key (only Gemini required)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # ChromaDB Configuration
    CHROMADB_PERSIST_DIR: str = os.getenv("CHROMADB_PERSIST_DIR", "./data/chromadb")
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "financial_statements")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Analysis Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
    MAX_CACHE_RESULTS: int = int(os.getenv("MAX_CACHE_RESULTS", "5"))
    ANALYSIS_TIMEOUT_SECONDS: int = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "30"))
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def api_keys_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY)  # Only Gemini required

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
