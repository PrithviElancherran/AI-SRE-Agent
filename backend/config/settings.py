"""Application settings and configuration."""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    # Application settings
    APP_NAME: str = "AI SRE Agent"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Enable debug mode for development")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ],
        description="CORS allowed origins"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts for trusted host middleware"
    )
    
    # Database settings (for demo, using SQLite)
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./ai_sre_agent.db",
        description="Database URL"
    )
    
    # AlloyDB settings (for production)
    ALLOYDB_HOST: Optional[str] = Field(default=None, description="AlloyDB host")
    ALLOYDB_PORT: Optional[int] = Field(default=5432, description="AlloyDB port")
    ALLOYDB_USER: Optional[str] = Field(default=None, description="AlloyDB username")
    ALLOYDB_PASSWORD: Optional[str] = Field(default=None, description="AlloyDB password")
    ALLOYDB_DATABASE: Optional[str] = Field(default=None, description="AlloyDB database name")
    
    # Redis settings
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching"
    )
    
    # GCP settings
    GCP_PROJECT_ID: Optional[str] = Field(default=None, description="GCP Project ID")
    GCP_REGION: str = Field(default="us-central1", description="GCP Region")
    GCP_CREDENTIALS_PATH: Optional[str] = Field(default=None, description="Path to GCP credentials JSON")
    USE_MOCK_GCP_SERVICES: bool = Field(default=True, description="Use mock GCP services for demo purposes")
    
    # AI/ML settings
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        description="Similarity threshold for incident correlation"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.8,
        description="Confidence threshold for automated actions"
    )
    
    # Demo settings
    DEMO_MODE: bool = Field(default=True, description="Enable demo mode with synthetic data")
    SYNTHETIC_DATA_PATH: str = Field(
        default="../data/synthetic",
        description="Path to synthetic data directory"
    )
    
    # Security settings
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Rate limiting settings
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Number of requests per minute per IP"
    )
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds"
    )
    
    # Monitoring settings
    METRICS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=9090, description="Metrics server port")
    
    # Performance settings
    MAX_CONCURRENT_ANALYSES: int = Field(
        default=10,
        description="Maximum concurrent incident analyses"
    )
    CACHE_TTL_SECONDS: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    
    # Demo scenario settings
    SCENARIO_1_ENABLED: bool = Field(default=True, description="Enable scenario 1: Past incident correlation")
    SCENARIO_2_ENABLED: bool = Field(default=True, description="Enable scenario 2: Playbook-driven debugging")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def alloydb_url(self) -> Optional[str]:
        """Get AlloyDB connection URL."""
        if not all([self.ALLOYDB_HOST, self.ALLOYDB_USER, self.ALLOYDB_PASSWORD, self.ALLOYDB_DATABASE]):
            return None
        return f"postgresql+asyncpg://{self.ALLOYDB_USER}:{self.ALLOYDB_PASSWORD}@{self.ALLOYDB_HOST}:{self.ALLOYDB_PORT}/{self.ALLOYDB_DATABASE}"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
