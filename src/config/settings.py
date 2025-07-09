"""
Configuration settings for NTT DATA RAG System.
Centralized configuration management using Pydantic Settings.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI specific settings."""
    
    # Azure OpenAI Configuration
    endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    api_key: str = Field(default="", description="Azure OpenAI API key")
    deployment: str = Field(default="gpt-4.1", description="Chat completion model deployment")
    api_version: str = Field(default="2024-12-01-preview", description="Azure OpenAI API version")
    
    # Embedding Configuration
    embedding_api_key: str = Field(default="", description="Azure OpenAI embedding API key")
    embedding_deployment: str = Field(default="text-embedding-3-large", description="Embedding model deployment")
    embedding_api_version: str = Field(default="2024-12-01-preview", description="Embedding API version")
    
    model_config = SettingsConfigDict(
        env_prefix="AZURE_",
        case_sensitive=False
    )


# settings.py'da RAGSettings class'ına şunları ekle:

class RAGSettings(BaseSettings):
    """RAG system specific settings."""
    
    # Document Processing
    chunk_size: int = Field(default=800, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=150, description="Overlap between chunks")
    similarity_threshold: float = Field(default=0.25, description="Minimum similarity threshold")
    max_chunks_per_query: int = Field(default=8, description="Maximum chunks to retrieve per query")  # 4'ten 8'e çıkarıldı
    
    # Score Boosting
    metrics_boost: float = Field(default=1.15, description="Score boost for metrics chunks")
    sustainability_boost: float = Field(default=1.10, description="Score boost for sustainability chunks")
    numbers_boost: float = Field(default=1.05, description="Score boost for chunks with numbers")
    
    # Query Processing
    enable_multi_query: bool = Field(default=True, description="Enable multi-query search")
    enable_synonym_expansion: bool = Field(default=True, description="Enable synonym expansion")
    enable_bilingual_support: bool = Field(default=True, description="Enable Turkish/English support")
    
    # Qdrant Configuration - YENİ EKLENEN
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_collection_name: str = Field(default="ntt_sustainability_chunks", description="Qdrant collection name")
    qdrant_vector_size: int = Field(default=3072, description="Vector dimension size")
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=False
    )


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, description="API port number")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    allowed_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    allowed_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # API Documentation
    title: str = Field(default="NTT DATA RAG API", description="API title")
    description: str = Field(
        default="NTT DATA Sürdürülebilirlik Raporları için Azure OpenAI Tabanlı RAG Sistemi",
        description="API description"
    )
    version: str = Field(default="2.1.0", description="API version")
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=False
    )


class DirectorySettings(BaseSettings):
    """Directory and file path settings."""
    
    # Data Directories
    reports_dir: str = Field(default="./reports", description="PDF reports directory")
    data_dir: str = Field(default="./data", description="Data storage directory")
    logs_dir: str = Field(default="./data/logs", description="Logs directory")
    cache_dir: str = Field(default="./data/cache", description="Cache directory")
    
    # Vector Database
    vector_db_dir: str = Field(default="./data/chroma_db", description="Vector database directory")
    
    model_config = SettingsConfigDict(
        env_prefix="DIR_",
        case_sensitive=False
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_file_max_size: int = Field(default=10485760, description="Max log file size in bytes (10MB)")
    log_file_backup_count: int = Field(default=5, description="Number of backup log files")
    
    # Disable noisy loggers
    disable_autogen_logs: bool = Field(default=True, description="Disable AutoGen core event logs")
    
    @field_validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    
    # Sub-settings
    azure: AzureOpenAISettings = AzureOpenAISettings()
    rag: RAGSettings = RAGSettings()
    api: APISettings = APISettings()
    directories: DirectorySettings = DirectorySettings()
    logging: LoggingSettings = LoggingSettings()
    
    @field_validator('environment')
    def validate_environment(cls, v: str) -> str:
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'environment must be one of {valid_envs}')
        return v.lower()
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Export settings for easy import
settings = get_settings()