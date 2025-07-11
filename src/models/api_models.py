"""
API request and response models for the NTT DATA RAG System.
Pydantic models for FastAPI endpoint validation and documentation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ConversationMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    max_chunks: Optional[int] = Field(default=4, ge=1, le=10, description="Maximum chunks to retrieve")
    include_metadata: Optional[bool] = Field(default=True, description="Include detailed metadata in response")
    conversation_history: Optional[List[ConversationMessage]] = Field(default=[], description="Previous conversation context")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class AnswerResponse(BaseModel):
    """Response model for question answers."""
    
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default=[], description="Source documents and pages")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata about the response")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "NTT DATA'nın 2020 yılındaki sürdürülebilirlik hedefleri...",
                "sources": ["ntt_data_sustainability_report_2020.pdf (Sayfa 15)", "ntt_data_sustainability_report_2021.pdf (Sayfa 23)"],
                "metadata": {
                    "chunks_found": 3,
                    "similarity_scores": [0.856, 0.742, 0.689],
                    "chat_model": "gpt-4.1",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    documents_loaded: int = Field(..., description="Number of loaded documents")
    total_chunks: int = Field(..., description="Total number of text chunks")
    chat_model: str = Field(..., description="Chat completion model")
    embedding_model: str = Field(..., description="Embedding model")
    model_status: str = Field(..., description="Azure OpenAI models status")
    embedding_dimension: Optional[int] = Field(default=0, description="Embedding vector dimension")
    chunk_distribution: Optional[Dict[str, int]] = Field(default={}, description="Distribution of chunk types")
    optimization_features: Optional[List[str]] = Field(default=[], description="Enabled optimization features")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "documents_loaded": 5,
                "total_chunks": 342,
                "chat_model": "gpt-4.1",
                "embedding_model": "text-embedding-3-large",
                "model_status": "healthy",
                "embedding_dimension": 3072,
                "chunk_distribution": {
                    "general": 180,
                    "sustainability": 85,
                    "metrics": 65,
                    "title": 12
                },
                "optimization_features": [
                    "multi_query_search",
                    "chunk_type_boosting",
                    "enhanced_preprocessing"
                ]
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "No relevant content found for the question",
                "error_type": "ContentNotFound",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456789"
            }
        }
    )


class SystemInfoResponse(BaseModel):
    """System information response model."""
    
    message: str = Field(..., description="Welcome message")
    version: str = Field(..., description="API version")
    models: Dict[str, str] = Field(..., description="Model configurations")
    optimizations: List[str] = Field(..., description="Enabled optimizations")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "NTT DATA RAG API with Azure OpenAI",
                "version": "2.1.0",
                "models": {
                    "chat": "gpt-4.1",
                    "embedding": "text-embedding-3-large"
                },
                "optimizations": [
                    "multi_query_search",
                    "chunk_type_analysis",
                    "score_boosting"
                ],
                "endpoints": {
                    "docs": "/docs",
                    "health": "/health",
                    "ask": "/ask"
                }
            }
        }
    )