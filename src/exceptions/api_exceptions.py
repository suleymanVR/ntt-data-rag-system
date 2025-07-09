"""
API-specific exception classes for HTTP error handling.
Provides structured error responses for FastAPI endpoints.
"""

from typing import Optional, Dict, Any
from .base import NTTRAGException


class RAGAPIException(NTTRAGException):
    """Base exception class for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, 
                 error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.status_code = status_code


class RAGNotInitializedError(RAGAPIException):
    """Raised when RAG system is not properly initialized."""
    
    def __init__(self, message: str = "RAG system not initialized"):
        super().__init__(message, status_code=503, error_code="RAG_NOT_INITIALIZED")


class QuestionProcessingError(RAGAPIException):
    """Raised when question processing fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, error_code="QUESTION_PROCESSING_ERROR", context=context)


class InvalidQuestionError(RAGAPIException):
    """Raised when question format or content is invalid."""
    
    def __init__(self, message: str = "Invalid question format"):
        super().__init__(message, status_code=400, error_code="INVALID_QUESTION")


class ContentNotFoundError(RAGAPIException):
    """Raised when no relevant content is found for a question."""
    
    def __init__(self, message: str = "No relevant content found"):
        super().__init__(message, status_code=404, error_code="CONTENT_NOT_FOUND")


class RateLimitExceededError(RAGAPIException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        context = {"retry_after": retry_after}
        super().__init__(message, status_code=429, error_code="RATE_LIMIT_EXCEEDED", context=context)


class ServiceUnavailableError(RAGAPIException):
    """Raised when service is temporarily unavailable."""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, status_code=503, error_code="SERVICE_UNAVAILABLE")


class AuthenticationError(RAGAPIException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, error_code="AUTHENTICATION_FAILED")


class AuthorizationError(RAGAPIException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403, error_code="ACCESS_DENIED")


class ValidationError(RAGAPIException):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        context = {"field": field} if field else {}
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR", context=context)