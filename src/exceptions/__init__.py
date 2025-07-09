"""
Exceptions package.
Custom exception hierarchy for error handling.
"""

# Base exceptions
from .base import (
    NTTRAGException,
    ConfigurationError,
    InitializationError,
    DocumentProcessingError,
    EmbeddingError,
    SearchError,
    AgentError
)

# API exceptions
from .api_exceptions import (
    RAGAPIException,
    RAGNotInitializedError,
    QuestionProcessingError,
    InvalidQuestionError,
    ContentNotFoundError,
    RateLimitExceededError,
    ServiceUnavailableError,
    AuthenticationError,
    AuthorizationError,
    ValidationError
)

__all__ = [
    # Base exceptions
    "NTTRAGException",
    "ConfigurationError", 
    "InitializationError",
    "DocumentProcessingError",
    "EmbeddingError",
    "SearchError",
    "AgentError",
    
    # API exceptions
    "RAGAPIException",
    "RAGNotInitializedError",
    "QuestionProcessingError", 
    "InvalidQuestionError",
    "ContentNotFoundError",
    "RateLimitExceededError",
    "ServiceUnavailableError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError"
]