"""
Base exception classes for the NTT DATA RAG system.
Defines the exception hierarchy and common functionality.
"""

from typing import Optional, Dict, Any


class NTTRAGException(Exception):
    """Base exception class for all NTT RAG system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Code: {self.error_code}, Context: {context_str})"
        return f"{self.message} (Code: {self.error_code})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "context": self.context
        }


class ConfigurationError(NTTRAGException):
    """Raised when there's a configuration problem."""
    pass


class InitializationError(NTTRAGException):
    """Raised when system initialization fails."""
    pass


class DocumentProcessingError(NTTRAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(NTTRAGException):
    """Raised when embedding operations fail."""
    pass


class SearchError(NTTRAGException):
    """Raised when search operations fail."""
    pass


class AgentError(NTTRAGException):
    """Raised when AutoGen agent operations fail."""
    pass