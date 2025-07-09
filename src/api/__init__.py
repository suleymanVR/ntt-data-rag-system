"""
API layer package.
FastAPI application, routes, and middleware.
"""

from .app import create_app, get_rag_pipeline

__all__ = [
    "create_app",
    "get_rag_pipeline"
]