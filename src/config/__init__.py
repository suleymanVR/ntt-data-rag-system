"""
Configuration management package.
Centralized settings and Azure client initialization.
"""

from .settings import settings, get_settings
from .azure_clients import (
    initialize_azure_clients,
    get_embedding_client,
    get_chat_client,
    is_azure_healthy
)

__all__ = [
    "settings",
    "get_settings", 
    "initialize_azure_clients",
    "get_embedding_client",
    "get_chat_client",
    "is_azure_healthy"
]