"""
Azure OpenAI client initialization and management.
Handles both chat completion and embedding clients.
"""

import logging
from typing import Optional
from openai import AzureOpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from .settings import settings

logger = logging.getLogger(__name__)


class AzureClientManager:
    """Manages Azure OpenAI clients for chat and embeddings."""
    
    def __init__(self):
        self._embedding_client: Optional[AzureOpenAI] = None
        self._chat_client: Optional[AzureOpenAIChatCompletionClient] = None
        self._initialized = False
    
    def initialize_clients(self) -> None:
        """Initialize both Azure OpenAI clients."""
        try:
            # Initialize embedding client
            self._embedding_client = AzureOpenAI(
                api_version=settings.azure.embedding_api_version,
                azure_endpoint=settings.azure.endpoint,
                api_key=settings.azure.embedding_api_key
            )
            
            # Initialize chat client for AutoGen
            self._chat_client = AzureOpenAIChatCompletionClient(
                model=settings.azure.deployment,
                api_version=settings.azure.api_version,
                azure_endpoint=settings.azure.endpoint,
                api_key=settings.azure.api_key
            )
            
            self._initialized = True
            
            logger.info("✅ Azure OpenAI clients initialized successfully")
            logger.info(f"🤖 Chat model: {settings.azure.deployment}")
            logger.info(f"🧠 Embedding model: {settings.azure.embedding_deployment}")
            
        except Exception as e:
            logger.error(f"❌ Azure OpenAI clients initialization failed: {e}")
            raise
    
    @property
    def embedding_client(self) -> AzureOpenAI:
        """Get the embedding client."""
        if not self._initialized or self._embedding_client is None:
            raise RuntimeError("Azure clients not initialized. Call initialize_clients() first.")
        return self._embedding_client
    
    @property
    def chat_client(self) -> AzureOpenAIChatCompletionClient:
        """Get the chat completion client."""
        if not self._initialized or self._chat_client is None:
            raise RuntimeError("Azure clients not initialized. Call initialize_clients() first.")
        return self._chat_client
    
    @property
    def is_healthy(self) -> bool:
        """Check if clients are healthy and initialized."""
        return self._initialized and self._embedding_client is not None and self._chat_client is not None
    
    def test_connection(self) -> bool:
        """Test Azure OpenAI connection."""
        try:
            # Test embedding client with a simple request
            response = self._embedding_client.embeddings.create(
                input=["test connection"],
                model=settings.azure.embedding_deployment
            )
            
            if response.data and len(response.data) > 0:
                logger.info("✅ Azure OpenAI connection test successful")
                return True
            else:
                logger.error("❌ Azure OpenAI connection test failed - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Azure OpenAI connection test failed: {e}")
            return False


# Global client manager instance
azure_client_manager = AzureClientManager()


def get_embedding_client() -> AzureOpenAI:
    """Get the Azure OpenAI embedding client."""
    return azure_client_manager.embedding_client


def get_chat_client() -> AzureOpenAIChatCompletionClient:
    """Get the Azure OpenAI chat completion client."""
    return azure_client_manager.chat_client


def initialize_azure_clients() -> None:
    """Initialize Azure OpenAI clients."""
    azure_client_manager.initialize_clients()


def is_azure_healthy() -> bool:
    """Check if Azure clients are healthy."""
    return azure_client_manager.is_healthy