"""
Embedding management module for creating and handling vector embeddings.
Handles Azure OpenAI embedding operations with batching and error handling.
"""

import asyncio
import logging
from typing import List, Optional, Tuple
import numpy as np

from ..config.azure_clients import get_embedding_client
from ..config.settings import settings
from ..models.chunk_models import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding creation and operations using Azure OpenAI."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.embedding_model = settings.azure.embedding_deployment
        self._client = None
    
    @property
    def client(self):
        """Get the Azure OpenAI embedding client."""
        if self._client is None:
            self._client = get_embedding_client()
        return self._client
    
    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts with batching."""
        try:
            if not texts:
                logger.warning("⚠️ No texts provided for embedding creation")
                return np.array([])
            
            logger.info(f"🔄 Creating embeddings for {len(texts)} texts using {self.embedding_model}")
            
            all_embeddings = []
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                try:
                    logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                    
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_model
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info(f"✅ Batch {batch_num} completed successfully")
                    
                    # Small delay to avoid rate limiting
                    if batch_num < total_batches:
                        await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_num}: {e}")
                    # Create zero embeddings as fallback
                    embedding_dim = 3072  # text-embedding-3-large dimension
                    zero_embeddings = [[0.0] * embedding_dim] * len(batch)
                    all_embeddings.extend(zero_embeddings)
            
            embeddings_array = np.array(all_embeddings)
            logger.info(f"✅ Created {len(embeddings_array)} embeddings with dimension {embeddings_array.shape[1] if len(embeddings_array) > 0 else 0}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    async def create_single_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            
            embedding = np.array([response.data[0].embedding])
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error creating single embedding: {e}")
            raise
    
    async def create_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Create embeddings for document chunks and update them in place."""
        try:
            if not chunks:
                logger.warning("⚠️ No chunks provided for embedding creation")
                return chunks
            
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Create embeddings
            embeddings = await self.create_embeddings(texts)
            
            # Update chunks with embeddings
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i].tolist()
                else:
                    logger.warning(f"⚠️ No embedding created for chunk {i}")
                    chunk.embedding = None
            
            logger.info(f"✅ Updated {len(chunks)} chunks with embeddings")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings for chunks: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        model_dimensions = {
            'text-embedding-3-large': 3072,
            'text-embedding-3-small': 1536,
            'text-embedding-ada-002': 1536
        }
        return model_dimensions.get(self.embedding_model, 1536)
    
    def validate_embeddings(self, embeddings: np.ndarray) -> Tuple[bool, str]:
        """Validate embedding array format and content."""
        try:
            if embeddings is None or len(embeddings) == 0:
                return False, "Embeddings array is empty"
            
            # Check if it's a 2D array
            if len(embeddings.shape) != 2:
                return False, f"Expected 2D array, got {len(embeddings.shape)}D"
            
            # Check dimension consistency
            expected_dim = self.get_embedding_dimension()
            if embeddings.shape[1] != expected_dim:
                return False, f"Expected dimension {expected_dim}, got {embeddings.shape[1]}"
            
            # Check for NaN or infinite values
            if np.isnan(embeddings).any():
                return False, "Embeddings contain NaN values"
            
            if np.isinf(embeddings).any():
                return False, "Embeddings contain infinite values"
            
            # Check if all embeddings are zero (potential error)
            zero_count = np.sum(np.all(embeddings == 0, axis=1))
            if zero_count > 0:
                return False, f"{zero_count} embeddings are all zeros"
            
            return True, "Embeddings are valid"
            
        except Exception as e:
            return False, f"Error validating embeddings: {e}"
    
    async def test_embedding_service(self) -> bool:
        """Test the embedding service with a simple request."""
        try:
            logger.info("🧪 Testing embedding service...")
            
            test_text = "Test embedding service connection"
            embedding = await self.create_single_embedding(test_text)
            
            if embedding is not None and len(embedding) > 0:
                logger.info("✅ Embedding service test successful")
                return True
            else:
                logger.error("❌ Embedding service test failed - no embedding returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Embedding service test failed: {e}")
            return False
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> bool:
        """Save embeddings to a file."""
        try:
            np.save(filepath, embeddings)
            logger.info(f"💾 Embeddings saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """Load embeddings from a file."""
        try:
            embeddings = np.load(filepath)
            logger.info(f"📖 Embeddings loaded from {filepath}")
            
            is_valid, message = self.validate_embeddings(embeddings)
            if not is_valid:
                logger.error(f"❌ Invalid embeddings loaded: {message}")
                return None
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            return None