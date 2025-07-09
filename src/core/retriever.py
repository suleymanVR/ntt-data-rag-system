"""
Vector retrieval module using Qdrant for enterprise-grade similarity search.
Handles vector search operations with advanced scoring and boosting.
"""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SearchRequest, Filter, 
    FieldCondition, MatchValue, SearchParams, PointStruct
)

from ..models.chunk_models import DocumentChunk, ChunkType
from ..models.search_models import SearchQuery, SearchResult, SearchResults, SimilarityMatrix
from ..core.embeddings import EmbeddingManager
from ..core.query_processor import QueryProcessor
from ..config.settings import settings

logger = logging.getLogger(__name__)


class QdrantVectorRetriever:
    """Enterprise-grade vector retrieval using Qdrant database."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.query_processor = QueryProcessor()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Qdrant configuration
        self.collection_name = "ntt_sustainability_chunks"
        self.vector_size = 3072  # text-embedding-3-large dimension
        self.client: Optional[QdrantClient] = None
        
        self._setup_qdrant_client()
    
    def _setup_qdrant_client(self):
        """Setup Qdrant client and collection."""
        try:
            # Get Qdrant connection details from environment or defaults
            qdrant_host = getattr(settings, 'qdrant_host', 'localhost')
            qdrant_port = getattr(settings, 'qdrant_port', 6333)
            
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            # Test connection
            health = self.client.get_collection(self.collection_name) if self._collection_exists() else None
            
            logger.info(f"✅ Qdrant client initialized at {qdrant_host}:{qdrant_port}")
            
        except Exception as e:
            logger.error(f"❌ Qdrant client setup failed: {e}")
            logger.warning("⚠️ Falling back to in-memory search")
            self.client = None
    
    def _collection_exists(self) -> bool:
        """Check if collection exists in Qdrant."""
        try:
            collections = self.client.get_collections()
            return any(col.name == self.collection_name for col in collections.collections)
        except:
            return False
    
    def _create_collection(self):
        """Create Qdrant collection with proper configuration."""
        try:
            if self._collection_exists():
                logger.info(f"📖 Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with optimized settings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,  # Best for text embeddings
                    on_disk=True  # Optimize for large datasets
                )
            )
            
            logger.info(f"✅ Created Qdrant collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"❌ Error creating Qdrant collection: {e}")
            raise
    
    def load_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Load document chunks into Qdrant with fallback to in-memory."""
        try:
            self.chunks = chunks
            logger.info(f"📚 Loading {len(chunks)} chunks for retrieval")
            
            # Always create in-memory embeddings as fallback
            self._create_in_memory_embeddings(chunks)
            
            # Try Qdrant integration
            if self.client:
                self._load_chunks_to_qdrant(chunks)
            else:
                logger.warning("⚠️ Using in-memory search (Qdrant unavailable)")
                
        except Exception as e:
            logger.error(f"❌ Error loading chunks: {e}")
            raise
    
    def _create_in_memory_embeddings(self, chunks: List[DocumentChunk]):
        """Create in-memory embeddings array for fallback."""
        embeddings_list = []
        for chunk in chunks:
            if chunk.embedding:
                embeddings_list.append(chunk.embedding)
            else:
                embedding_dim = self.embedding_manager.get_embedding_dimension()
                embeddings_list.append([0.0] * embedding_dim)
                logger.warning(f"⚠️ Missing embedding for chunk {chunk.metadata.chunk_id}")
        
        if embeddings_list:
            self.embeddings = np.array(embeddings_list)
            logger.info(f"✅ Created in-memory embeddings: {len(self.embeddings)} vectors")
    
    def _load_chunks_to_qdrant(self, chunks: List[DocumentChunk]):
        """Load chunks into Qdrant vector database."""
        try:
            # Create collection if needed
            self._create_collection()
            
            # Check if data already exists
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == len(chunks):
                logger.info(f"📖 Qdrant collection already contains {len(chunks)} points")
                return
            
            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                if not chunk.embedding:
                    continue
                
                # Create metadata for filtering and search
                payload = {
                    "text": chunk.text,
                    "source": chunk.metadata.source,
                    "page": chunk.metadata.page,
                    "chunk_id": chunk.metadata.chunk_id,
                    "chunk_type": str(chunk.metadata.chunk_type),
                    "has_numbers": chunk.metadata.has_numbers,
                    "has_keywords": chunk.metadata.has_keywords,
                    "chunk_index": chunk.metadata.chunk_index,
                    "created_at": chunk.metadata.created_at.isoformat() if chunk.metadata.created_at else None
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Unique ID for each point
                    vector=chunk.embedding,
                    payload=payload
                )
                points.append(point)
            
            # Batch upload to Qdrant
            if points:
                # Clear existing points
                self.client.delete_collection(self.collection_name)
                self._create_collection()
                
                # Upload in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"📤 Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
                
                logger.info(f"✅ Successfully loaded {len(points)} points to Qdrant")
            
        except Exception as e:
            logger.error(f"❌ Error loading chunks to Qdrant: {e}")
            logger.warning("⚠️ Continuing with in-memory search")
    
    def calculate_boosted_score(self, base_score: float, chunk_metadata: Dict) -> float:
        """Calculate boosted similarity score based on chunk characteristics."""
        try:
            boosted_score = base_score
            
            # Apply chunk type boosting
            chunk_type = chunk_metadata.get('chunk_type', 'general')
            if chunk_type == 'metrics':
                boosted_score *= settings.rag.metrics_boost
            elif chunk_type == 'sustainability':
                boosted_score *= settings.rag.sustainability_boost
            
            # Apply content-based boosting
            if chunk_metadata.get('has_numbers', False):
                boosted_score *= settings.rag.numbers_boost
            
            # Additional boosting for high-value content
            if chunk_metadata.get('has_keywords', False) and chunk_metadata.get('has_numbers', False):
                boosted_score *= 1.02
            
            return min(boosted_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating boosted score: {e}")
            return base_score
    
    async def search_similar_chunks(self, query: str, max_results: int = None) -> SearchResults:
        """Search for similar chunks using Qdrant or in-memory fallback."""
        try:
            start_time = time.time()
            max_results = max_results or settings.rag.max_chunks_per_query
            
            logger.info(f"🔍 Searching for: '{query}' (max_results: {max_results})")
            
            # Generate query variations
            processed_queries = self.query_processor.generate_multi_queries(query)
            logger.info(f"🔄 Generated {len(processed_queries)} query variations")
            
            # Try Qdrant search first
            if self.client and self._collection_exists():
                search_results = await self._qdrant_search(processed_queries, max_results)
            else:
                # Fallback to in-memory search
                search_results = await self._fallback_search(processed_queries, max_results)
            
            search_time = (time.time() - start_time) * 1000
            
            # Log results
            if search_results:
                scores_str = [f"{r.boosted_score:.3f}" for r in search_results]
                logger.info(f"📊 Found {len(search_results)} relevant chunks")
                logger.info(f"📈 Scores: {scores_str}")
            
            return SearchResults(
                query=query,
                processed_queries=processed_queries,
                results=search_results,
                total_found=len(search_results),
                search_time_ms=search_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error in search_similar_chunks: {e}")
            return SearchResults(
                query=query,
                processed_queries=[query],
                results=[],
                total_found=0,
                search_time_ms=0.0
            )
    
    async def _qdrant_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """Perform search using Qdrant vector database."""
        try:
            all_results = {}
            threshold = settings.rag.similarity_threshold
            
            for query in queries:
                # Create query embedding
                query_embedding = await self.embedding_manager.create_single_embedding(query)
                
                # Search in Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding[0].tolist(),
                    limit=max_results * 2,  # Get more candidates
                    score_threshold=threshold,
                    search_params=SearchParams(hnsw_ef=128, exact=False)
                )
                
                for result in search_results:
                    chunk_id = result.payload['chunk_id']
                    base_score = float(result.score)
                    
                    # Calculate boosted score
                    boosted_score = self.calculate_boosted_score(base_score, result.payload)
                    
                    # Keep highest score for each chunk
                    if chunk_id not in all_results or boosted_score > all_results[chunk_id]['boosted_score']:
                        # Find original chunk for metadata
                        original_chunk = None
                        for chunk in self.chunks:
                            if chunk.metadata.chunk_id == chunk_id:
                                original_chunk = chunk
                                break
                        
                        if original_chunk:
                            all_results[chunk_id] = {
                                'text': result.payload['text'],
                                'metadata': original_chunk.metadata,
                                'similarity_score': base_score,
                                'boosted_score': boosted_score
                            }
            
            # Sort and return top results
            sorted_results = sorted(all_results.values(), key=lambda x: x['boosted_score'], reverse=True)
            
            final_results = []
            for i, result in enumerate(sorted_results[:max_results]):
                search_result = SearchResult(
                    text=result['text'],
                    metadata=result['metadata'],
                    similarity_score=result['similarity_score'],
                    boosted_score=result['boosted_score'],
                    rank=i + 1
                )
                final_results.append(search_result)
            
            logger.info(f"🎯 Qdrant search completed: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Qdrant search failed: {e}")
            return []
    
    async def _fallback_search(self, queries: List[str], max_results: int) -> List[SearchResult]:
        """Fallback to in-memory search when Qdrant is unavailable."""
        logger.warning("⚠️ Using fallback in-memory search")
        
        if not self.chunks or self.embeddings is None:
            return []
        
        # Use the existing in-memory search logic
        from sklearn.metrics.pairwise import cosine_similarity
        
        all_results = {}
        threshold = settings.rag.similarity_threshold
        
        for query in queries:
            try:
                query_embedding = await self.embedding_manager.create_single_embedding(query)
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
                
                top_indices = np.argsort(similarities)[-(max_results * 2):][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > threshold:
                        chunk = self.chunks[idx]
                        chunk_id = chunk.metadata.chunk_id
                        base_score = float(similarities[idx])
                        
                        # Calculate boosted score
                        chunk_dict = {
                            'chunk_type': str(chunk.metadata.chunk_type),
                            'has_numbers': chunk.metadata.has_numbers,
                            'has_keywords': chunk.metadata.has_keywords
                        }
                        boosted_score = self.calculate_boosted_score(base_score, chunk_dict)
                        
                        if chunk_id not in all_results or boosted_score > all_results[chunk_id]['boosted_score']:
                            all_results[chunk_id] = {
                                'text': chunk.text,
                                'metadata': chunk.metadata,
                                'similarity_score': base_score,
                                'boosted_score': boosted_score
                            }
                            
            except Exception as e:
                logger.error(f"❌ Error in fallback search for query '{query}': {e}")
                continue
        
        # Sort and return results
        sorted_results = sorted(all_results.values(), key=lambda x: x['boosted_score'], reverse=True)
        
        final_results = []
        for i, result in enumerate(sorted_results[:max_results]):
            search_result = SearchResult(
                text=result['text'],
                metadata=result['metadata'],
                similarity_score=result['similarity_score'],
                boosted_score=result['boosted_score'],
                rank=i + 1
            )
            final_results.append(search_result)
        
        return final_results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval system statistics."""
        try:
            if not self.chunks:
                return {"status": "no_data", "chunks": 0, "embeddings": 0}
            
            # Calculate chunk type distribution
            chunk_types = {}
            for chunk in self.chunks:
                chunk_type = str(chunk.metadata.chunk_type)
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Check embedding status
            embedding_count = sum(1 for chunk in self.chunks if chunk.embedding)
            missing_embeddings = len(self.chunks) - embedding_count
            
            # Get embedding dimension
            embedding_dimension = self.vector_size
            if self.embeddings is not None:
                embedding_dimension = self.embeddings.shape[1]
            
            # Qdrant-specific stats
            qdrant_stats = {}
            if self.client:
                try:
                    if self._collection_exists():
                        collection_info = self.client.get_collection(self.collection_name)
                        qdrant_stats = {
                            "qdrant_points": collection_info.points_count,
                            "qdrant_status": "connected",
                            "vector_database": "Qdrant v1.7.0"
                        }
                    else:
                        qdrant_stats = {
                            "qdrant_points": 0,
                            "qdrant_status": "no_collection",
                            "vector_database": "Qdrant v1.7.0"
                        }
                except:
                    qdrant_stats = {
                        "qdrant_points": 0,
                        "qdrant_status": "error",
                        "vector_database": "Qdrant v1.7.0"
                    }
            else:
                qdrant_stats = {
                    "qdrant_points": 0,
                    "qdrant_status": "disconnected",
                    "vector_database": "In-memory fallback"
                }
            
            return {
                "status": "ready",
                "total_chunks": len(self.chunks),
                "chunk_distribution": chunk_types,
                "embeddings_loaded": embedding_count,
                "missing_embeddings": missing_embeddings,
                "embedding_dimension": embedding_dimension,
                "similarity_threshold": settings.rag.similarity_threshold,
                "max_chunks_per_query": settings.rag.max_chunks_per_query,
                **qdrant_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting retrieval stats: {e}")
            return {"status": "error", "error": str(e)}


# Backward compatibility - create an alias
VectorRetriever = QdrantVectorRetriever