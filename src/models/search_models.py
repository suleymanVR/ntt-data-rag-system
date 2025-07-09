"""
Models for search operations and results.
Used for vector similarity search and retrieval.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from .chunk_models import ChunkMetadata, ChunkType


class SearchQuery(BaseModel):
    """Search query configuration."""
    
    query: str = Field(..., min_length=1, description="Search query text")
    max_results: int = Field(default=4, ge=1, le=20, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_multi_query: bool = Field(default=True, description="Enable multi-query search")
    enable_boosting: bool = Field(default=True, description="Enable score boosting")
    chunk_types: Optional[List[ChunkType]] = Field(default=None, description="Filter by chunk types")
    
    model_config = ConfigDict(
        use_enum_values=True
    )


class SearchResult(BaseModel):
    """Single search result with chunk and similarity data."""
    
    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    boosted_score: float = Field(..., ge=0.0, description="Score after boosting")
    rank: int = Field(..., ge=1, description="Result rank in search results")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "NTT DATA'nın karbon emisyon hedefleri 2025 yılına kadar %30 azaltma şeklindedir...",
                "metadata": {
                    "source": "ntt_data_sustainability_report_2020.pdf",
                    "page": 15,
                    "chunk_id": "ntt_data_sustainability_report_2020.pdf_p15_c2",
                    "chunk_index": 2,
                    "total_chunks": 5,
                    "chunk_type": "sustainability",
                    "has_numbers": True,
                    "has_keywords": True
                },
                "similarity_score": 0.742,
                "boosted_score": 0.817,
                "rank": 1
            }
        }
    )


class SearchResults(BaseModel):
    """Complete search results with metadata."""
    
    query: str = Field(..., description="Original search query")
    processed_queries: List[str] = Field(..., description="All processed query variations")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total results found")
    search_time_ms: float = Field(..., ge=0, description="Search execution time in milliseconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "karbon emisyon hedefleri",
                "processed_queries": [
                    "karbon emisyon hedefleri",
                    "karbon emisyon hedefleri carbon emission targets",
                    "carbon emission targets"
                ],
                "results": [
                    {
                        "text": "Sample chunk text...",
                        "metadata": {"source": "report.pdf", "page": 15},
                        "similarity_score": 0.742,
                        "boosted_score": 0.817,
                        "rank": 1
                    }
                ],
                "total_found": 3,
                "search_time_ms": 45.2
            }
        }
    )


class QueryExpansion(BaseModel):
    """Query expansion and preprocessing results."""
    
    original_query: str = Field(..., description="Original user query")
    cleaned_query: str = Field(..., description="Cleaned and normalized query")
    expanded_query: str = Field(..., description="Query with synonyms")
    english_query: Optional[str] = Field(default=None, description="English translation/version")
    detected_keywords: List[str] = Field(default=[], description="Detected sustainability keywords")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_query": "sürdürülebilirlik hedefleri",
                "cleaned_query": "sürdürülebilirlik hedefleri",
                "expanded_query": "sürdürülebilirlik hedefleri sustainability çevre environment",
                "english_query": "sustainability targets",
                "detected_keywords": ["sürdürülebilirlik", "hedef"]
            }
        }
    )


class SimilarityMatrix(BaseModel):
    """Similarity computation results."""
    
    query_embedding: List[float] = Field(..., description="Query vector embedding")
    chunk_similarities: List[float] = Field(..., description="Similarities to all chunks")
    top_indices: List[int] = Field(..., description="Indices of top similar chunks")
    computation_time_ms: float = Field(..., ge=0, description="Computation time in milliseconds")