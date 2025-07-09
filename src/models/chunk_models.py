"""
Data models for text chunks and document metadata.
Models used internally for document processing and storage.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ChunkType(str, Enum):
    """Enumeration of chunk types."""
    GENERAL = "general"
    SUSTAINABILITY = "sustainability"
    METRICS = "metrics"
    VISUAL = "visual"
    TITLE = "title"


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    
    source: str = Field(..., description="Source document filename")
    page: int = Field(..., ge=1, description="Page number in source document")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within page")
    total_chunks: int = Field(..., ge=1, description="Total chunks in page")
    chunk_type: ChunkType = Field(default=ChunkType.GENERAL, description="Type/category of chunk")
    has_numbers: bool = Field(default=False, description="Whether chunk contains numerical data")
    has_keywords: bool = Field(default=False, description="Whether chunk contains sustainability keywords")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Chunk creation timestamp")
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "source": "ntt_data_sustainability_report_2020.pdf",
                "page": 15,
                "chunk_id": "ntt_data_sustainability_report_2020.pdf_p15_c2",
                "chunk_index": 2,
                "total_chunks": 5,
                "chunk_type": "sustainability",
                "has_numbers": True,
                "has_keywords": True,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    )


class DocumentChunk(BaseModel):
    """A processed document chunk with content and metadata."""
    
    text: str = Field(..., min_length=1, description="Processed text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the text")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "NTT DATA'nın 2020 yılı sürdürülebilirlik hedefleri arasında karbon emisyonlarını %30 azaltmak yer almaktadır...",
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
                "embedding": None  # Would contain 3072-dimensional vector in practice
            }
        }
    )


class ChunkAnalysis(BaseModel):
    """Analysis results for a text chunk."""
    
    text: str = Field(..., description="Original text")
    chunk_type: ChunkType = Field(..., description="Detected chunk type")
    has_numbers: bool = Field(..., description="Contains numerical data")
    has_keywords: bool = Field(..., description="Contains sustainability keywords")
    word_count: int = Field(..., ge=0, description="Number of words in chunk")
    sustainability_keywords: List[str] = Field(default=[], description="Found sustainability keywords")
    numerical_patterns: List[str] = Field(default=[], description="Found numerical patterns")
    
    model_config = ConfigDict(
        use_enum_values=True
    )


class DocumentInfo(BaseModel):
    """Information about a processed document."""
    
    filename: str = Field(..., description="Document filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    chunk_distribution: Dict[ChunkType, int] = Field(default={}, description="Distribution of chunk types")
    processing_date: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "filename": "ntt_data_sustainability_report_2020.pdf",
                "total_pages": 45,
                "total_chunks": 89,
                "chunk_distribution": {
                    "general": 35,
                    "sustainability": 28,
                    "metrics": 20,
                    "visual": 4,
                    "title": 2
                },
                "processing_date": "2024-01-15T10:30:00Z",
                "file_size": 2548736
            }
        }
    )