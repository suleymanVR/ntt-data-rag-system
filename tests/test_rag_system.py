"""
Comprehensive test suite for the NTT DATA AI Case RAG system.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.text_processor import DocumentProcessor
from src.core.embeddings import EmbeddingManager
from src.core.retriever import DocumentRetriever
from src.core.query_processor import QueryProcessor
from src.core.rag_pipeline import RAGPipeline
from src.models.api_models import QueryRequest, DocumentInput
from src.models.chunk_models import DocumentChunk
from src.models.search_models import QueryAnalysis, SearchResult
from src.config.settings import get_settings


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return DocumentInput(
        content="""
        NTT DATA is committed to sustainability and environmental responsibility. 
        Our sustainability report highlights key initiatives in reducing carbon footprint, 
        promoting renewable energy, and supporting community development. 
        We have implemented various green technologies and sustainable practices 
        across our global operations to minimize environmental impact.
        """,
        metadata={
            "source": "sustainability_report_2023.pdf",
            "type": "sustainability_report",
            "year": "2023",
            "department": "Environmental Affairs"
        }
    )


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            id="chunk_1",
            content="NTT DATA is committed to sustainability and environmental responsibility.",
            source="sustainability_report_2023.pdf",
            chunk_index=0,
            metadata={"type": "sustainability_report", "year": "2023"}
        ),
        DocumentChunk(
            id="chunk_2", 
            content="Our sustainability report highlights key initiatives in reducing carbon footprint.",
            source="sustainability_report_2023.pdf",
            chunk_index=1,
            metadata={"type": "sustainability_report", "year": "2023"}
        )
    ]


@pytest.fixture
def mock_azure_client():
    """Mock Azure OpenAI client."""
    mock_client = AsyncMock()
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 512)]  # Mock 1536-dim embedding
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    mock_completion_response = Mock()
    mock_completion_response.choices = [
        Mock(message=Mock(content="This is a test response about sustainability initiatives."))
    ]
    mock_client.chat.completions.create.return_value = mock_completion_response
    
    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    mock_client = AsyncMock()
    mock_search_result = [
        Mock(
            id="chunk_1",
            score=0.95,
            payload={
                "content": "NTT DATA is committed to sustainability",
                "source": "sustainability_report_2023.pdf",
                "metadata": {"type": "sustainability_report"}
            }
        )
    ]
    mock_client.search.return_value = mock_search_result
    mock_client.upsert.return_value = Mock(operation_id=123, status="completed")
    return mock_client


class TestDocumentProcessor:
    """Test the document processing functionality."""
    
    def test_text_chunking_basic(self):
        """Test basic text chunking functionality."""
        processor = DocumentProcessor()
        
        text = "This is a test document. " * 50  # Create a longer text
        chunks = processor.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # Allow for overlap
        
    def test_text_chunking_short_text(self):
        """Test chunking with text shorter than chunk size."""
        processor = DocumentProcessor()
        
        text = "Short text."
        chunks = processor.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        
    def test_document_processing(self, sample_document):
        """Test full document processing."""
        processor = DocumentProcessor()
        
        chunks = processor.process_document(sample_document)
        
        assert len(chunks) >= 1
        assert all(chunk.source == sample_document.metadata["source"] for chunk in chunks)
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
    def test_document_processing_empty_content(self):
        """Test document processing with empty content."""
        processor = DocumentProcessor()
        
        doc_input = DocumentInput(
            content="",
            metadata={"source": "empty_doc.txt"}
        )
        
        chunks = processor.process_document(doc_input)
        assert len(chunks) == 0
        
    def test_metadata_preservation(self, sample_document):
        """Test that metadata is preserved during processing."""
        processor = DocumentProcessor()
        
        chunks = processor.process_document(sample_document)
        
        for chunk in chunks:
            assert chunk.source == sample_document.metadata["source"]
            assert "type" in chunk.metadata
            assert chunk.metadata["type"] == sample_document.metadata["type"]


class TestEmbeddingManager:
    """Test embedding functionality."""
    
    @pytest.fixture
    def embedding_manager(self, mock_azure_client):
        """Create EmbeddingManager with mocked Azure client."""
        with patch('src.core.embeddings.get_azure_client') as mock_get_client:
            mock_get_client.return_value = mock_azure_client
            return EmbeddingManager()
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_manager, mock_azure_client):
        """Test embedding generation."""
        text = "Test text for embedding"
        
        embedding = await embedding_manager.get_embedding(text)
        
        assert embedding is not None
        assert len(embedding) > 0
        mock_azure_client.embeddings.create.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedding_manager, mock_azure_client, sample_chunks):
        """Test batch embedding generation."""
        texts = [chunk.content for chunk in sample_chunks]
        
        embeddings = await embedding_manager.get_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
        
    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, embedding_manager, mock_azure_client):
        """Test embedding error handling."""
        mock_azure_client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            await embedding_manager.get_embedding("test text")


class TestDocumentRetriever:
    """Test document retrieval functionality."""
    
    @pytest.fixture
    def document_retriever(self, mock_qdrant_client):
        """Create DocumentRetriever with mocked Qdrant client."""
        with patch('src.core.retriever.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = mock_qdrant_client
            retriever = DocumentRetriever()
            retriever.client = mock_qdrant_client
            return retriever
    
    @pytest.mark.asyncio
    async def test_document_storage(self, document_retriever, mock_qdrant_client, sample_chunks):
        """Test storing documents in vector database."""
        # Mock embeddings
        embeddings = [[0.1, 0.2, 0.3] * 512 for _ in sample_chunks]
        
        result = await document_retriever.store_documents(sample_chunks, embeddings)
        
        assert result is not None
        mock_qdrant_client.upsert.assert_called()
        
    @pytest.mark.asyncio
    async def test_document_search(self, document_retriever, mock_qdrant_client):
        """Test searching documents."""
        query_embedding = [0.1, 0.2, 0.3] * 512
        
        results = await document_retriever.search_similar_documents(
            query_embedding, 
            limit=5
        )
        
        assert len(results) > 0
        assert all(isinstance(result, SearchResult) for result in results)
        mock_qdrant_client.search.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_search_with_filters(self, document_retriever, mock_qdrant_client):
        """Test searching with metadata filters."""
        query_embedding = [0.1, 0.2, 0.3] * 512
        filters = {"type": "sustainability_report", "year": "2023"}
        
        results = await document_retriever.search_similar_documents(
            query_embedding,
            limit=5,
            filters=filters
        )
        
        assert len(results) > 0
        mock_qdrant_client.search.assert_called()


class TestQueryProcessor:
    """Test query processing functionality."""
    
    def test_query_analysis_basic(self):
        """Test basic query analysis."""
        processor = QueryProcessor()
        
        query = "What are NTT DATA's sustainability initiatives?"
        analyzed = processor.analyze_query(query)
        
        assert isinstance(analyzed, QueryAnalysis)
        assert analyzed.original_query == query
        assert analyzed.processed_query is not None
        assert analyzed.intent is not None
        
    def test_query_analysis_empty_query(self):
        """Test query analysis with empty query."""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError):
            processor.analyze_query("")
            
    def test_query_analysis_special_characters(self):
        """Test query analysis with special characters."""
        processor = QueryProcessor()
        
        query = "What are NTT DATA's CSR initiatives? (2023 report)"
        analyzed = processor.analyze_query(query)
        
        assert analyzed.original_query == query
        assert analyzed.processed_query is not None
        
    def test_intent_detection(self):
        """Test intent detection in queries."""
        processor = QueryProcessor()
        
        # Test different types of queries
        queries = [
            "What is sustainability?",  # informational
            "Show me carbon reduction data",  # data request
            "How does NTT DATA reduce emissions?",  # process question
            "Compare 2022 and 2023 metrics"  # comparison
        ]
        
        for query in queries:
            analyzed = processor.analyze_query(query)
            assert analyzed.intent is not None
            assert len(analyzed.intent) > 0


class TestRAGPipeline:
    """Test the complete RAG pipeline."""
    
    @pytest.fixture
    def rag_pipeline(self, mock_azure_client, mock_qdrant_client):
        """Create RAG pipeline with mocked dependencies."""
        with patch('src.core.rag_pipeline.get_azure_client') as mock_get_azure, \
             patch('src.core.rag_pipeline.QdrantClient') as mock_qdrant:
            
            mock_get_azure.return_value = mock_azure_client
            mock_qdrant.return_value = mock_qdrant_client
            
            pipeline = RAGPipeline()
            return pipeline
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, rag_pipeline):
        """Test RAG pipeline initialization."""
        assert rag_pipeline is not None
        assert hasattr(rag_pipeline, 'embedding_manager')
        assert hasattr(rag_pipeline, 'retriever')
        assert hasattr(rag_pipeline, 'query_processor')
        
    @pytest.mark.asyncio
    async def test_document_ingestion(self, rag_pipeline, sample_document, mock_azure_client):
        """Test document ingestion process."""
        # Mock embedding response
        mock_azure_client.embeddings.create.return_value.data = [
            Mock(embedding=[0.1, 0.2, 0.3] * 512)
        ]
        
        result = await rag_pipeline.ingest_document(sample_document)
        
        assert result is not None
        assert "chunks_processed" in result
        assert result["chunks_processed"] >= 1
        
    @pytest.mark.asyncio
    async def test_query_processing_end_to_end(self, rag_pipeline, mock_azure_client, mock_qdrant_client):
        """Test end-to-end query processing."""
        query = QueryRequest(
            query="What are NTT DATA's sustainability initiatives?",
            filters={"type": "sustainability_report"}
        )
        
        # Mock embedding response
        mock_azure_client.embeddings.create.return_value.data = [
            Mock(embedding=[0.1, 0.2, 0.3] * 512)
        ]
        
        # Mock completion response
        mock_azure_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="NTT DATA focuses on carbon reduction and renewable energy."))
        ]
        
        response = await rag_pipeline.process_query(query)
        
        assert response is not None
        assert "answer" in response
        assert "sources" in response
        assert len(response["answer"]) > 0
        
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, rag_pipeline, mock_azure_client):
        """Test pipeline error handling."""
        # Mock API failure
        mock_azure_client.embeddings.create.side_effect = Exception("API Error")
        
        query = QueryRequest(query="Test query")
        
        with pytest.raises(Exception):
            await rag_pipeline.process_query(query)


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow(self, mock_azure_client, mock_qdrant_client, sample_document):
        """Test complete workflow from document ingestion to query processing."""
        with patch('src.core.rag_pipeline.get_azure_client') as mock_get_azure, \
             patch('src.core.rag_pipeline.QdrantClient') as mock_qdrant:
            
            mock_get_azure.return_value = mock_azure_client
            mock_qdrant.return_value = mock_qdrant_client
            
            # Setup mock responses
            mock_azure_client.embeddings.create.return_value.data = [
                Mock(embedding=[0.1, 0.2, 0.3] * 512)
            ]
            
            mock_azure_client.chat.completions.create.return_value.choices = [
                Mock(message=Mock(content="Comprehensive sustainability response"))
            ]
            
            # Initialize pipeline
            pipeline = RAGPipeline()
            
            # Ingest document
            ingest_result = await pipeline.ingest_document(sample_document)
            assert ingest_result["chunks_processed"] > 0
            
            # Process query
            query = QueryRequest(query="What sustainability initiatives are mentioned?")
            query_result = await pipeline.process_query(query)
            
            assert query_result["answer"] is not None
            assert len(query_result["answer"]) > 0
            
    def test_error_propagation(self):
        """Test that errors are properly propagated through the system."""
        # Test various error scenarios
        processor = DocumentProcessor()
        
        # Test with None input
        with pytest.raises((TypeError, AttributeError)):
            processor.process_document(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
