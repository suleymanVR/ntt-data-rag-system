"""
Test suite for API endpoints.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
from httpx import AsyncClient

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from src.api.app import app
from src.models.api_models import QueryRequest, DocumentInput


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def async_client():
    """Async test client for FastAPI app."""
    return AsyncClient(app=app, base_url="http://test")


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    mock_pipeline = AsyncMock()
    
    # Mock successful query response
    mock_pipeline.process_query.return_value = {
        "answer": "NTT DATA focuses on sustainability through various initiatives including carbon reduction, renewable energy adoption, and community development programs.",
        "sources": [
            {
                "source": "sustainability_report_2023.pdf",
                "content": "NTT DATA is committed to sustainability...",
                "score": 0.95
            }
        ],
        "query_id": "test-query-123"
    }
    
    # Mock successful document ingestion
    mock_pipeline.ingest_document.return_value = {
        "status": "success",
        "chunks_processed": 5,
        "document_id": "doc-123"
    }
    
    return mock_pipeline


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        
    def test_health_check_includes_dependencies(self, client):
        """Test that health check includes dependency status."""
        response = client.get("/health")
        data = response.json()
        
        assert "dependencies" in data
        # Should include checks for Azure OpenAI and Qdrant
        dependencies = data["dependencies"]
        assert isinstance(dependencies, dict)


class TestRAGEndpoints:
    """Test RAG-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, async_client, mock_rag_pipeline):
        """Test successful query processing."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            query_data = {
                "query": "What are NTT DATA's sustainability initiatives?",
                "filters": {"type": "sustainability_report"},
                "limit": 5
            }
            
            response = await async_client.post("/api/v1/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "query_id" in data
            assert len(data["answer"]) > 0
            
    @pytest.mark.asyncio 
    async def test_query_endpoint_validation(self, async_client):
        """Test query endpoint input validation."""
        # Test empty query
        response = await async_client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422
        
        # Test missing query
        response = await async_client.post("/api/v1/query", json={})
        assert response.status_code == 422
        
        # Test invalid limit
        response = await async_client.post("/api/v1/query", json={
            "query": "test",
            "limit": -1
        })
        assert response.status_code == 422
        
    @pytest.mark.asyncio
    async def test_query_endpoint_with_filters(self, async_client, mock_rag_pipeline):
        """Test query endpoint with metadata filters."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            query_data = {
                "query": "carbon emissions data",
                "filters": {
                    "type": "sustainability_report",
                    "year": "2023",
                    "department": "Environmental Affairs"
                },
                "limit": 3
            }
            
            response = await async_client.post("/api/v1/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            
            # Verify filters were passed to pipeline
            mock_rag_pipeline.process_query.assert_called_once()
            call_args = mock_rag_pipeline.process_query.call_args[0][0]
            assert call_args.filters == query_data["filters"]
            
    @pytest.mark.asyncio
    async def test_ingest_document_success(self, async_client, mock_rag_pipeline):
        """Test successful document ingestion."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            document_data = {
                "content": "This is a test sustainability report content with multiple paragraphs about environmental initiatives and carbon reduction strategies.",
                "metadata": {
                    "source": "test_report.pdf",
                    "type": "sustainability_report",
                    "year": "2023"
                }
            }
            
            response = await async_client.post("/api/v1/ingest", json=document_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "chunks_processed" in data
            assert "document_id" in data
            
    @pytest.mark.asyncio
    async def test_ingest_document_validation(self, async_client):
        """Test document ingestion input validation."""
        # Test empty content
        response = await async_client.post("/api/v1/ingest", json={
            "content": "",
            "metadata": {"source": "test.pdf"}
        })
        assert response.status_code == 422
        
        # Test missing content
        response = await async_client.post("/api/v1/ingest", json={
            "metadata": {"source": "test.pdf"}
        })
        assert response.status_code == 422
        
        # Test missing metadata
        response = await async_client.post("/api/v1/ingest", json={
            "content": "test content"
        })
        assert response.status_code == 422
        
    @pytest.mark.asyncio
    async def test_query_endpoint_error_handling(self, async_client, mock_rag_pipeline):
        """Test error handling in query endpoint."""
        # Mock pipeline error
        mock_rag_pipeline.process_query.side_effect = Exception("Processing error")
        
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            query_data = {"query": "test query"}
            
            response = await async_client.post("/api/v1/query", json=query_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            
    @pytest.mark.asyncio
    async def test_ingest_endpoint_error_handling(self, async_client, mock_rag_pipeline):
        """Test error handling in ingest endpoint."""
        # Mock pipeline error
        mock_rag_pipeline.ingest_document.side_effect = Exception("Ingestion error")
        
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            document_data = {
                "content": "test content",
                "metadata": {"source": "test.pdf"}
            }
            
            response = await async_client.post("/api/v1/ingest", json=document_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data


class TestMiddleware:
    """Test API middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are properly set."""
        response = await async_client.options("/api/v1/query")
        
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
        
    @pytest.mark.asyncio
    async def test_request_logging(self, async_client):
        """Test that requests are properly logged."""
        with patch('src.api.middleware.logger') as mock_logger:
            response = await async_client.get("/health")
            
            # Verify logging was called
            assert mock_logger.info.called
            
    @pytest.mark.asyncio
    async def test_error_handling_middleware(self, async_client):
        """Test error handling middleware."""
        # This would test custom error handling if implemented
        response = await async_client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestAuthentication:
    """Test authentication and authorization (if implemented)."""
    
    @pytest.mark.skipif(True, reason="Authentication not implemented yet")
    @pytest.mark.asyncio
    async def test_protected_endpoint_without_auth(self, async_client):
        """Test accessing protected endpoint without authentication."""
        response = await async_client.post("/api/v1/admin/status")
        assert response.status_code == 401
        
    @pytest.mark.skipif(True, reason="Authentication not implemented yet") 
    @pytest.mark.asyncio
    async def test_protected_endpoint_with_valid_auth(self, async_client):
        """Test accessing protected endpoint with valid authentication."""
        headers = {"Authorization": "Bearer valid-token"}
        response = await async_client.post("/api/v1/admin/status", headers=headers)
        assert response.status_code == 200


class TestPerformance:
    """Test API performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client, mock_rag_pipeline):
        """Test handling concurrent requests."""
        import asyncio
        
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = async_client.post("/api/v1/query", json={
                    "query": f"test query {i}"
                })
                tasks.append(task)
            
            # Execute concurrently
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                
    @pytest.mark.asyncio
    async def test_large_document_ingestion(self, async_client, mock_rag_pipeline):
        """Test ingesting large documents."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            # Create a large document
            large_content = "This is a test document. " * 1000  # ~25KB
            
            document_data = {
                "content": large_content,
                "metadata": {"source": "large_document.pdf"}
            }
            
            response = await async_client.post("/api/v1/ingest", json=document_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestDataValidation:
    """Test data validation and sanitization."""
    
    @pytest.mark.asyncio
    async def test_query_sanitization(self, async_client, mock_rag_pipeline):
        """Test that queries are properly sanitized."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            # Test with potentially harmful input
            query_data = {
                "query": "<script>alert('xss')</script>What is sustainability?"
            }
            
            response = await async_client.post("/api/v1/query", json=query_data)
            
            assert response.status_code == 200
            # Verify the query was processed (exact sanitization depends on implementation)
            mock_rag_pipeline.process_query.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_metadata_validation(self, async_client, mock_rag_pipeline):
        """Test metadata validation during document ingestion."""
        with patch('src.api.routes.rag.rag_pipeline', mock_rag_pipeline):
            # Test with various metadata formats
            test_cases = [
                {
                    "content": "test",
                    "metadata": {"source": "test.pdf", "valid_field": "value"}
                },
                {
                    "content": "test", 
                    "metadata": {"source": "test.pdf", "number_field": 123}
                },
                {
                    "content": "test",
                    "metadata": {"source": "test.pdf", "list_field": ["a", "b", "c"]}
                }
            ]
            
            for test_data in test_cases:
                response = await async_client.post("/api/v1/ingest", json=test_data)
                assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
