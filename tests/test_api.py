"""
Test suite for API endpoints.
Tests FastAPI routes, middleware, and error handling.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from fastapi.testclient import TestClient

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api.app import create_app
from src.models.api_models import QuestionRequest, AnswerResponse, HealthResponse, ConversationMessage


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def setup_method(self):
        """Set up test client for each test."""
        # Mock all Azure and RAG dependencies
        with patch('src.config.azure_clients.initialize_azure_clients'), \
             patch('src.api.app.RAGPipeline') as mock_rag_class:
            
            # Mock RAG pipeline instance
            mock_rag_instance = Mock()
            mock_rag_instance.initialize = AsyncMock(return_value=True)
            mock_rag_instance.get_system_status.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "documents_loaded": 5,
                "total_chunks": 100,
                "chat_model": "gpt-4",
                "embedding_model": "text-embedding-3-large",
                "initialized": True,
                "embedding_dimension": 3072,
                "chunk_distribution": {"general": 100},
                "optimization_features": ["multi_query"]
            }
            mock_rag_class.return_value = mock_rag_instance
            
            self.app = create_app()
            # Manually set the RAG pipeline in app state
            self.app.state.rag_pipeline = mock_rag_instance
            self.client = TestClient(self.app)
    
    def test_health_endpoint_success(self):
        """Test health endpoint returns success."""
        with patch('src.api.routes.health.is_azure_healthy', return_value=True):
            response = self.client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "documents_loaded" in data
            assert "total_chunks" in data
    
    def test_health_endpoint_azure_unhealthy(self):
        """Test health endpoint when Azure is unhealthy."""
        with patch('src.api.routes.health.is_azure_healthy', return_value=False):
            response = self.client.get("/health")
            
            assert response.status_code == 200  # Still returns 200 but status is degraded
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_status"] == "unhealthy"
    
    def test_ask_endpoint_success(self):
        """Test ask endpoint with successful response."""
        mock_response = {
            "answer": "NTT DATA's sustainability goals include carbon neutrality by 2030.",
            "sources": ["sustainability_report_2023.pdf (Page 15)"],
            "metadata": {
                "chunks_found": 5,
                "search_time_ms": 123.4,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
        # Set up mock for ask_question method
        self.app.state.rag_pipeline.ask_question = AsyncMock(return_value=mock_response)
        
        response = self.client.post(
            "/ask",
            json={"question": "What are NTT DATA's sustainability goals?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == mock_response["answer"]
        assert "sources" in data
        assert "metadata" in data
    
    def test_ask_endpoint_invalid_request(self):
        """Test ask endpoint with invalid request."""
        response = self.client.post(
            "/ask",
            json={"invalid_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_empty_question(self):
        """Test ask endpoint with empty question."""
        response = self.client.post(
            "/ask",
            json={"question": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_rag_error(self):
        """Test ask endpoint when RAG pipeline fails."""
        # Mock the pipeline to raise an exception
        self.app.state.rag_pipeline.ask_question = AsyncMock(side_effect=Exception("RAG pipeline error"))
        
        response = self.client.post(
            "/ask",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"] or "error" in str(data)
    
    def test_ask_endpoint_with_conversation_history(self):
        """Test ask endpoint with conversation history."""
        mock_response = {
            "answer": "You asked about NTT DATA's sustainability goals.",
            "sources": [],
            "metadata": {
                "chunks_found": 2,
                "search_time_ms": 89.5,
                "timestamp": "2024-01-01T00:00:00Z",
                "context_used": True
            }
        }
        
        # Mock conversation history
        conversation_history = [
            {"role": "user", "content": "What are NTT DATA's sustainability goals?"},
            {"role": "assistant", "content": "NTT DATA aims for carbon neutrality by 2030."}
        ]
        
        # Set up mock for ask_question method
        self.app.state.rag_pipeline.ask_question = AsyncMock(return_value=mock_response)
        
        response = self.client.post(
            "/ask",
            json={
                "question": "What did I just ask?",
                "conversation_history": conversation_history,
                "max_chunks": 5,
                "include_metadata": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sustainability goals" in data["answer"].lower()
        assert data["metadata"]["context_used"] is True
    
    def test_ask_endpoint_invalid_conversation_history(self):
        """Test ask endpoint with invalid conversation history format."""
        response = self.client.post(
            "/ask",
            json={
                "question": "Test question",
                "conversation_history": "invalid_format",  # Should be list
                "max_chunks": 5
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_empty_conversation_history(self):
        """Test ask endpoint with empty conversation history."""
        mock_response = {
            "answer": "Test answer without context.",
            "sources": [],
            "metadata": {
                "chunks_found": 3,
                "search_time_ms": 95.2,
                "timestamp": "2024-01-01T00:00:00Z",
                "context_used": False
            }
        }
        
        # Set up mock for ask_question method
        self.app.state.rag_pipeline.ask_question = AsyncMock(return_value=mock_response)
        
        response = self.client.post(
            "/ask",
            json={
                "question": "Test question",
                "conversation_history": [],  # Empty history
                "max_chunks": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == mock_response["answer"]
        assert data["metadata"]["context_used"] is False


class TestAPIModels:
    """Test Pydantic models for API."""
    
    def test_question_request_validation(self):
        """Test QuestionRequest model validation."""
        # Valid request
        valid_request = QuestionRequest(question="What is sustainability?")
        assert valid_request.question == "What is sustainability?"
        
        # Invalid request - empty question should be caught by validator
        with pytest.raises(ValueError):
            QuestionRequest(question="")
    
    def test_answer_response_creation(self):
        """Test AnswerResponse model creation."""
        response = AnswerResponse(
            answer="Test answer",
            sources=[],
            metadata={"processing_time": 1.0}
        )
        
        assert response.answer == "Test answer"
        assert response.sources == []
        assert response.metadata["processing_time"] == 1.0
    
    def test_health_response_creation(self):
        """Test HealthResponse model creation."""
        response = HealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            documents_loaded=5,
            total_chunks=100,
            chat_model="gpt-4",
            embedding_model="text-embedding-3-large",
            model_status="healthy"
        )
        
        assert response.status == "healthy"
        assert response.documents_loaded == 5
        assert response.total_chunks == 100
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage model creation."""
        message = ConversationMessage(
            role="user",
            content="Test message"
        )
        
        assert message.role == "user"
        assert message.content == "Test message"
        
        # Test assistant message
        assistant_message = ConversationMessage(
            role="assistant",
            content="Test response"
        )
        
        assert assistant_message.role == "assistant"
        assert assistant_message.content == "Test response"


class TestAPIMiddleware:
    """Test API middleware functionality."""
    
    def setup_method(self):
        """Set up test client for each test."""
        # Mock all Azure and RAG dependencies
        with patch('src.config.azure_clients.initialize_azure_clients'), \
             patch('src.api.app.RAGPipeline') as mock_rag_class:
            
            # Mock RAG pipeline instance
            mock_rag_instance = Mock()
            mock_rag_instance.initialize = AsyncMock(return_value=True)
            mock_rag_instance.get_system_status.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "documents_loaded": 5,
                "total_chunks": 100,
                "chat_model": "gpt-4",
                "embedding_model": "text-embedding-3-large",
                "initialized": True,
                "embedding_dimension": 3072,
                "chunk_distribution": {"general": 100},
                "optimization_features": ["multi_query"]
            }
            mock_rag_class.return_value = mock_rag_instance
            
            self.app = create_app()
            # Manually set the RAG pipeline in app state
            self.app.state.rag_pipeline = mock_rag_instance
            self.client = TestClient(self.app)
    
    def test_request_logging_middleware(self):
        """Test that requests are logged properly."""
        with patch('src.api.routes.health.is_azure_healthy', return_value=True):
            response = self.client.get("/health")
            
            # Should complete successfully
            assert response.status_code == 200


class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    def setup_method(self):
        """Set up test client for each test."""
        # Mock all Azure and RAG dependencies
        with patch('src.config.azure_clients.initialize_azure_clients'), \
             patch('src.api.app.RAGPipeline') as mock_rag_class:
            
            # Mock RAG pipeline instance
            mock_rag_instance = Mock()
            mock_rag_instance.initialize = AsyncMock(return_value=True)
            mock_rag_instance.get_system_status.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "documents_loaded": 5,
                "total_chunks": 100,
                "chat_model": "gpt-4",
                "embedding_model": "text-embedding-3-large",
                "initialized": True,
                "embedding_dimension": 3072,
                "chunk_distribution": {"general": 100},
                "optimization_features": ["multi_query"]
            }
            mock_rag_class.return_value = mock_rag_instance
            
            self.app = create_app()
            # Manually set the RAG pipeline in app state
            self.app.state.rag_pipeline = mock_rag_instance
            self.client = TestClient(self.app)
            self.client = TestClient(self.app)
    
    def test_404_not_found(self):
        """Test 404 handling for unknown endpoints."""
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self):
        """Test 405 handling for wrong HTTP methods."""
        response = self.client.delete("/health")
        assert response.status_code == 405
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request body."""
        response = self.client.post(
            "/ask",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
