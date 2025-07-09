"""
Integration tests for the complete RAG system.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.rag_pipeline import RAGPipeline
from src.models.api_models import QueryRequest, DocumentInput
from src.config.settings import get_settings


@pytest.mark.integration
class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.fixture
    def integration_pipeline(self, mock_azure_openai_client, mock_qdrant_client):
        """Create RAG pipeline with mocked external dependencies for integration testing."""
        with patch('src.core.rag_pipeline.get_azure_client') as mock_get_azure, \
             patch('src.core.rag_pipeline.QdrantClient') as mock_qdrant_class:
            
            mock_get_azure.return_value = mock_azure_openai_client
            mock_qdrant_class.return_value = mock_qdrant_client
            
            pipeline = RAGPipeline()
            return pipeline
    
    @pytest.mark.asyncio
    async def test_complete_document_to_query_workflow(self, integration_pipeline, sample_sustainability_document):
        """Test complete workflow from document ingestion to query processing."""
        
        # Step 1: Ingest document
        ingest_result = await integration_pipeline.ingest_document(sample_sustainability_document)
        
        assert ingest_result is not None
        assert "chunks_processed" in ingest_result
        assert ingest_result["chunks_processed"] > 0
        assert ingest_result["status"] == "success"
        
        # Step 2: Process multiple related queries
        test_queries = [
            QueryRequest(
                query="What are NTT DATA's carbon reduction achievements?",
                filters={"type": "sustainability_report"}
            ),
            QueryRequest(
                query="How much renewable energy does NTT DATA use?",
                filters={"section": "environmental_initiatives"}
            ),
            QueryRequest(
                query="What diversity and inclusion metrics are reported?",
                limit=3
            )
        ]
        
        for query in test_queries:
            response = await integration_pipeline.process_query(query)
            
            assert response is not None
            assert "answer" in response
            assert "sources" in response
            assert len(response["answer"]) > 0
            assert isinstance(response["sources"], list)
            
            # Verify sources contain relevant information
            for source in response["sources"]:
                assert "source" in source
                assert "content" in source
                assert "score" in source
                assert source["score"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_document_ingestion(self, integration_pipeline):
        """Test ingesting multiple documents and querying across them."""
        
        # Create multiple test documents
        documents = [
            DocumentInput(
                content="NTT DATA 2023 Environmental Report: Carbon emissions reduced by 25% through energy efficiency programs and renewable energy adoption.",
                metadata={"source": "env_report_2023.pdf", "type": "environmental_report", "year": "2023"}
            ),
            DocumentInput(
                content="NTT DATA 2023 Social Impact Report: Community investment of $2.5M in education and digital literacy programs worldwide.",
                metadata={"source": "social_report_2023.pdf", "type": "social_report", "year": "2023"}
            ),
            DocumentInput(
                content="NTT DATA 2022 Sustainability Summary: Previous year baseline metrics for carbon emissions and renewable energy usage.",
                metadata={"source": "sustainability_2022.pdf", "type": "sustainability_report", "year": "2022"}
            )
        ]
        
        # Ingest all documents
        ingest_results = []
        for doc in documents:
            result = await integration_pipeline.ingest_document(doc)
            ingest_results.append(result)
            assert result["status"] == "success"
        
        # Query across multiple documents
        cross_doc_query = QueryRequest(
            query="Compare sustainability initiatives between 2022 and 2023",
            limit=10
        )
        
        response = await integration_pipeline.process_query(cross_doc_query)
        
        assert response is not None
        assert "answer" in response
        assert len(response["sources"]) > 0
        
        # Verify sources come from multiple documents
        source_files = {source["source"] for source in response["sources"]}
        assert len(source_files) > 1  # Should have sources from multiple documents
    
    @pytest.mark.asyncio
    async def test_filtered_queries(self, integration_pipeline, sample_sustainability_document):
        """Test queries with various metadata filters."""
        
        # Ingest document first
        await integration_pipeline.ingest_document(sample_sustainability_document)
        
        # Test different filter combinations
        filter_test_cases = [
            {
                "query": "What environmental initiatives are mentioned?",
                "filters": {"type": "sustainability_report", "year": "2023"},
                "expected_sources": True
            },
            {
                "query": "Social responsibility programs",
                "filters": {"section": "social_responsibility"},
                "expected_sources": True
            },
            {
                "query": "Future commitments",
                "filters": {"type": "sustainability_report", "year": "2023", "section": "future"},
                "expected_sources": False  # Very specific filter might not match
            }
        ]
        
        for test_case in filter_test_cases:
            query = QueryRequest(
                query=test_case["query"],
                filters=test_case["filters"],
                limit=5
            )
            
            response = await integration_pipeline.process_query(query)
            
            assert response is not None
            assert "answer" in response
            
            if test_case["expected_sources"]:
                assert len(response["sources"]) > 0
    
    @pytest.mark.asyncio
    async def test_query_similarity_ranking(self, integration_pipeline, sample_sustainability_document):
        """Test that query results are properly ranked by similarity."""
        
        await integration_pipeline.ingest_document(sample_sustainability_document)
        
        # Test query that should have clear relevance ranking
        query = QueryRequest(
            query="carbon emissions reduction renewable energy",
            limit=5
        )
        
        response = await integration_pipeline.process_query(query)
        
        assert len(response["sources"]) > 0
        
        # Verify sources are ranked by score (descending)
        scores = [source["score"] for source in response["sources"]]
        assert scores == sorted(scores, reverse=True)
        
        # Verify scores are reasonable
        for score in scores:
            assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, integration_pipeline):
        """Test processing of large documents."""
        
        # Create a large document
        large_content = """
        NTT DATA Comprehensive Sustainability Report 2023
        
        """ + "This is additional content about sustainability initiatives. " * 1000
        
        large_document = DocumentInput(
            content=large_content,
            metadata={"source": "large_sustainability_report.pdf", "type": "sustainability_report"}
        )
        
        # Should handle large documents without errors
        result = await integration_pipeline.ingest_document(large_document)
        
        assert result["status"] == "success"
        assert result["chunks_processed"] > 10  # Should create multiple chunks
        
        # Query the large document
        query = QueryRequest(
            query="sustainability initiatives",
            limit=5
        )
        
        response = await integration_pipeline.process_query(query)
        assert response is not None
        assert len(response["answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_pipeline, sample_sustainability_document):
        """Test concurrent document ingestion and querying."""
        
        # Prepare multiple documents
        documents = []
        for i in range(3):
            doc = DocumentInput(
                content=f"NTT DATA sustainability document {i+1}. Environmental initiatives and carbon reduction programs are key focus areas.",
                metadata={"source": f"doc_{i+1}.pdf", "type": "sustainability_report"}
            )
            documents.append(doc)
        
        # Ingest documents concurrently
        ingest_tasks = [
            integration_pipeline.ingest_document(doc) for doc in documents
        ]
        
        ingest_results = await asyncio.gather(*ingest_tasks)
        
        for result in ingest_results:
            assert result["status"] == "success"
        
        # Process queries concurrently
        queries = [
            QueryRequest(query=f"sustainability topic {i+1}") 
            for i in range(3)
        ]
        
        query_tasks = [
            integration_pipeline.process_query(query) for query in queries
        ]
        
        query_results = await asyncio.gather(*query_tasks)
        
        for result in query_results:
            assert result is not None
            assert "answer" in result
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, integration_pipeline):
        """Test system recovery from various error conditions."""
        
        # Test with invalid document
        invalid_doc = DocumentInput(
            content="",  # Empty content
            metadata={"source": "empty.pdf"}
        )
        
        with pytest.raises((ValueError, Exception)):
            await integration_pipeline.ingest_document(invalid_doc)
        
        # Test with invalid query
        invalid_query = QueryRequest(query="")  # Empty query
        
        with pytest.raises((ValueError, Exception)):
            await integration_pipeline.process_query(invalid_query)
        
        # Verify system can still process valid requests after errors
        valid_doc = DocumentInput(
            content="Valid sustainability content about NTT DATA initiatives.",
            metadata={"source": "valid.pdf", "type": "sustainability_report"}
        )
        
        result = await integration_pipeline.ingest_document(valid_doc)
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, integration_pipeline):
        """Test that metadata is preserved throughout the pipeline."""
        
        rich_metadata_doc = DocumentInput(
            content="NTT DATA sustainability initiatives focus on environmental protection.",
            metadata={
                "source": "detailed_report.pdf",
                "type": "sustainability_report",
                "year": "2023",
                "department": "Environmental Affairs",
                "author": "Sustainability Team",
                "language": "en",
                "region": "Global",
                "classification": "Public"
            }
        )
        
        await integration_pipeline.ingest_document(rich_metadata_doc)
        
        query = QueryRequest(
            query="environmental protection initiatives",
            filters={"department": "Environmental Affairs", "year": "2023"}
        )
        
        response = await integration_pipeline.process_query(query)
        
        assert len(response["sources"]) > 0
        
        # Verify metadata is preserved in sources
        for source in response["sources"]:
            assert source["source"] == "detailed_report.pdf"
            # Additional metadata should be preserved in source metadata
    
    @pytest.mark.asyncio
    async def test_query_context_consistency(self, integration_pipeline, sample_sustainability_document):
        """Test that responses are consistent with document context."""
        
        await integration_pipeline.ingest_document(sample_sustainability_document)
        
        # Test queries that should have consistent answers based on document content
        consistency_tests = [
            {
                "query": "What percentage of carbon emissions reduction was achieved?",
                "expected_content": ["25%", "carbon", "reduction"]
            },
            {
                "query": "What percentage of operations use renewable energy?",
                "expected_content": ["60%", "renewable", "energy"]
            },
            {
                "query": "How much was invested in community development?",
                "expected_content": ["2.5 million", "community", "development"]
            }
        ]
        
        for test in consistency_tests:
            query = QueryRequest(query=test["query"])
            response = await integration_pipeline.process_query(query)
            
            answer = response["answer"].lower()
            
            # Check that expected content appears in the answer
            for expected in test["expected_content"]:
                assert expected.lower() in answer, f"Expected '{expected}' not found in answer for query '{test['query']}'"


@pytest.mark.integration
class TestSystemPerformance:
    """Performance and load testing for the RAG system."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_response_time_benchmarks(self, integration_pipeline, sample_sustainability_document):
        """Test that system meets response time requirements."""
        import time
        
        # Ingest document and measure time
        start_time = time.time()
        await integration_pipeline.ingest_document(sample_sustainability_document)
        ingest_time = time.time() - start_time
        
        # Ingestion should complete within reasonable time
        assert ingest_time < 30.0  # 30 seconds max for ingestion
        
        # Query processing time
        query = QueryRequest(query="What are the main sustainability initiatives?")
        
        start_time = time.time()
        response = await integration_pipeline.process_query(query)
        query_time = time.time() - start_time
        
        # Query should complete within reasonable time
        assert query_time < 10.0  # 10 seconds max for query
        assert response is not None
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_load(self, integration_pipeline, sample_sustainability_document):
        """Test system under concurrent load."""
        
        await integration_pipeline.ingest_document(sample_sustainability_document)
        
        # Simulate concurrent users
        num_concurrent_queries = 10
        queries = [
            QueryRequest(query=f"sustainability query variant {i}")
            for i in range(num_concurrent_queries)
        ]
        
        start_time = time.time()
        
        # Execute all queries concurrently
        tasks = [integration_pipeline.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Verify all queries completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == num_concurrent_queries
        
        # Total time should be reasonable (not much longer than single query)
        assert total_time < 60.0  # Should complete within 1 minute
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, integration_pipeline):
        """Test that memory usage remains stable under load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple documents
        for i in range(5):
            doc = DocumentInput(
                content=f"Sustainability document {i} with environmental data and carbon metrics. " * 100,
                metadata={"source": f"doc_{i}.pdf", "type": "sustainability_report"}
            )
            
            await integration_pipeline.ingest_document(doc)
            
            # Force garbage collection
            gc.collect()
        
        # Process multiple queries
        for i in range(10):
            query = QueryRequest(query=f"sustainability metrics query {i}")
            await integration_pipeline.process_query(query)
            
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB limit


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
