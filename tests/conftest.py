"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_azure_openai_client():
    """Mock Azure OpenAI client for testing."""
    mock_client = AsyncMock()
    
    # Mock embedding response
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 512)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock completion response
    mock_completion_response = Mock()
    mock_completion_response.choices = [
        Mock(message=Mock(content="This is a test AI response about sustainability."))
    ]
    mock_client.chat.completions.create.return_value = mock_completion_response
    
    # Mock models list
    mock_client.models.list.return_value = Mock(data=[
        Mock(id="text-embedding-ada-002"),
        Mock(id="gpt-4")
    ])
    
    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = AsyncMock()
    
    # Mock search results
    mock_search_result = [
        Mock(
            id="test_chunk_1",
            score=0.95,
            payload={
                "content": "NTT DATA sustainability initiatives focus on carbon reduction",
                "source": "sustainability_report_2023.pdf",
                "metadata": {"type": "sustainability_report", "year": "2023"}
            }
        ),
        Mock(
            id="test_chunk_2",
            score=0.87,
            payload={
                "content": "Renewable energy adoption across global operations",
                "source": "sustainability_report_2023.pdf", 
                "metadata": {"type": "sustainability_report", "year": "2023"}
            }
        )
    ]
    mock_client.search.return_value = mock_search_result
    
    # Mock upsert response
    mock_client.upsert.return_value = Mock(operation_id=123, status="completed")
    
    # Mock collection operations
    mock_client.get_collections.return_value = Mock(collections=[
        Mock(name="ntt_data_docs")
    ])
    mock_client.create_collection.return_value = Mock(status="ok")
    
    return mock_client


@pytest.fixture
def sample_test_environment():
    """Set up test environment variables."""
    test_env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-api-key-12345",
        "AZURE_OPENAI_API_VERSION": "2023-12-01-preview",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_OPENAI_COMPLETION_DEPLOYMENT": "gpt-4",
        "AZURE_OPENAI_EMBEDDING_MODEL": "text-embedding-ada-002",
        "AZURE_OPENAI_COMPLETION_MODEL": "gpt-4",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "test-qdrant-key",
        "QDRANT_COLLECTION_NAME": "test_collection",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG"
    }
    
    original_env = {}
    for key, value in test_env_vars.items():
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = value
    
    yield test_env_vars
    
    # Restore original environment
    for key in test_env_vars.keys():
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]


@pytest.fixture
def sample_sustainability_document():
    """Sample sustainability document data for testing."""
    from src.models.api_models import DocumentInput
    
    return DocumentInput(
        content="""
        NTT DATA Sustainability Report 2023
        
        Executive Summary
        NTT DATA is committed to creating a sustainable future through innovative technology solutions 
        and responsible business practices. Our 2023 sustainability initiatives have focused on three 
        key areas: environmental stewardship, social responsibility, and governance excellence.
        
        Environmental Initiatives
        Carbon Footprint Reduction: We have achieved a 25% reduction in carbon emissions compared to 
        2022 baseline through energy efficiency improvements and renewable energy adoption.
        
        Renewable Energy: 60% of our global operations now run on renewable energy sources, 
        including solar and wind power installations at major data centers.
        
        Waste Management: Implementation of circular economy principles has resulted in 90% waste 
        diversion from landfills through recycling and reuse programs.
        
        Social Responsibility
        Community Development: $2.5 million invested in local community development programs, 
        focusing on education and digital literacy initiatives.
        
        Diversity and Inclusion: 45% of our workforce consists of women and underrepresented groups, 
        with ongoing programs to increase diversity at all organizational levels.
        
        Employee Wellbeing: Comprehensive mental health and wellness programs have been expanded 
        to support employee wellbeing across all global locations.
        
        Governance Excellence
        Ethical Business Practices: Zero tolerance policy for corruption with mandatory ethics 
        training for all employees completed in 2023.
        
        Data Privacy: Enhanced data protection measures implemented in compliance with global 
        privacy regulations including GDPR and CCPA.
        
        Future Commitments
        Net Zero Target: Commitment to achieve net zero carbon emissions by 2030 through 
        continued investment in renewable energy and carbon offset programs.
        
        Sustainable Technology: Development of green technology solutions that help clients 
        reduce their environmental impact while improving operational efficiency.
        """,
        metadata={
            "source": "sustainability_report_2023.pdf",
            "type": "sustainability_report",
            "year": "2023",
            "department": "Environmental Affairs",
            "language": "en",
            "pages": 45,
            "published_date": "2023-12-01"
        }
    )


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    from src.models.chunk_models import DocumentChunk
    
    return [
        DocumentChunk(
            id="chunk_1",
            content="NTT DATA is committed to creating a sustainable future through innovative technology solutions and responsible business practices.",
            source="sustainability_report_2023.pdf",
            chunk_index=0,
            metadata={
                "type": "sustainability_report",
                "year": "2023",
                "section": "executive_summary"
            }
        ),
        DocumentChunk(
            id="chunk_2",
            content="Carbon Footprint Reduction: We have achieved a 25% reduction in carbon emissions compared to 2022 baseline through energy efficiency improvements.",
            source="sustainability_report_2023.pdf",
            chunk_index=1,
            metadata={
                "type": "sustainability_report", 
                "year": "2023",
                "section": "environmental_initiatives"
            }
        ),
        DocumentChunk(
            id="chunk_3",
            content="Renewable Energy: 60% of our global operations now run on renewable energy sources, including solar and wind power installations.",
            source="sustainability_report_2023.pdf",
            chunk_index=2,
            metadata={
                "type": "sustainability_report",
                "year": "2023", 
                "section": "environmental_initiatives"
            }
        )
    ]


@pytest.fixture
def mock_embedding_vectors():
    """Mock embedding vectors for testing."""
    # Generate mock 1536-dimensional embeddings (typical for OpenAI ada-002)
    import numpy as np
    
    embeddings = []
    for i in range(3):  # 3 sample embeddings
        # Create slightly different embeddings for variety
        base_vector = np.random.random(1536) * 0.1
        base_vector[i*100:(i+1)*100] += 0.5  # Make each embedding slightly different
        embeddings.append(base_vector.tolist())
    
    return embeddings


@pytest.fixture
def test_query_requests():
    """Sample query requests for testing."""
    from src.models.api_models import QueryRequest
    
    return [
        QueryRequest(
            query="What are NTT DATA's carbon reduction initiatives?",
            filters={"type": "sustainability_report", "year": "2023"},
            limit=5
        ),
        QueryRequest(
            query="How much renewable energy does NTT DATA use?",
            filters={"section": "environmental_initiatives"},
            limit=3
        ),
        QueryRequest(
            query="What is NTT DATA's diversity and inclusion progress?",
            filters={"type": "sustainability_report"},
            limit=5
        ),
        QueryRequest(
            query="What are the future sustainability commitments?",
            limit=10
        )
    ]


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers during tests
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing."""
    mock_files = {
        "test_document.pdf": b"Mock PDF content",
        "config.json": '{"test": "config"}',
        "data.txt": "Mock text content for testing"
    }
    
    with patch('builtins.open', create=True) as mock_open:
        def side_effect(filename, mode='r', *args, **kwargs):
            if filename in mock_files:
                content = mock_files[filename]
                if 'b' in mode:
                    return Mock(read=lambda: content)
                else:
                    return Mock(read=lambda: content.decode() if isinstance(content, bytes) else content)
            else:
                raise FileNotFoundError(f"No such file: {filename}")
        
        mock_open.side_effect = side_effect
        yield mock_open


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "api", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add api marker to tests in test_api.py
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Add integration marker to tests that use external services
        if any(keyword in item.name.lower() for keyword in ["integration", "e2e", "end_to_end"]):
            item.add_marker(pytest.mark.integration)
