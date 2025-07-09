# Development Guide

## Project Structure

The NTT DATA AI Case RAG system is organized as follows:

```
NTT_DATA_AI_CASE/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── routes/               # API route handlers
│   │   │   ├── health.py         # Health check endpoints
│   │   │   └── rag.py            # RAG-specific endpoints
│   │   ├── app.py                # FastAPI app initialization
│   │   └── middleware.py         # Custom middleware
│   ├── config/                   # Configuration management
│   │   ├── settings.py           # Pydantic settings
│   │   └── azure_clients.py      # Azure service clients
│   ├── core/                     # Core business logic
│   │   ├── text_processor.py     # Document processing
│   │   ├── embeddings.py         # Embedding generation
│   │   ├── retriever.py          # Document retrieval
│   │   ├── query_processor.py    # Query analysis
│   │   └── rag_pipeline.py       # Main RAG pipeline
│   ├── models/                   # Data models
│   │   ├── api_models.py         # API request/response models
│   │   ├── chunk_models.py       # Document chunk models
│   │   └── search_models.py      # Search-related models
│   ├── utils/                    # Utility functions
│   │   ├── logger.py             # Logging configuration
│   │   └── health_monitor.py     # Health monitoring
│   └── exceptions/               # Custom exceptions
│       ├── base.py               # Base exception classes
│       └── api_exceptions.py     # API-specific exceptions
├── data/                         # Data storage
│   ├── chroma_db/                # Vector database files
│   ├── logs/                     # Application logs
│   └── cache/                    # Temporary cache files
├── tests/                        # Test suite
├── reports/                      # Generated reports
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Multi-container setup
├── .env                          # Environment variables
├── setup.py                      # Setup script
├── example.py                    # Usage examples
└── README.md                     # Main documentation
```

## Key Components

### 1. Configuration Management (`src/config/`)

- **settings.py**: Centralized configuration using Pydantic Settings
- **azure_clients.py**: Azure service client initialization and management

### 2. Core Processing Logic (`src/core/`)

- **text_processor.py**: Document parsing, cleaning, and chunking
- **embeddings.py**: Text embedding generation using Azure OpenAI
- **retriever.py**: Vector similarity search and document retrieval
- **query_processor.py**: Query analysis and enhancement
- **rag_pipeline.py**: Orchestrates the entire RAG process

### 3. API Layer (`src/api/`)

- **app.py**: FastAPI application setup with middleware and routing
- **middleware.py**: Custom middleware for logging, CORS, and rate limiting
- **routes/**: Modular route handlers for different functionalities

### 4. Data Models (`src/models/`)

- **api_models.py**: Pydantic models for API requests and responses
- **chunk_models.py**: Models for document chunks and metadata
- **search_models.py**: Models for search queries and results

## Development Workflow

### 1. Environment Setup

```bash
# Run the setup script
python setup.py

# Or manually:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Unix/Linux

pip install -r requirements.txt
```

### 2. Configuration

1. Copy `.env.example` to `.env` (or run setup.py)
2. Fill in your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   ```

### 3. Running the Application

```bash
# Development mode
python main.py

# Production mode with Gunicorn (Unix/Linux)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.app:app

# Docker
docker-compose up --build
```

### 4. Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run example script
python example.py
```

## Code Style and Standards

### 1. Python Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use docstrings for all classes and functions

### 2. Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional

# Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from src.config.settings import get_settings
from src.models.api_models import QueryRequest
```

### 3. Error Handling

- Use custom exceptions from `src/exceptions/`
- Always log errors with appropriate context
- Provide meaningful error messages to users
- Handle Azure service errors gracefully

### 4. Logging

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Usage
logger.info("Processing document", document_id=doc_id)
logger.error("Failed to generate embedding", error=str(e))
```

## Testing Strategy

### 1. Unit Tests

- Test individual functions and methods
- Mock external dependencies (Azure APIs, database)
- Focus on business logic and edge cases

### 2. Integration Tests

- Test component interactions
- Use test databases and mock services
- Verify API endpoints work correctly

### 3. Performance Tests

- Load testing for API endpoints
- Memory usage monitoring
- Embedding generation performance

## Deployment

### 1. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Scale services
docker-compose up --scale api=3
```

### 2. Azure Container Apps

```bash
# Deploy to Azure Container Apps
az containerapp up \
  --resource-group myResourceGroup \
  --name ntt-data-rag \
  --ingress external \
  --target-port 8000 \
  --source .
```

### 3. Environment Variables

Ensure these are configured in production:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `SECRET_KEY` (generate a secure key)
- `ENVIRONMENT=production`
- `LOG_LEVEL=WARNING`

## Monitoring and Observability

### 1. Health Checks

- `/health` - Basic health check
- `/health/detailed` - Detailed system status
- `/metrics` - Prometheus metrics

### 2. Logging

- Structured logging with structlog
- File and console output
- Log rotation and retention policies

### 3. Performance Monitoring

- Response time tracking
- Error rate monitoring
- Resource usage metrics

## Security Considerations

### 1. API Security

- Rate limiting on all endpoints
- Input validation with Pydantic
- CORS configuration
- Secret management

### 2. Data Security

- Encrypt sensitive data at rest
- Use Azure Key Vault for secrets
- Implement access controls
- Regular security audits

## Contributing

### 1. Development Process

1. Create a feature branch
2. Write tests for new functionality
3. Implement the feature
4. Run the full test suite
5. Submit a pull request

### 2. Code Review Checklist

- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Security considerations addressed
- [ ] Performance impact assessed

## Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Errors**
   - Verify endpoint and API key
   - Check network connectivity
   - Validate API version

2. **ChromaDB Issues**
   - Ensure data directory is writable
   - Check disk space
   - Verify collection names

3. **Memory Issues**
   - Monitor embedding cache size
   - Adjust chunk sizes
   - Implement document size limits

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## Performance Optimization

### 1. Embedding Optimization

- Batch embedding generation
- Cache frequently used embeddings
- Use appropriate chunk sizes

### 2. Search Optimization

- Index optimization for ChromaDB
- Query result caching
- Parallel processing where possible

### 3. API Optimization

- Response compression
- Connection pooling
- Async/await patterns
