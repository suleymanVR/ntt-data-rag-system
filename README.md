# NTT DATA AI Case - RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with FastAPI and AutoGen multi-agent framework for analyzing NTT DATA sustainability reports.

## 🚀 Features

- **Multi-Agent RAG**: Utilizes AutoGen framework with specialized agents for retrieval, analysis, synthesis, and quality assurance
- **Azure OpenAI Integration**: Leverages Azure OpenAI for embeddings and chat completions with Managed Identity support
- **Document Processing**: Advanced PDF and text processing with intelligent chunking
- **Vector Search**: ChromaDB-powered semantic search with similarity matching
- **REST API**: Comprehensive FastAPI-based REST API with automatic documentation
- **Health Monitoring**: Built-in health checks and system monitoring
- **Security**: Production-ready security features including CORS, rate limiting, and security headers
- **Containerization**: Docker and Docker Compose support for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   AutoGen        │    │   Azure OpenAI  │
│   REST API      │◄──►│   Multi-Agent    │◄──►│   GPT-4 & Ada   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text          │    │   ChromaDB       │    │   Document      │
│   Processing    │    │   Vector Store   │    │   Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Multi-Agent System

- **DocumentRetriever**: Evaluates and suggests search strategy improvements
- **DataAnalyst**: Analyzes retrieved information for patterns and insights  
- **ResponseSynthesizer**: Creates comprehensive, well-structured responses
- **QualityAssurance**: Validates accuracy and completeness of responses

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- Azure OpenAI Service account
- Git

## 🛠️ Installation

### Local Development Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd NTT_DATA_AI_CASE
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

5. **Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

6. **Run the application:**
```bash
python main.py
```

### Docker Setup

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Access the application:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | Yes | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes* | - |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | GPT model deployment name | Yes | gpt-4 |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | Yes | text-embedding-ada-002 |
| `CHROMA_HOST` | ChromaDB host | No | localhost |
| `CHROMA_PORT` | ChromaDB port | No | 8001 |
| `CHUNK_SIZE` | Text chunk size | No | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | No | 200 |
| `MAX_FILE_SIZE_MB` | Max upload size | No | 50 |

*Required unless using Managed Identity

### Azure Authentication

The system supports multiple authentication methods:

1. **API Key**: Set `AZURE_OPENAI_API_KEY` in environment
2. **Managed Identity**: For Azure-hosted deployments
3. **Service Principal**: Set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`

## 📚 API Documentation

### Core Endpoints

#### Query Documents
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are NTT DATA's sustainability initiatives?",
  "query_type": "sustainability",
  "max_chunks": 10,
  "temperature": 0.7
}
```

#### Upload Document
```http
POST /api/v1/upload
Content-Type: multipart/form-data

file: [PDF/TXT file]
metadata: [optional JSON metadata]
```

#### Health Check
```http
GET /health
```

### Response Format

```json
{
  "query": "Your question",
  "answer": "Comprehensive AI-generated response",
  "retrieved_chunks": [
    {
      "chunk_id": "chunk_123",
      "content": "Relevant text content",
      "similarity_score": 0.95,
      "document_name": "sustainability_report_2023.pdf",
      "page_number": 5
    }
  ],
  "processing_time": 2.34,
  "confidence_score": 0.87,
  "suggested_followup": [
    "Related question 1",
    "Related question 2"
  ]
}
```

## 🔧 Development

### Project Structure

```
NTT_DATA_AI_CASE/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routes/            # API route handlers
│   │   ├── app.py             # FastAPI app setup
│   │   └── middleware.py      # Custom middleware
│   ├── core/                  # Business logic
│   │   ├── text_processor.py  # Document processing
│   │   ├── embeddings.py      # Embedding management
│   │   ├── retriever.py       # Vector search
│   │   ├── query_processor.py # Query enhancement
│   │   └── rag_pipeline.py    # RAG orchestration
│   ├── models/                # Pydantic models
│   ├── config/                # Configuration
│   ├── utils/                 # Utilities
│   └── exceptions/            # Custom exceptions
├── data/                      # Data storage
│   ├── chroma_db/            # Vector database
│   ├── logs/                 # Application logs
│   └── cache/                # Cache directory
├── reports/                   # PDF reports
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # This file
```

### Adding New Features

1. **New API Endpoint**: Add to `src/api/routes/`
2. **New Model**: Add to `src/models/`
3. **New Business Logic**: Add to `src/core/`
4. **New Configuration**: Update `src/config/settings.py`

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_rag_pipeline.py
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 🚀 Deployment

### Azure Container Apps

1. **Build and push image:**
```bash
docker build -t ntt-data-ai-case .
docker tag ntt-data-ai-case myregistry.azurecr.io/ntt-data-ai-case
docker push myregistry.azurecr.io/ntt-data-ai-case
```

2. **Deploy to Azure Container Apps:**
```bash
az containerapp create \
  --name ntt-data-ai-case \
  --resource-group myResourceGroup \
  --environment myEnvironment \
  --image myregistry.azurecr.io/ntt-data-ai-case \
  --target-port 8000 \
  --ingress external
```

### Environment-Specific Configurations

- **Development**: Full debugging, local ChromaDB
- **Staging**: Reduced logging, shared ChromaDB
- **Production**: Minimal logging, managed services, security hardened

## 📊 Monitoring

### Health Endpoints

- `/health` - Basic health check
- `/health/detailed` - Comprehensive system info
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe
- `/health/metrics` - Basic metrics

### Logging

Structured logging with different formats:
- **Development**: Human-readable console output
- **Production**: JSON format for log aggregation

### Metrics

Basic application metrics available at `/health/metrics`:
- Request counts and timing
- System resource usage
- Service health status
- Cache hit rates

## 🔒 Security

### Features

- **CORS Protection**: Configurable allowed origins
- **Rate Limiting**: Per-IP request limiting
- **Security Headers**: OWASP recommended headers
- **Input Validation**: Pydantic model validation
- **File Upload Security**: Type and size restrictions
- **Error Handling**: No sensitive data in error responses

### Best Practices

- Use Managed Identity in Azure
- Store secrets in Azure Key Vault
- Enable HTTPS in production
- Regular security updates
- Monitor for unusual activity

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for public methods
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Q: ChromaDB connection fails**
A: Check if ChromaDB is running and accessible at the configured host/port

**Q: Azure OpenAI authentication fails**
A: Verify your API key/credentials and endpoint URL

**Q: Document upload fails**
A: Check file size limits and supported file types

**Q: Out of memory errors**
A: Reduce chunk size or increase system memory

### Getting Help

- Check the [API documentation](http://localhost:8000/docs)
- Review application logs in `data/logs/`
- Check health status at `/health/detailed`
- Open an issue on GitHub

## 📋 Changelog

### Version 1.0.0
- Initial release
- Multi-agent RAG system
- Azure OpenAI integration
- Document upload and processing
- Vector search capabilities
- REST API with documentation
- Docker containerization
- Health monitoring
- Security features

---

**Built with ❤️ for NTT DATA AI Case Study**
