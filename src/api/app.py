"""
FastAPI application setup and configuration.
Main application factory with lifespan management.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health, rag
from .middleware import setup_middleware
from ..core.rag_pipeline import RAGPipeline
from ..config.azure_clients import initialize_azure_clients
from ..config.settings import settings
from ..utils.logger import setup_logging

logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler."""
    # Startup
    global rag_pipeline
    
    logger.info("🚀 Starting NTT DATA RAG System...")
    logger.info(f"🌍 Environment: {settings.environment}")
    
    try:
        # Initialize Azure clients
        initialize_azure_clients()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        success = await rag_pipeline.initialize(settings.directories.reports_dir)
        
        if success:
            logger.info("🎉 RAG System initialized successfully")
        else:
            logger.warning("⚠️ RAG System partially initialized - no documents loaded")
        
        # Store pipeline in app state
        app.state.rag_pipeline = rag_pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("👋 Shutting down RAG system...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "health",
                "description": "Health check and system status endpoints"
            },
            {
                "name": "rag",
                "description": "RAG question-answering endpoints"
            },
            {
                "name": "system",
                "description": "System information endpoints"
            }
        ]
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(health.router, prefix="", tags=["health"])
    app.include_router(rag.router, prefix="", tags=["rag"])
    
    # Root endpoint
    @app.get("/", tags=["system"])
    async def root():
        """System information endpoint."""
        return {
            "message": "NTT DATA RAG API with Azure OpenAI",
            "version": settings.api.version,
            "environment": settings.environment,
            "models": {
                "chat": settings.azure.azure_deployment,
                "embedding": settings.azure.azure_embedding_deployment
            },
            "optimizations": [
                "multi_query_search",
                "chunk_type_analysis", 
                "score_boosting",
                "enhanced_preprocessing",
                "synonym_expansion",
                "bilingual_support",
                "autogen_integration"
            ],
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "ask": "/ask"
            },
            "features": {
                "multi_query": settings.rag.enable_multi_query,
                "synonym_expansion": settings.rag.enable_synonym_expansion,
                "bilingual_support": settings.rag.enable_bilingual_support
            }
        }
    
    return app


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        raise RuntimeError("RAG pipeline not initialized")
    return rag_pipeline


# Create app instance
app = create_app()