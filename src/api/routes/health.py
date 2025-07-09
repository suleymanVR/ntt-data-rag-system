"""
Health check endpoints for monitoring system status.
Provides comprehensive health information about the RAG system.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ...models.api_models import HealthResponse
from ...config.azure_clients import is_azure_healthy
from ...utils.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)

router = APIRouter()
health_monitor = HealthMonitor()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Comprehensive health check endpoint.
    
    Returns detailed information about system status including:
    - RAG pipeline status
    - Azure OpenAI connectivity
    - Document and chunk statistics
    - Model configurations
    """
    try:
        # Get RAG pipeline from app state
        rag_pipeline = getattr(request.app.state, 'rag_pipeline', None)
        
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Get system status from RAG pipeline
        system_status = rag_pipeline.get_system_status()
        
        # Check Azure client health
        azure_healthy = is_azure_healthy()
        
        # Determine overall status
        overall_status = "healthy"
        if not azure_healthy:
            overall_status = "degraded"
        if not system_status.get("initialized", False):
            overall_status = "unhealthy"
        
        # Create response
        health_response = HealthResponse(
            status=overall_status,
            timestamp=system_status["timestamp"],
            documents_loaded=system_status["documents_loaded"],
            total_chunks=system_status["total_chunks"],
            chat_model=system_status["chat_model"],
            embedding_model=system_status["embedding_model"],
            model_status="healthy" if azure_healthy else "unhealthy",
            embedding_dimension=system_status.get("embedding_dimension", 0),
            chunk_distribution=system_status.get("chunk_distribution", {}),
            optimization_features=system_status.get("optimization_features", [])
        )
        
        return health_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """
    Detailed health check with additional diagnostic information.
    
    Includes:
    - System metrics
    - Resource usage
    - Performance indicators
    - Diagnostic details
    """
    try:
        # Get RAG pipeline from app state
        rag_pipeline = getattr(request.app.state, 'rag_pipeline', None)
        
        if not rag_pipeline:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": "RAG pipeline not initialized",
                    "components": {
                        "rag_pipeline": "not_initialized",
                        "azure_clients": "unknown",
                        "documents": "unknown"
                    }
                }
            )
        
        # Get detailed system status
        system_status = rag_pipeline.get_system_status()
        
        # Get health monitor metrics
        health_metrics = health_monitor.get_health_metrics()
        
        # Component health checks
        component_health = {
            "rag_pipeline": "healthy" if system_status.get("initialized", False) else "unhealthy",
            "azure_clients": "healthy" if is_azure_healthy() else "unhealthy",
            "documents": "healthy" if system_status["total_chunks"] > 0 else "no_data",
            "retrieval": system_status.get("retrieval_stats", {}).get("status", "unknown"),
        }
        
        # Overall status determination
        overall_status = "healthy"
        if any(status in ["unhealthy", "error"] for status in component_health.values()):
            overall_status = "unhealthy"
        elif any(status in ["degraded", "no_data"] for status in component_health.values()):
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": system_status["timestamp"],
            "components": component_health,
            "system_info": {
                "documents_loaded": system_status["documents_loaded"],
                "total_chunks": system_status["total_chunks"],
                "chunk_distribution": system_status.get("chunk_distribution", {}),
                "embedding_dimension": system_status.get("embedding_dimension", 0)
            },
            "models": {
                "chat_model": system_status["chat_model"],
                "embedding_model": system_status["embedding_model"],
                "model_status": system_status["model_status"]
            },
            "features": {
                "optimization_features": system_status.get("optimization_features", []),
                "retrieval_stats": system_status.get("retrieval_stats", {})
            },
            "metrics": health_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Error in detailed health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": health_monitor.get_current_timestamp()
            }
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Simple liveness probe for Kubernetes/Docker health checks.
    
    Returns 200 if the service is running, regardless of functional status.
    """
    return {"status": "alive", "timestamp": health_monitor.get_current_timestamp()}


@router.get("/health/ready")
async def readiness_probe(request: Request):
    """
    Readiness probe for Kubernetes/Docker readiness checks.
    
    Returns 200 only if the service is ready to handle requests.
    """
    try:
        # Get RAG pipeline from app state
        rag_pipeline = getattr(request.app.state, 'rag_pipeline', None)
        
        if not rag_pipeline:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "RAG pipeline not initialized"}
            )
        
        # Check if system is ready
        system_status = rag_pipeline.get_system_status()
        
        if not system_status.get("initialized", False):
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "RAG pipeline not initialized"}
            )
        
        if not is_azure_healthy():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Azure OpenAI clients not healthy"}
            )
        
        return {"status": "ready", "timestamp": health_monitor.get_current_timestamp()}
        
    except Exception as e:
        logger.error(f"❌ Error in readiness probe: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": f"Health check error: {str(e)}"}
        )