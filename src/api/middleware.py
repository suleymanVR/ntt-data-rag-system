"""
Middleware configuration for the FastAPI application.
Handles CORS, request logging, error handling, and security.
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import settings
from ..exceptions.api_exceptions import RAGAPIException

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.api.allowed_methods,
        allow_headers=settings.api.allowed_headers,
    )
    
    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request ID middleware
    app.middleware("http")(add_request_id_middleware)
    
    # Request logging middleware
    app.middleware("http")(request_logging_middleware)
    
    # Error handling middleware
    app.middleware("http")(error_handling_middleware)
    
    logger.info("✅ Middleware setup completed")


async def add_request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add unique request ID to each request for tracking."""
    
    # Generate or extract request ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log request and response information."""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    start_time = time.time()
    
    # Log request
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Client: {client_host} - User-Agent: {request.headers.get('user-agent', 'unknown')}"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"[{request_id}] Response: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        
        return response
        
    except Exception as e:
        # Log error
        process_time = time.time() - start_time
        logger.error(
            f"[{request_id}] Error: {str(e)} - "
            f"Time: {process_time:.3f}s"
        )
        raise


async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    """Global error handling middleware."""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        response = await call_next(request)
        return response
        
    except RAGAPIException as e:
        # Handle custom RAG API exceptions
        logger.error(f"[{request_id}] RAG API Error: {e}")
        
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": e.message,
                "error_type": e.__class__.__name__,
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "An unexpected error occurred",
                "error_type": "InternalServerError",
                "request_id": request_id,
                "timestamp": time.time(),
                "detail": str(e) if settings.api.debug else "Internal server error"
            }
        )


class SecurityHeadersMiddleware:
    """Add security headers to responses."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP header for production
        if settings.environment == "production":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'"
            )
        
        return response


class RateLimitMiddleware:
    """Simple rate limiting middleware (for basic protection)."""
    
    def __init__(self, app: FastAPI, max_requests: int = 100, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # In production, use Redis or similar
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_requests(current_time)
        
        # Check rate limit
        if client_id in self.requests:
            request_times = self.requests[client_id]
            recent_requests = [t for t in request_times if current_time - t < self.window_seconds]
            
            if len(recent_requests) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "error_type": "RateLimitExceeded",
                        "retry_after": self.window_seconds
                    },
                    headers={"Retry-After": str(self.window_seconds)}
                )
            
            self.requests[client_id] = recent_requests + [current_time]
        else:
            self.requests[client_id] = [current_time]
        
        return await call_next(request)
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove old request records to prevent memory leaks."""
        cutoff_time = current_time - self.window_seconds * 2
        
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                t for t in self.requests[client_id] 
                if current_time - t < self.window_seconds * 2
            ]
            
            if not self.requests[client_id]:
                del self.requests[client_id]