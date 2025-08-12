"""
Main FastAPI application for AI SRE Agent.

This module provides the main FastAPI application with CORS, middleware, error handling,
and route registration for all API endpoints and WebSocket connections.
"""

import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from api.incidents import router as incidents_router
from api.playbooks import router as playbooks_router
from api.analysis import router as analysis_router
from api.websocket import router as websocket_router
# from api.demo_scenarios import router as demo_scenarios_router  # Temporarily disabled
from config.database import init_database, close_database
from config.settings import get_settings
from services.synthetic_data_loader import synthetic_data_loader

settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and timing."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        
        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"-> {response.status_code} ({process_time:.3f}s)"
            )
            
            return response
            
        except Exception as e:
            process_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"-> ERROR ({process_time:.3f}s): {str(e)}"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions (they're handled by FastAPI)
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unhandled exception in {request.method} {request.url.path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    # Startup
    logger.info("Starting AI SRE Agent application...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Load synthetic data for demo
        await synthetic_data_loader.load_all_synthetic_data()
        logger.info("Synthetic data loaded successfully")
        
        # Initialize any other required services
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down AI SRE Agent application...")
        
        try:
            # Close database connections
            await close_database()
            logger.info("Database connections closed")
            
            logger.info("Application shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AI SRE Agent",
    description="AI-powered Site Reliability Engineering Agent for incident analysis and resolution",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-API-Key",
        "X-User-ID"
    ],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Add trusted host middleware for security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add custom middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    
    logger.warning(
        f"HTTP exception in {request.method} {request.url.path}: "
        f"{exc.status_code} - {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    
    logger.warning(f"ValueError in {request.method} {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "status_code": 400,
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 Not Found errors."""
    
    logger.warning(f"404 Not Found: {request.method} {request.url.path}")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "status_code": 404,
            "message": f"The requested resource '{request.url.path}' was not found",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "AI SRE Agent"
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with service status."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "AI SRE Agent",
        "components": {
            "database": "healthy",
            "gcp_monitoring": "healthy" if not settings.USE_MOCK_GCP_SERVICES else "mock",
            "gcp_logging": "healthy" if not settings.USE_MOCK_GCP_SERVICES else "mock",
            "gcp_error_reporting": "healthy" if not settings.USE_MOCK_GCP_SERVICES else "mock"
        },
        "configuration": {
            "debug_mode": settings.DEBUG,
            "mock_gcp_services": settings.USE_MOCK_GCP_SERVICES,
            "database_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "configured",
            "gcp_project": settings.GCP_PROJECT_ID
        }
    }
    
    # Check if any component is unhealthy
    if any(status != "healthy" and status != "mock" for status in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "service": "AI SRE Agent",
        "version": "1.0.0",
        "description": "AI-powered Site Reliability Engineering Agent for incident analysis and resolution",
        "documentation": "/docs" if settings.DEBUG else "Documentation available in debug mode",
        "health": "/health",
        "endpoints": {
            "incidents": "/api/v1/incidents",
            "playbooks": "/api/v1/playbooks",
            "analysis": "/api/v1/analysis",
            "demo_scenarios": "/api/v1/scenarios",
            "websocket": "/ws"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# API route registration
app.include_router(
    incidents_router,
    prefix="/api/v1/incidents",
    tags=["Incidents"]
)

app.include_router(
    playbooks_router,
    prefix="/api/v1/playbooks",
    tags=["Playbooks"]
)

app.include_router(
    analysis_router,
    prefix="/api/v1/analysis",
    tags=["Analysis"]
)

# Demo Scenarios route registration (temporarily disabled)
# app.include_router(
#     demo_scenarios_router,
#     tags=["Demo Scenarios"]
# )

# WebSocket route registration
app.include_router(
    websocket_router,
    prefix="/ws",
    tags=["WebSocket"]
)


# Additional middleware for API versioning
@app.middleware("http")
async def add_api_version_header(request: Request, call_next):
    """Add API version header to all responses."""
    response = await call_next(request)
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Service"] = "AI-SRE-Agent"
    return response


# Request ID middleware for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing."""
    import uuid
    request_id = str(uuid.uuid4())
    
    # Add to request state for use in handlers
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    if not settings.DEBUG:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Rate limiting (basic implementation)
request_counts = {}

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Basic rate limiting middleware."""
    
    if not settings.ENABLE_RATE_LIMITING:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    current_time = datetime.utcnow().timestamp()
    
    # Clean old entries (older than 1 minute)
    cutoff_time = current_time - 60
    request_counts[client_ip] = [
        timestamp for timestamp in request_counts.get(client_ip, [])
        if timestamp > cutoff_time
    ]
    
    # Check rate limit (60 requests per minute)
    if len(request_counts.get(client_ip, [])) >= 60:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "status_code": 429,
                "message": "Rate limit exceeded. Please try again later.",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # Add current request
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    request_counts[client_ip].append(current_time)
    
    return await call_next(request)


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/app.log",
        rotation="100 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    logger.info("Starting AI SRE Agent application...")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG,
        loop="asyncio"
    )
