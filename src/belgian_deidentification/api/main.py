"""FastAPI application for the Belgian deidentification system."""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..core.config import get_config
from .routes import router
from .middleware import SecurityMiddleware, LoggingMiddleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Belgian Deidentification API...")
    
    # Initialize configuration
    config = get_config()
    config.ensure_directories()
    
    # Validate configuration
    from ..core.pipeline import DeidentificationPipeline
    pipeline = DeidentificationPipeline(config)
    validation_errors = pipeline.validate_configuration()
    
    if validation_errors:
        logger.error(f"Configuration validation failed: {validation_errors}")
        raise RuntimeError("Invalid configuration")
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Belgian Deidentification API...")


# Create FastAPI application
app = FastAPI(
    title="Belgian Document Deidentification API",
    description="A waterproof deidentification system for Belgian healthcare documents in Dutch",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
config = get_config()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


# Readiness check endpoint
@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check if pipeline can be initialized
        from ..core.pipeline import DeidentificationPipeline
        pipeline = DeidentificationPipeline()
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "components": {
                "pipeline": "ready",
                "nlp": "ready",
                "entities": "ready",
                "deidentification": "ready",
                "quality": "ready"
            }
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring."""
    try:
        from ..core.pipeline import DeidentificationPipeline
        pipeline = DeidentificationPipeline()
        stats = pipeline.get_stats()
        
        return {
            "timestamp": time.time(),
            "pipeline_stats": stats,
            "system_info": {
                "version": "1.0.0",
                "config_mode": config.deidentification.mode.value,
                "debug": config.debug
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "timestamp": time.time(),
            "error": "Metrics collection failed"
        }


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Belgian Document Deidentification API",
        "version": "1.0.0",
        "description": "A waterproof deidentification system for Belgian healthcare documents in Dutch",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_base": "/api/v1"
    }


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    # Run the application
    config = get_config()
    
    uvicorn.run(
        "belgian_deidentification.api.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level,
        access_log=True
    )

