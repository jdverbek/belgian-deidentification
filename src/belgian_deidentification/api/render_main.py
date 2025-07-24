"""Render-optimized FastAPI application for the Belgian deidentification system."""

import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..core.config import get_config, Config
from .routes import router


# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager optimized for Render."""
    # Startup
    logger.info("🚀 Starting Belgian Deidentification API on Render...")
    
    try:
        # Initialize configuration with Render-specific settings
        config = get_config()
        
        # Override with Render environment variables
        render_config = {
            'api': {
                'host': os.getenv('BELGIAN_DEIDENT_API__HOST', '0.0.0.0'),
                'port': int(os.getenv('PORT', os.getenv('BELGIAN_DEIDENT_API__PORT', '10000'))),
                'workers': 1,  # Single worker for Render
                'reload': False,
                'log_level': 'info'
            },
            'debug': os.getenv('BELGIAN_DEIDENT_DEBUG', 'false').lower() == 'true',
            'data_dir': os.getenv('BELGIAN_DEIDENT_DATA_DIR', '/opt/render/project/src/data'),
            'models_dir': os.getenv('BELGIAN_DEIDENT_MODELS_DIR', '/opt/render/project/src/data/models'),
            'temp_dir': os.getenv('BELGIAN_DEIDENT_TEMP_DIR', '/tmp/belgian_deidentification'),
        }
        
        # Update configuration
        config_dict = config.dict()
        config_dict.update(render_config)
        config = Config(**config_dict)
        
        # Ensure directories exist
        config.ensure_directories()
        
        # Initialize pipeline with lightweight configuration
        logger.info("🔧 Initializing deidentification pipeline...")
        from ..core.pipeline import DeidentificationPipeline
        
        # Use a lightweight configuration for Render
        pipeline = DeidentificationPipeline(config)
        
        # Store pipeline in app state
        app.state.pipeline = pipeline
        app.state.config = config
        
        logger.info("✅ API startup completed successfully on Render")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to start application: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Belgian Deidentification API...")


# Create FastAPI application optimized for Render
app = FastAPI(
    title="Belgian Document Deidentification API",
    description="A waterproof deidentification system for Belgian healthcare documents in Dutch",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware optimized for Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


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
            "timestamp": time.time()
        }
    )


# Health check endpoint (required by Render)
@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "platform": "render"
    }


# Readiness check endpoint
@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check if pipeline is available
        if hasattr(app.state, 'pipeline') and app.state.pipeline:
            return {
                "status": "ready",
                "timestamp": time.time(),
                "components": {
                    "pipeline": "ready",
                    "api": "ready"
                }
            }
        else:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring."""
    try:
        if hasattr(app.state, 'pipeline') and app.state.pipeline:
            stats = app.state.pipeline.get_stats()
            
            return {
                "timestamp": time.time(),
                "pipeline_stats": stats,
                "system_info": {
                    "version": "1.0.0",
                    "platform": "render",
                    "debug": getattr(app.state.config, 'debug', False) if hasattr(app.state, 'config') else False
                }
            }
        else:
            return {
                "timestamp": time.time(),
                "error": "Pipeline not available"
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
        "platform": "render",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_base": "/api/v1"
    }


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    # Get port from environment (Render sets this)
    port = int(os.getenv("PORT", "10000"))
    
    # Run the application
    uvicorn.run(
        "belgian_deidentification.api.render_main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        log_level="info",
        access_log=True
    )

