"""
Main application entry point for the Insurance Policy Advisor API.

Creates and configures the FastAPI application, registers routes,
sets up middleware, and defines application lifespan events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import setup_middleware
from src.api.routes import chat, health, ingest
from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for application startup
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events (startup and shutdown).

    Initializes resources on startup and cleans up on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    # Startup: log application start and initialize resources
    settings = get_settings()
    logger.info(f"Starting {settings.app.name} v{settings.app.version}")
    logger.info(f"Environment: {settings.app.env}")
    logger.info(f"LLM Model: {settings.llm.model}")

    # Yield control to the application
    yield

    # Shutdown: clean up resources
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Registers all routes, sets up middleware, and configures
    the application metadata.

    Returns:
        The configured FastAPI application instance.
    """
    # Load application settings
    settings = get_settings()

    # Create the FastAPI application
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="AI-powered insurance policy advisor using RAG and GraphRAG",
        lifespan=lifespan,
    )

    # Register API routes
    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(ingest.router, prefix="/api")

    # Setup middleware (CORS, logging, error handling)
    setup_middleware(app)

    logger.info("FastAPI application created and configured")
    return app


# Create the application instance for uvicorn
app = create_app()
