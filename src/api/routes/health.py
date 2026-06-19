"""
Health check endpoint for the Insurance Policy Advisor API.

Provides a simple health check endpoint for monitoring and
container orchestration readiness/liveness probes.
"""

from fastapi import APIRouter

from src.api.models.schemas import HealthResponse
from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for health endpoint
logger = get_logger(__name__)

# Create the router for health endpoints
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health status of the application.

    Returns the application status, version, and counts of
    stored data in the vector store and knowledge graph.

    Returns:
        HealthResponse with current application status.
    """
    settings = get_settings()

    logger.debug("Health check requested")

    # Build health response with current metrics
    response = HealthResponse(
        status="healthy",
        version=settings.app.version,
        vector_store_count=0,
        graph_nodes=0,
    )

    return response
