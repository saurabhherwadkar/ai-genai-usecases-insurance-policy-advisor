"""
Middleware module for the Insurance Policy Advisor API.

Configures CORS, global error handling, and request logging
middleware for the FastAPI application.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

# Module-level logger for middleware operations
logger = get_logger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the FastAPI application.

    Sets up CORS, request logging, and global error handling.

    Args:
        app: The FastAPI application instance to configure.
    """
    # Configure CORS middleware for frontend access
    _setup_cors(app)

    # Configure global exception handler
    _setup_exception_handlers(app)

    # Configure request logging middleware
    _setup_request_logging(app)

    logger.info("Middleware configured successfully")


def _setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware to allow Streamlit frontend access.

    Args:
        app: The FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.debug("CORS middleware configured")


def _setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers for unhandled errors.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Handle any unhandled exceptions with a clean error response.

        Args:
            request: The incoming request that caused the error.
            exc: The unhandled exception.

        Returns:
            JSONResponse with error details and 500 status code.
        """
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred",
                "path": str(request.url.path),
            },
        )


def _setup_request_logging(app: FastAPI) -> None:
    """
    Configure request/response logging middleware.

    Args:
        app: The FastAPI application instance.
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Log incoming requests and outgoing responses.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response from the next handler.
        """
        # Log the incoming request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process the request and get the response
        response = await call_next(request)

        # Log the response status
        logger.info(f"Response: {request.method} {request.url.path} -> {response.status_code}")

        return response
