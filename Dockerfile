# Dockerfile for the Insurance Policy Advisor FastAPI backend
# Multi-stage build for optimized production image

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install Poetry for dependency management
RUN pip install --no-cache-dir poetry==1.8.4

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock* ./

# Export dependencies to requirements.txt (without dev deps)
RUN poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt

# Stage 2: Production image
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# Install system dependencies needed at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from builder stage
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create directories for persistent data
RUN mkdir -p /app/chroma_data /app/graph_data /app/logs

# Expose the API port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
