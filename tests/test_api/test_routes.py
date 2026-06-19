"""
Unit tests for the FastAPI routes.

Tests the API endpoints using the FastAPI test client with
mocked service dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

from src.main import app


class TestHealthEndpoint:
    """Test suite for the /api/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self) -> None:
        """Test that the health endpoint returns a 200 status."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_contains_version(self) -> None:
        """Test that the health response includes the app version."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

        data = response.json()
        assert "version" in data
        assert "status" in data
        assert data["status"] == "healthy"


class TestChatEndpoint:
    """Test suite for the /api/chat endpoint."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chat._get_chat_agent")
    async def test_chat_returns_answer(self, mock_get_agent) -> None:
        """Test that the chat endpoint returns an answer."""
        # Mock the chat agent
        mock_agent = AsyncMock()
        mock_agent.answer_question.return_value = MagicMock(
            answer="Your policy covers dwelling damage.",
            sources=["home_policy.txt"],
            rag_context_used=True,
            graph_context_used=True,
        )
        mock_get_agent.return_value = mock_agent

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"question": "What does my policy cover?", "use_graph": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Your policy covers dwelling damage."
        assert "sources" in data

    @pytest.mark.asyncio
    async def test_chat_rejects_empty_question(self) -> None:
        """Test that an empty question is rejected with 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"question": "", "use_graph": True},
            )

        # Pydantic validation should reject empty string (min_length=1)
        assert response.status_code == 422

    @pytest.mark.asyncio
    @patch("src.api.routes.chat._get_chat_agent")
    async def test_chat_handles_agent_error(self, mock_get_agent) -> None:
        """Test that agent errors return 500 status."""
        mock_agent = AsyncMock()
        mock_agent.answer_question.side_effect = Exception("LLM unavailable")
        mock_get_agent.return_value = mock_agent

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"question": "Test question", "use_graph": False},
            )

        assert response.status_code == 500
