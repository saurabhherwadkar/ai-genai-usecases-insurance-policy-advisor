"""
Unit tests for the ChatAgent module.

Tests the orchestration of RAG, GraphRAG, and LLM for answer generation
with mocked retrieval and API dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agent.chat_agent import ChatAgent, ChatResponse


class TestChatAgent:
    """Test suite for the ChatAgent class."""

    @pytest.mark.asyncio
    async def test_answer_question_with_both_contexts(self) -> None:
        """Test answer generation when both RAG and graph context are available."""
        # Create mocks for dependencies
        mock_retriever = MagicMock()
        mock_retriever.get_context_string.return_value = "Coverage A covers dwelling damage up to $350,000."
        mock_retriever.retrieve.return_value = [
            MagicMock(metadata={"filename": "home_policy.txt"}),
        ]

        mock_graph_retriever = MagicMock()
        mock_graph_result = MagicMock()
        mock_graph_result.context_string = "ENTITIES:\n- [COVERAGE] Dwelling Coverage"
        mock_graph_retriever.retrieve.return_value = mock_graph_result

        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = "Your homeowners policy covers dwelling damage up to $350,000."

        # Create agent with mocked dependencies
        agent = ChatAgent(
            retriever=mock_retriever,
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm,
        )

        # Ask a question
        response = await agent.answer_question("What does my policy cover?")

        # Verify response structure
        assert isinstance(response, ChatResponse)
        assert response.answer != ""
        assert response.rag_context_used is True
        assert response.graph_context_used is True
        assert "home_policy.txt" in response.sources

    @pytest.mark.asyncio
    async def test_answer_question_without_graph(self) -> None:
        """Test answer generation with only RAG context (graph disabled)."""
        mock_retriever = MagicMock()
        mock_retriever.get_context_string.return_value = "Flood damage is excluded."
        mock_retriever.retrieve.return_value = [
            MagicMock(metadata={"filename": "home_policy.txt"}),
        ]

        mock_graph_retriever = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = "Flood damage is not covered under your homeowners policy."

        agent = ChatAgent(
            retriever=mock_retriever,
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm,
        )

        # Ask with graph disabled
        response = await agent.answer_question("Is flood covered?", use_graph=False)

        # Graph should not be used
        assert response.graph_context_used is False
        assert response.rag_context_used is True
        mock_graph_retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_answer_question_no_context_found(self) -> None:
        """Test answer generation when no context is found."""
        mock_retriever = MagicMock()
        mock_retriever.get_context_string.return_value = ""
        mock_retriever.retrieve.return_value = []

        mock_graph_retriever = MagicMock()
        mock_graph_result = MagicMock()
        mock_graph_result.context_string = ""
        mock_graph_retriever.retrieve.return_value = mock_graph_result

        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = "I couldn't find relevant information in the documents."

        agent = ChatAgent(
            retriever=mock_retriever,
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm,
        )

        response = await agent.answer_question("What about spacecraft insurance?")

        # Should still return a response even without context
        assert response.answer != ""
        assert response.rag_context_used is False
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_answer_question_handles_retriever_error(self) -> None:
        """Test that agent handles retriever errors gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.get_context_string.side_effect = Exception("Vector store unavailable")
        mock_retriever.retrieve.return_value = []

        mock_graph_retriever = MagicMock()
        mock_graph_result = MagicMock()
        mock_graph_result.context_string = ""
        mock_graph_retriever.retrieve.return_value = mock_graph_result

        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = "I couldn't retrieve policy information."

        agent = ChatAgent(
            retriever=mock_retriever,
            graph_retriever=mock_graph_retriever,
            llm_client=mock_llm,
        )

        # Should not raise despite retriever error
        response = await agent.answer_question("What is covered?")

        assert response.answer != ""
        assert response.rag_context_used is False
