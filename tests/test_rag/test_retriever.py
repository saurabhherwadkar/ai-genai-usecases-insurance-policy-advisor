"""
Unit tests for the Retriever module.

Tests retrieval with relevance filtering and context string formatting.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.rag.retriever import Retriever, RetrievalResult


class TestRetriever:
    """Test suite for the Retriever class."""

    @patch("src.rag.retriever.VectorStore")
    def test_retrieve_returns_filtered_results(self, mock_vector_store_class) -> None:
        """Test that retrieve returns results filtered by distance threshold."""
        # Set up mock vector store
        mock_vs = MagicMock()
        mock_vs.query.return_value = [
            {"content": "Coverage A", "metadata": {"filename": "home.txt"}, "distance": 0.3, "id": "chunk_0"},
            {"content": "Exclusion B", "metadata": {"filename": "home.txt"}, "distance": 0.8, "id": "chunk_1"},
            {"content": "Irrelevant", "metadata": {"filename": "auto.txt"}, "distance": 2.0, "id": "chunk_2"},
        ]

        # Create retriever with max_distance of 1.5
        retriever = Retriever(vector_store=mock_vs, max_distance=1.5)
        results = retriever.retrieve("What is covered?")

        # Should filter out the result with distance > 1.5
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].relevance_score == 0.3
        assert results[1].relevance_score == 0.8

    @patch("src.rag.retriever.VectorStore")
    def test_retrieve_empty_query_returns_empty(self, mock_vector_store_class) -> None:
        """Test that an empty query returns an empty list."""
        mock_vs = MagicMock()
        retriever = Retriever(vector_store=mock_vs)

        results = retriever.retrieve("")

        assert results == []

    @patch("src.rag.retriever.VectorStore")
    def test_get_context_string_formats_sources(self, mock_vector_store_class) -> None:
        """Test that get_context_string returns formatted context with sources."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = [
            {"content": "Coverage A details", "metadata": {"filename": "home.txt"}, "distance": 0.2, "id": "chunk_0"},
            {"content": "Coverage B details", "metadata": {"filename": "auto.txt"}, "distance": 0.4, "id": "chunk_1"},
        ]

        retriever = Retriever(vector_store=mock_vs, max_distance=1.5)
        context = retriever.get_context_string("What is covered?")

        # Should contain source references and content
        assert "Source 1: home.txt" in context
        assert "Source 2: auto.txt" in context
        assert "Coverage A details" in context
        assert "Coverage B details" in context

    @patch("src.rag.retriever.VectorStore")
    def test_get_context_string_returns_empty_for_no_results(self, mock_vector_store_class) -> None:
        """Test that get_context_string returns empty string when no results found."""
        mock_vs = MagicMock()
        mock_vs.query.return_value = []

        retriever = Retriever(vector_store=mock_vs)
        context = retriever.get_context_string("unknown topic")

        assert context == ""
