"""
Unit tests for the VectorStore module.

Tests ChromaDB operations including adding chunks, querying,
and collection management with mocked embeddings.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.ingestion.text_splitter import TextChunk
from src.rag.vector_store import VectorStore


class TestVectorStore:
    """Test suite for the VectorStore class."""

    @patch("src.rag.vector_store.EmbeddingsGenerator")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    def test_add_chunks_stores_documents(self, mock_chroma_client, mock_embeddings_class) -> None:
        """Test that add_chunks stores chunks in ChromaDB."""
        # Set up mock embeddings generator
        mock_embeddings = MagicMock()
        mock_embeddings.generate.return_value = [[0.1] * 384, [0.2] * 384]
        mock_embeddings_class.return_value = mock_embeddings

        # Set up mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Create test chunks
        chunks = [
            TextChunk(content="Coverage A content", metadata={"filename": "test.txt"}, chunk_index=0),
            TextChunk(content="Coverage B content", metadata={"filename": "test.txt"}, chunk_index=1),
        ]

        # Initialize vector store and add chunks
        vector_store = VectorStore(embeddings_generator=mock_embeddings)
        result = vector_store.add_chunks(chunks)

        # Verify chunks were added
        assert result == 2
        mock_collection.add.assert_called_once()

    @patch("src.rag.vector_store.EmbeddingsGenerator")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    def test_add_empty_chunks_returns_zero(self, mock_chroma_client, mock_embeddings_class) -> None:
        """Test that adding an empty chunk list returns 0."""
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        vector_store = VectorStore(embeddings_generator=mock_embeddings)
        result = vector_store.add_chunks([])

        assert result == 0

    @patch("src.rag.vector_store.EmbeddingsGenerator")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    def test_query_returns_formatted_results(self, mock_chroma_client, mock_embeddings_class) -> None:
        """Test that query returns properly formatted results."""
        # Set up mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.generate_single.return_value = [0.1] * 384
        mock_embeddings_class.return_value = mock_embeddings

        # Set up mock ChromaDB collection with query results
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["Coverage A provides...", "Exclusions include..."]],
            "metadatas": [[{"filename": "home.txt"}, {"filename": "home.txt"}]],
            "distances": [[0.2, 0.5]],
            "ids": [["chunk_0", "chunk_1"]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        vector_store = VectorStore(embeddings_generator=mock_embeddings)
        results = vector_store.query("What is covered?")

        # Verify results are properly formatted
        assert len(results) == 2
        assert results[0]["content"] == "Coverage A provides..."
        assert results[0]["metadata"]["filename"] == "home.txt"
        assert results[0]["distance"] == 0.2

    @patch("src.rag.vector_store.EmbeddingsGenerator")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    def test_query_with_empty_embedding_returns_empty(self, mock_chroma_client, mock_embeddings_class) -> None:
        """Test that a failed embedding generation returns empty results."""
        mock_embeddings = MagicMock()
        mock_embeddings.generate_single.return_value = []
        mock_embeddings_class.return_value = mock_embeddings

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        vector_store = VectorStore(embeddings_generator=mock_embeddings)
        results = vector_store.query("test query")

        assert results == []
