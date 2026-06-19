"""
Unit tests for the EmbeddingsGenerator module.

Tests embedding generation for single texts and batches,
including dimension validation and edge cases.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.rag.embeddings import EmbeddingsGenerator


class TestEmbeddingsGenerator:
    """Test suite for the EmbeddingsGenerator class."""

    @patch("src.rag.embeddings.SentenceTransformer")
    def test_generate_returns_embeddings_for_multiple_texts(self, mock_transformer_class) -> None:
        """Test that generate returns embeddings for a list of texts."""
        # Set up mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_transformer_class.return_value = mock_model

        generator = EmbeddingsGenerator(model_name="test-model")
        texts = ["Coverage A", "Coverage B", "Exclusions"]

        result = generator.generate(texts)

        # Should return one embedding per input text
        assert len(result) == 3
        # Each embedding should have 384 dimensions
        assert len(result[0]) == 384
        # All values should be floats
        assert all(isinstance(v, float) for v in result[0])

    @patch("src.rag.embeddings.SentenceTransformer")
    def test_generate_single_returns_one_embedding(self, mock_transformer_class) -> None:
        """Test that generate_single returns a single embedding vector."""
        # Set up mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer_class.return_value = mock_model

        generator = EmbeddingsGenerator(model_name="test-model")

        result = generator.generate_single("What does my policy cover?")

        # Should return a single vector
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    @patch("src.rag.embeddings.SentenceTransformer")
    def test_generate_empty_list_returns_empty(self, mock_transformer_class) -> None:
        """Test that an empty input list returns an empty result."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model

        generator = EmbeddingsGenerator(model_name="test-model")

        result = generator.generate([])

        assert result == []

    @patch("src.rag.embeddings.SentenceTransformer")
    def test_generate_single_empty_text_returns_empty(self, mock_transformer_class) -> None:
        """Test that empty text input returns an empty list."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model

        generator = EmbeddingsGenerator(model_name="test-model")

        result = generator.generate_single("")

        assert result == []

    @patch("src.rag.embeddings.SentenceTransformer")
    def test_dimension_property_returns_model_dimension(self, mock_transformer_class) -> None:
        """Test that the dimension property returns the model's embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model

        generator = EmbeddingsGenerator(model_name="test-model")

        assert generator.dimension == 384
