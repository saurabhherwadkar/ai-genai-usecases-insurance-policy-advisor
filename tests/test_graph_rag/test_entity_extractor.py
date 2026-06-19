"""
Unit tests for the EntityExtractor module.

Tests entity extraction with mocked Anthropic API responses.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from src.graph_rag.entity_extractor import EntityExtractor, ExtractionResult, ExtractedEntity


class TestEntityExtractor:
    """Test suite for the EntityExtractor class."""

    @patch("src.graph_rag.entity_extractor.anthropic.Anthropic")
    def test_extract_returns_entities_and_relationships(self, mock_anthropic_class, sample_extraction_response) -> None:
        """Test that extract parses Claude's response into entities and relationships."""
        # Set up mock Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(sample_extraction_response))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        extractor = EntityExtractor()
        result = extractor.extract("Coverage A covers the dwelling up to $350,000.")

        # Verify entities were extracted
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 4
        assert len(result.relationships) == 3

        # Verify entity types
        entity_types = [e.entity_type for e in result.entities]
        assert "POLICY" in entity_types
        assert "COVERAGE" in entity_types
        assert "EXCLUSION" in entity_types

    @patch("src.graph_rag.entity_extractor.anthropic.Anthropic")
    def test_extract_empty_text_returns_empty_result(self, mock_anthropic_class) -> None:
        """Test that empty text returns an empty extraction result."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        extractor = EntityExtractor()
        result = extractor.extract("")

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    @patch("src.graph_rag.entity_extractor.anthropic.Anthropic")
    def test_extract_handles_malformed_json_response(self, mock_anthropic_class) -> None:
        """Test that malformed JSON from Claude is handled gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        extractor = EntityExtractor()
        result = extractor.extract("Some policy text")

        # Should return empty result without raising
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 0

    @patch("src.graph_rag.entity_extractor.anthropic.Anthropic")
    def test_extract_handles_json_in_markdown_code_block(self, mock_anthropic_class, sample_extraction_response) -> None:
        """Test that JSON wrapped in markdown code blocks is parsed correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Claude sometimes wraps JSON in ```json ... ``` blocks
        mock_response.content = [MagicMock(text=f"```json\n{json.dumps(sample_extraction_response)}\n```")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        extractor = EntityExtractor()
        result = extractor.extract("Coverage text")

        # Should still parse correctly
        assert len(result.entities) == 4

    @patch("src.graph_rag.entity_extractor.anthropic.Anthropic")
    def test_extract_batch_processes_multiple_chunks(self, mock_anthropic_class, sample_extraction_response) -> None:
        """Test that extract_batch processes all provided chunks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(sample_extraction_response))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        extractor = EntityExtractor()
        texts = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
        results = extractor.extract_batch(texts)

        # Should return one result per input text
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)
