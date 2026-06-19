"""
Unit tests for the TextSplitter module.

Tests text chunking with configurable sizes, overlap behavior,
and edge cases like empty text and very short documents.
"""

import pytest

from src.ingestion.text_splitter import TextSplitter, TextChunk


class TestTextSplitter:
    """Test suite for the TextSplitter class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a splitter with small chunk size for testing
        self.splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    def test_split_short_text_returns_single_chunk(self) -> None:
        """Test that text shorter than chunk_size produces a single chunk."""
        text = "This is a short sentence."

        result = self.splitter.split(text)

        # Short text should produce exactly one chunk
        assert len(result) == 1
        assert isinstance(result[0], TextChunk)
        assert "short sentence" in result[0].content

    def test_split_long_text_returns_multiple_chunks(self) -> None:
        """Test that text longer than chunk_size produces multiple chunks."""
        # Create text that exceeds the chunk size
        text = "This is sentence one about coverage. " * 10

        result = self.splitter.split(text)

        # Should produce more than one chunk
        assert len(result) > 1
        # All results should be TextChunk instances
        assert all(isinstance(chunk, TextChunk) for chunk in result)

    def test_split_empty_text_returns_empty_list(self) -> None:
        """Test that empty text returns an empty list."""
        result = self.splitter.split("")

        assert result == []

    def test_split_whitespace_only_returns_empty_list(self) -> None:
        """Test that whitespace-only text returns an empty list."""
        result = self.splitter.split("   \n\t  ")

        assert result == []

    def test_chunks_preserve_metadata(self) -> None:
        """Test that metadata is passed through to each chunk."""
        text = "This is a test document about insurance coverage."
        metadata = {"filename": "test.txt", "format": ".txt"}

        result = self.splitter.split(text, metadata=metadata)

        # Each chunk should have the source metadata
        assert len(result) >= 1
        assert result[0].metadata["filename"] == "test.txt"
        assert result[0].metadata["format"] == ".txt"

    def test_chunks_have_sequential_indexes(self) -> None:
        """Test that chunks are assigned sequential chunk_index values."""
        text = "Sentence one about dwelling coverage. " * 10

        result = self.splitter.split(text)

        # Verify sequential indexing
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_chunk_metadata_includes_char_count(self) -> None:
        """Test that each chunk's metadata includes its character count."""
        text = "Coverage A provides dwelling protection up to the policy limit."

        result = self.splitter.split(text)

        # Each chunk should have char_count in metadata
        assert len(result) >= 1
        assert "char_count" in result[0].metadata
        assert result[0].metadata["char_count"] > 0

    def test_custom_chunk_size_is_respected(self) -> None:
        """Test that a custom chunk_size limits chunk content length."""
        # Use a very small chunk size
        small_splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This sentence is about property damage coverage limits. " * 5

        result = small_splitter.split(text)

        # Should produce multiple chunks due to small size
        assert len(result) > 1
