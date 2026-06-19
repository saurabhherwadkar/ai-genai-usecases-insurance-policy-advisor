"""
Unit tests for the DocumentLoader module.

Tests document loading for TXT format and error handling
for missing files and unsupported formats.
"""

import pytest

from src.ingestion.document_loader import DocumentLoader, Document


class TestDocumentLoader:
    """Test suite for the DocumentLoader class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a fresh DocumentLoader instance for each test
        self.loader = DocumentLoader()

    def test_load_txt_file_successfully(self, tmp_path) -> None:
        """Test that a valid TXT file is loaded with correct content and metadata."""
        # Create a test text file
        test_file = tmp_path / "policy.txt"
        test_file.write_text("This is a test insurance policy document.")

        # Load the file
        result = self.loader.load(str(test_file))

        # Verify the document was loaded successfully
        assert result is not None
        assert isinstance(result, Document)
        assert result.content == "This is a test insurance policy document."
        assert result.metadata["filename"] == "policy.txt"
        assert result.metadata["format"] == ".txt"

    def test_load_nonexistent_file_returns_none(self) -> None:
        """Test that loading a nonexistent file returns None."""
        result = self.loader.load("/nonexistent/path/file.txt")

        # Should return None for missing files
        assert result is None

    def test_load_unsupported_format_returns_none(self, tmp_path) -> None:
        """Test that an unsupported file format returns None."""
        # Create a file with unsupported extension
        test_file = tmp_path / "policy.xyz"
        test_file.write_text("Some content")

        result = self.loader.load(str(test_file))

        # Should return None for unsupported formats
        assert result is None

    def test_load_directory_finds_txt_files(self, tmp_documents_dir) -> None:
        """Test that load_directory finds and loads TXT files."""
        # Load all documents from the test directory
        results = self.loader.load_directory(tmp_documents_dir)

        # Should find at least one document
        assert len(results) >= 1
        assert all(isinstance(doc, Document) for doc in results)

    def test_load_directory_nonexistent_returns_empty(self) -> None:
        """Test that a nonexistent directory returns an empty list."""
        results = self.loader.load_directory("/nonexistent/directory")

        # Should return empty list for missing directory
        assert results == []

    def test_load_directory_with_format_filter(self, tmp_path) -> None:
        """Test that load_directory respects the supported_formats filter."""
        # Create files of different types
        (tmp_path / "doc1.txt").write_text("Text document")
        (tmp_path / "doc2.xyz").write_text("Unsupported document")

        # Load only .txt files
        results = self.loader.load_directory(str(tmp_path), supported_formats=[".txt"])

        # Should only find the .txt file
        assert len(results) == 1
        assert results[0].metadata["format"] == ".txt"

    def test_loaded_document_has_size_metadata(self, tmp_path) -> None:
        """Test that loaded documents include file size in metadata."""
        # Create a test file with known content
        test_file = tmp_path / "sized_policy.txt"
        content = "A" * 100
        test_file.write_text(content)

        result = self.loader.load(str(test_file))

        # Verify size_bytes is present and reasonable
        assert result is not None
        assert "size_bytes" in result.metadata
        assert result.metadata["size_bytes"] > 0
