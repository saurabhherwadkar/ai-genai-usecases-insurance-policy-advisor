"""
Ingestion pipeline module for the Insurance Policy Advisor.

Orchestrates the end-to-end document ingestion flow: loading documents
from a directory, splitting them into chunks, and returning the processed
chunks ready for embedding and storage.
"""

from src.config.settings import get_settings
from src.ingestion.document_loader import Document, DocumentLoader
from src.ingestion.text_splitter import TextChunk, TextSplitter
from src.utils.logger import get_logger

# Module-level logger for pipeline operations
logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the document ingestion process.

    Coordinates the document loader and text splitter to transform
    raw files into processed text chunks ready for embedding and
    storage in the vector store and knowledge graph.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        """
        Initialize the IngestionPipeline with its component services.

        Args:
            chunk_size: Optional override for text chunk size.
            chunk_overlap: Optional override for text chunk overlap.
        """
        # Initialize the document loader for reading files
        self._document_loader = DocumentLoader()

        # Initialize the text splitter with configurable parameters
        self._text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Load application settings for directory paths
        self._settings = get_settings()

        logger.info("IngestionPipeline initialized successfully")

    def ingest_directory(self, directory_path: str = None) -> list[TextChunk]:
        """
        Ingest all documents from a directory.

        Loads all supported documents from the specified directory,
        splits each into chunks, and returns the combined list of
        chunks from all documents.

        Args:
            directory_path: Path to the directory containing documents.
                           Defaults to the configured documents directory.

        Returns:
            A list of TextChunk objects from all processed documents.
        """
        # Use configured directory if none specified
        target_directory = directory_path or self._settings.ingestion.documents_directory

        logger.info(f"Starting ingestion from directory: {target_directory}")

        # Load all documents from the directory
        documents = self._document_loader.load_directory(
            directory_path=target_directory,
            supported_formats=self._settings.ingestion.supported_formats,
        )

        # Process each document into chunks
        all_chunks = []
        for document in documents:
            chunks = self._process_document(document)
            all_chunks.extend(chunks)

        logger.info(f"Ingestion complete: {len(documents)} documents -> {len(all_chunks)} chunks")
        return all_chunks

    def ingest_file(self, file_path: str) -> list[TextChunk]:
        """
        Ingest a single document file.

        Loads the document and splits it into chunks.

        Args:
            file_path: Path to the document file to ingest.

        Returns:
            A list of TextChunk objects from the processed document.
        """
        logger.info(f"Ingesting single file: {file_path}")

        # Load the document
        document = self._document_loader.load(file_path)

        # Return empty list if loading failed
        if document is None:
            logger.warning(f"Failed to load file for ingestion: {file_path}")
            return []

        # Process the document into chunks
        chunks = self._process_document(document)

        logger.info(f"File ingestion complete: {file_path} -> {len(chunks)} chunks")
        return chunks

    def _process_document(self, document: Document) -> list[TextChunk]:
        """
        Process a single loaded document into text chunks.

        Splits the document's content using the text splitter,
        passing along the document's metadata to each chunk.

        Args:
            document: The loaded Document object to process.

        Returns:
            A list of TextChunk objects from the document.
        """
        # Split the document content into chunks with metadata preserved
        chunks = self._text_splitter.split(
            text=document.content,
            metadata=document.metadata,
        )

        logger.debug(f"Document '{document.metadata.get('filename', 'unknown')}' split into {len(chunks)} chunks")
        return chunks
