"""
Text splitter module for the Insurance Policy Advisor.

Splits large document text into smaller, overlapping chunks suitable
for embedding and retrieval. Uses configurable chunk size and overlap
to maintain context across chunk boundaries.
"""

from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for text splitting operations
logger = get_logger(__name__)


class TextChunk(BaseModel):
    """
    Represents a chunk of text split from a larger document.

    Attributes:
        content: The text content of this chunk.
        metadata: Dictionary containing chunk metadata like position, source document, etc.
        chunk_index: The sequential index of this chunk within its source document.
    """

    content: str = Field(description="The text content of this chunk")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata including source and position info")
    chunk_index: int = Field(default=0, description="Sequential index of this chunk in the document")


class TextSplitter:
    """
    Splits document text into overlapping chunks for RAG processing.

    Uses a sliding window approach with configurable chunk size and
    overlap. Attempts to split at sentence boundaries to avoid
    breaking text mid-sentence.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        """
        Initialize the TextSplitter with chunk size and overlap settings.

        Args:
            chunk_size: Maximum number of characters per chunk. Defaults to config value.
            chunk_overlap: Number of overlapping characters between chunks. Defaults to config value.
        """
        # Load settings and apply defaults from configuration
        settings = get_settings()
        self._chunk_size = chunk_size or settings.rag.chunk_size
        self._chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

        logger.info(f"TextSplitter initialized with chunk_size={self._chunk_size}, overlap={self._chunk_overlap}")

    def split(self, text: str, metadata: dict = None) -> list[TextChunk]:
        """
        Split text into overlapping chunks.

        Uses sentence-aware splitting to avoid breaking text in the
        middle of sentences. Falls back to character-level splitting
        if sentences are longer than the chunk size.

        Args:
            text: The full text content to split into chunks.
            metadata: Optional metadata to attach to each chunk (e.g., source document info).

        Returns:
            A list of TextChunk objects representing the split text.
        """
        # Return empty list for empty or whitespace-only text
        if not text or not text.strip():
            logger.warning("Received empty text for splitting")
            return []

        # Use empty dict if no metadata provided
        base_metadata = metadata or {}

        # Split the text into sentences first
        sentences = self._split_into_sentences(text)

        # Build chunks from sentences with overlap
        chunks = self._build_chunks_from_sentences(sentences, base_metadata)

        logger.info(f"Split text into {len(chunks)} chunks (source: {base_metadata.get('filename', 'unknown')})")
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into individual sentences.

        Uses common sentence-ending punctuation as delimiters while
        preserving the punctuation in the output.

        Args:
            text: The text to split into sentences.

        Returns:
            A list of sentence strings.
        """
        # Define sentence-ending markers
        sentence_endings = [".", "!", "?", "\n\n"]

        # Start with the full text as one segment
        segments = [text]

        # Iteratively split by each sentence ending
        for ending in sentence_endings:
            new_segments = []
            for segment in segments:
                # Split on the ending and re-attach it to each part
                parts = segment.split(ending)
                for i, part in enumerate(parts):
                    if part.strip():
                        # Re-attach the ending marker to all parts except the last
                        if i < len(parts) - 1:
                            new_segments.append(part + ending)
                        else:
                            new_segments.append(part)
            segments = new_segments

        return segments

    def _build_chunks_from_sentences(self, sentences: list[str], base_metadata: dict) -> list[TextChunk]:
        """
        Build overlapping chunks from a list of sentences.

        Accumulates sentences until the chunk size is reached, then
        creates a chunk and starts the next one with overlap from
        the previous chunk's end.

        Args:
            sentences: List of sentence strings to combine into chunks.
            base_metadata: Base metadata to include in each chunk.

        Returns:
            A list of TextChunk objects built from the sentences.
        """
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, finalize the current chunk
            if current_chunk_length + sentence_length > self._chunk_size and current_chunk_sentences:
                # Create the chunk from accumulated sentences
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunk = TextChunk(
                    content=chunk_text,
                    metadata={**base_metadata, "chunk_index": chunk_index, "char_count": len(chunk_text)},
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Calculate overlap by keeping trailing sentences within overlap size
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk_sentences):
                    if overlap_length + len(sent) <= self._chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break

                # Start new chunk with overlap sentences
                current_chunk_sentences = overlap_sentences
                current_chunk_length = overlap_length

            # Add the current sentence to the accumulator
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length

        # Create the final chunk from any remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunk = TextChunk(
                content=chunk_text,
                metadata={**base_metadata, "chunk_index": chunk_index, "char_count": len(chunk_text)},
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        return chunks
