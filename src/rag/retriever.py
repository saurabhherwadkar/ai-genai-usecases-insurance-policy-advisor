"""
Retriever module for the Insurance Policy Advisor.

Provides a high-level interface for retrieving relevant document
chunks from the vector store based on a user's query. Handles
result formatting and relevance filtering.
"""

from pydantic import BaseModel, Field

from src.rag.vector_store import VectorStore
from src.utils.logger import get_logger

# Module-level logger for retrieval operations
logger = get_logger(__name__)


class RetrievalResult(BaseModel):
    """
    Represents a single retrieval result with content and relevance info.

    Attributes:
        content: The text content of the retrieved chunk.
        metadata: Source document metadata for the chunk.
        relevance_score: Similarity score (lower distance = more relevant).
        source_id: Unique identifier of the chunk in the vector store.
    """

    content: str = Field(description="The text content of the retrieved chunk")
    metadata: dict = Field(default_factory=dict, description="Source document metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score (lower = more relevant)")
    source_id: str = Field(default="", description="Unique identifier in the vector store")


class Retriever:
    """
    High-level retrieval interface for the RAG system.

    Wraps the vector store to provide query-based retrieval with
    relevance filtering and result formatting.
    """

    def __init__(self, vector_store: VectorStore = None, max_distance: float = 1.5) -> None:
        """
        Initialize the Retriever with a vector store instance.

        Args:
            vector_store: Optional VectorStore instance. Creates a new one if not provided.
            max_distance: Maximum distance threshold for filtering results.
                         Results with distance above this are considered irrelevant.
        """
        # Initialize or use provided vector store
        self._vector_store = vector_store or VectorStore()

        # Set the maximum distance threshold for relevance filtering
        self._max_distance = max_distance

        logger.info(f"Retriever initialized with max_distance={self._max_distance}")

    def retrieve(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        """
        Retrieve relevant document chunks for a given query.

        Performs semantic search in the vector store and filters
        results by relevance distance threshold.

        Args:
            query: The user's question or search query.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        # Validate query is not empty
        if not query or not query.strip():
            logger.warning("Received empty query for retrieval")
            return []

        # Perform vector similarity search
        logger.info(f"Retrieving chunks for query: '{query[:80]}...'")
        raw_results = self._vector_store.query(query_text=query, top_k=top_k)

        # Filter and format results
        results = []
        for raw_result in raw_results:
            distance = raw_result.get("distance", float("inf"))

            # Filter out results above the distance threshold
            if distance > self._max_distance:
                logger.debug(f"Filtered result with distance {distance:.4f} (above threshold {self._max_distance})")
                continue

            # Create a structured retrieval result
            result = RetrievalResult(
                content=raw_result["content"],
                metadata=raw_result.get("metadata", {}),
                relevance_score=distance,
                source_id=raw_result.get("id", ""),
            )
            results.append(result)

        logger.info(f"Retrieved {len(results)} relevant chunks (filtered from {len(raw_results)} total)")
        return results

    def get_context_string(self, query: str, top_k: int = None) -> str:
        """
        Retrieve chunks and format them as a single context string.

        Convenience method that retrieves relevant chunks and
        concatenates them into a formatted string suitable for
        injection into an LLM prompt.

        Args:
            query: The user's question or search query.
            top_k: Maximum number of results to include.

        Returns:
            A formatted string containing all retrieved chunk contents
            with source attribution.
        """
        # Retrieve relevant chunks
        results = self.retrieve(query=query, top_k=top_k)

        # Return empty string if no results found
        if not results:
            logger.warning("No relevant context found for query")
            return ""

        # Format each result with source information
        context_parts = []
        for i, result in enumerate(results, start=1):
            source = result.metadata.get("filename", "Unknown source")
            context_parts.append(f"[Source {i}: {source}]\n{result.content}")

        # Join all context parts with separator
        context_string = "\n\n---\n\n".join(context_parts)

        logger.debug(f"Built context string with {len(results)} sources ({len(context_string)} chars)")
        return context_string
