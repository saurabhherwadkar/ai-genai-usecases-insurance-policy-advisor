"""
Vector store module for the Insurance Policy Advisor.

Manages the ChromaDB vector database for storing and querying
document chunk embeddings. Handles collection creation, document
addition, similarity search, and deletion operations.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings
from src.ingestion.text_splitter import TextChunk
from src.rag.embeddings import EmbeddingsGenerator
from src.utils.logger import get_logger

# Module-level logger for vector store operations
logger = get_logger(__name__)


class VectorStore:
    """
    Manages ChromaDB operations for storing and querying embeddings.

    Provides methods to add document chunks with their embeddings,
    perform similarity search, and manage the underlying collection.
    """

    def __init__(self, embeddings_generator: EmbeddingsGenerator = None) -> None:
        """
        Initialize the VectorStore with ChromaDB client and collection.

        Args:
            embeddings_generator: Optional EmbeddingsGenerator instance.
                                 Creates a new one if not provided.
        """
        # Load settings for ChromaDB configuration
        settings = get_settings()
        self._collection_name = settings.vector_store.collection_name
        self._persist_directory = settings.vector_store.persist_directory

        # Initialize the embeddings generator
        self._embeddings_generator = embeddings_generator or EmbeddingsGenerator()

        # Initialize ChromaDB persistent client
        logger.info(f"Initializing ChromaDB at: {self._persist_directory}")
        self._client = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection for insurance policies
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Insurance policy document embeddings"},
        )

        logger.info(
            f"VectorStore initialized: collection='{self._collection_name}', "
            f"existing_documents={self._collection.count()}"
        )

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Add text chunks to the vector store with their embeddings.

        Generates embeddings for all chunks and stores them in ChromaDB
        along with the text content and metadata.

        Args:
            chunks: List of TextChunk objects to store.

        Returns:
            The number of chunks successfully added.
        """
        # Skip if no chunks provided
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return 0

        # Extract text content from chunks for embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self._embeddings_generator.generate(texts)

        # Prepare data for ChromaDB insertion
        ids = [f"chunk_{chunk.metadata.get('filename', 'unknown')}_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to the ChromaDB collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        return len(chunks)

    def query(self, query_text: str, top_k: int = None) -> list[dict]:
        """
        Perform similarity search against stored embeddings.

        Generates an embedding for the query text and finds the
        most similar chunks in the vector store.

        Args:
            query_text: The text query to search for.
            top_k: Number of results to return. Defaults to configured value.

        Returns:
            List of result dictionaries containing content, metadata, and distance.
        """
        # Use configured top_k if not specified
        settings = get_settings()
        num_results = top_k or settings.rag.top_k

        # Generate embedding for the query
        query_embedding = self._embeddings_generator.generate_single(query_text)

        # Return empty results if embedding generation failed
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        # Perform similarity search in ChromaDB
        logger.debug(f"Querying vector store for: '{query_text[:50]}...' (top_k={num_results})")
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(num_results, self._collection.count()),
        )

        # Format results into a list of dictionaries
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                result = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
                formatted_results.append(result)

        logger.info(f"Query returned {len(formatted_results)} results")
        return formatted_results

    def delete_collection(self) -> None:
        """
        Delete the entire collection from ChromaDB.

        Used for resetting the vector store during re-ingestion
        or testing scenarios.
        """
        logger.warning(f"Deleting collection: {self._collection_name}")
        self._client.delete_collection(self._collection_name)

        # Recreate the empty collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Insurance policy document embeddings"},
        )
        logger.info("Collection deleted and recreated")

    @property
    def count(self) -> int:
        """
        Get the total number of documents in the collection.

        Returns:
            The count of stored document chunks.
        """
        return self._collection.count()
