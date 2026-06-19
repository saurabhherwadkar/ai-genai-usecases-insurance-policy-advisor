"""
Embeddings generator module for the Insurance Policy Advisor.

Uses sentence-transformers to generate dense vector embeddings
from text chunks. These embeddings are stored in ChromaDB for
semantic similarity search during retrieval.
"""

from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for embedding operations
logger = get_logger(__name__)


class EmbeddingsGenerator:
    """
    Generates vector embeddings from text using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model by default which produces 384-dimensional
    embeddings. The model is loaded once and reused for all embedding requests.
    """

    def __init__(self, model_name: str = None) -> None:
        """
        Initialize the EmbeddingsGenerator with the specified model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to the configured embedding model.
        """
        # Load the model name from settings if not provided
        settings = get_settings()
        self._model_name = model_name or settings.embeddings.model

        # Load the sentence-transformers model
        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)
        logger.info(f"Embedding model loaded successfully (dimension: {self._model.get_sentence_embedding_dimension()})")

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Encodes all texts in a single batch for efficiency.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        # Validate input is not empty
        if not texts:
            logger.warning("Received empty text list for embedding generation")
            return []

        # Generate embeddings using the model in batch mode
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self._model.encode(texts, show_progress_bar=False)

        # Convert numpy arrays to Python lists for JSON serialization compatibility
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        logger.info(f"Generated {len(embeddings_list)} embeddings (dimension: {len(embeddings_list[0])})")
        return embeddings_list

    def generate_single(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Convenience method for embedding a single query or chunk.

        Args:
            text: The text string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        # Validate input is not empty
        if not text or not text.strip():
            logger.warning("Received empty text for single embedding generation")
            return []

        # Generate embedding for the single text
        logger.debug(f"Generating single embedding for text of length {len(text)}")
        embedding = self._model.encode(text, show_progress_bar=False)

        # Convert numpy array to Python list
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this model.

        Returns:
            The number of dimensions in each embedding vector.
        """
        return self._model.get_sentence_embedding_dimension()
