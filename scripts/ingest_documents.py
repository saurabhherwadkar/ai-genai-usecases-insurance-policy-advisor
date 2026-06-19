"""
CLI script to run the document ingestion pipeline.

Loads insurance policy documents from the configured directory,
splits them into chunks, stores embeddings in ChromaDB, and
builds the knowledge graph.

Usage:
    poetry run python scripts/ingest_documents.py
    poetry run python scripts/ingest_documents.py --directory ./data/sample_policies
    poetry run python scripts/ingest_documents.py --skip-graph
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.graph_rag.entity_extractor import EntityExtractor
from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.graph_store import GraphStore
from src.ingestion.pipeline import IngestionPipeline
from src.rag.vector_store import VectorStore
from src.utils.logger import get_logger

# Script-level logger
logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the ingestion script.

    Returns:
        Parsed argument namespace with directory and graph options.
    """
    parser = argparse.ArgumentParser(
        description="Ingest insurance policy documents into the RAG system"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Path to the directory containing documents (defaults to config value)",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph construction (faster, vector RAG only)",
    )
    return parser.parse_args()


def run_ingestion(directory_path: str = None, build_graph: bool = True) -> None:
    """
    Execute the full document ingestion pipeline.

    Loads documents, creates embeddings, stores in ChromaDB,
    and optionally builds the knowledge graph.

    Args:
        directory_path: Optional custom directory path for documents.
        build_graph: Whether to build the knowledge graph after embedding.
    """
    settings = get_settings()

    # Step 1: Run the ingestion pipeline
    logger.info("=" * 60)
    logger.info("STARTING DOCUMENT INGESTION")
    logger.info("=" * 60)

    pipeline = IngestionPipeline()
    chunks = pipeline.ingest_directory(directory_path=directory_path)

    if not chunks:
        logger.warning("No chunks were created. Check your document directory.")
        return

    logger.info(f"Created {len(chunks)} chunks from documents")

    # Step 2: Store chunks in the vector store
    logger.info("-" * 40)
    logger.info("STORING EMBEDDINGS IN VECTOR STORE")
    logger.info("-" * 40)

    vector_store = VectorStore()
    # Clear existing data for a fresh ingestion
    vector_store.delete_collection()
    chunks_stored = vector_store.add_chunks(chunks)

    logger.info(f"Stored {chunks_stored} chunks in ChromaDB")

    # Step 3: Build the knowledge graph (optional)
    if build_graph:
        logger.info("-" * 40)
        logger.info("BUILDING KNOWLEDGE GRAPH")
        logger.info("-" * 40)

        entity_extractor = EntityExtractor()
        graph_builder = GraphBuilder()
        graph_store = GraphStore()

        # Extract entities from each chunk
        texts = [chunk.content for chunk in chunks]
        references = [
            f"{chunk.metadata.get('filename', 'unknown')}_chunk_{chunk.chunk_index}"
            for chunk in chunks
        ]

        logger.info(f"Extracting entities from {len(texts)} chunks (this may take a while)...")
        extraction_results = entity_extractor.extract_batch(texts=texts, source_references=references)

        # Build the graph from all extractions
        graph_builder.build_from_extractions(extraction_results)

        # Persist the graph to disk
        graph_store.save(graph_builder.graph)

        logger.info(f"Knowledge graph built: {graph_builder.node_count} nodes, {graph_builder.edge_count} edges")
    else:
        logger.info("Skipping knowledge graph construction (--skip-graph)")

    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Documents directory: {directory_path or settings.ingestion.documents_directory}")
    logger.info(f"  Chunks stored: {chunks_stored}")
    if build_graph:
        logger.info(f"  Graph nodes: {graph_builder.node_count}")
        logger.info(f"  Graph edges: {graph_builder.edge_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Run the ingestion pipeline
    run_ingestion(
        directory_path=args.directory,
        build_graph=not args.skip_graph,
    )
