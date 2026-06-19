"""
Ingestion endpoint for the Insurance Policy Advisor API.

Handles document ingestion requests, orchestrating the full pipeline
from loading documents through embedding storage and graph construction.
"""

from fastapi import APIRouter, HTTPException

from src.api.models.schemas import IngestRequest, IngestResponse
from src.graph_rag.entity_extractor import EntityExtractor
from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.graph_store import GraphStore
from src.ingestion.pipeline import IngestionPipeline
from src.rag.vector_store import VectorStore
from src.utils.logger import get_logger

# Module-level logger for ingest endpoint
logger = get_logger(__name__)

# Create the router for ingestion endpoints
router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Ingest documents into the RAG system.

    Loads documents from the specified directory, splits them into
    chunks, stores embeddings in the vector store, and optionally
    builds the knowledge graph.

    Args:
        request: IngestRequest with directory path and graph rebuild options.

    Returns:
        IngestResponse with ingestion statistics.

    Raises:
        HTTPException: If an error occurs during ingestion.
    """
    try:
        logger.info("Document ingestion request received")

        # Step 1: Run the ingestion pipeline to load and chunk documents
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_directory(directory_path=request.directory_path)

        # Return early if no chunks were created
        if not chunks:
            logger.warning("No chunks created from ingestion")
            return IngestResponse(
                status="completed",
                documents_processed=0,
                chunks_created=0,
                graph_nodes=0,
                graph_edges=0,
            )

        # Step 2: Store chunks in the vector store
        logger.info(f"Storing {len(chunks)} chunks in vector store")
        vector_store = VectorStore()
        vector_store.delete_collection()
        chunks_stored = vector_store.add_chunks(chunks)

        # Step 3: Build the knowledge graph if requested
        graph_nodes = 0
        graph_edges = 0
        if request.rebuild_graph:
            logger.info("Building knowledge graph from chunks")
            graph_nodes, graph_edges = _build_knowledge_graph(chunks)

        # Count unique documents processed
        unique_documents = len(set(
            chunk.metadata.get("filename", "unknown") for chunk in chunks
        ))

        # Build response with ingestion statistics
        response = IngestResponse(
            status="completed",
            documents_processed=unique_documents,
            chunks_created=chunks_stored,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )

        logger.info(
            f"Ingestion complete: {unique_documents} docs, {chunks_stored} chunks, "
            f"{graph_nodes} nodes, {graph_edges} edges"
        )

        return response

    except Exception as error:
        logger.error(f"Error during document ingestion: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(error)}",
        )


def _build_knowledge_graph(chunks: list) -> tuple[int, int]:
    """
    Build the knowledge graph from ingested chunks.

    Extracts entities and relationships from each chunk using Claude,
    then builds and persists the knowledge graph.

    Args:
        chunks: List of TextChunk objects to extract entities from.

    Returns:
        Tuple of (node_count, edge_count) for the built graph.
    """
    try:
        # Initialize graph components
        entity_extractor = EntityExtractor()
        graph_builder = GraphBuilder()
        graph_store = GraphStore()

        # Extract entities from each chunk
        texts = [chunk.content for chunk in chunks]
        references = [f"{chunk.metadata.get('filename', 'unknown')}_chunk_{chunk.chunk_index}" for chunk in chunks]

        # Run batch extraction
        extraction_results = entity_extractor.extract_batch(texts=texts, source_references=references)

        # Build the graph from extraction results
        graph_builder.build_from_extractions(extraction_results)

        # Persist the graph
        graph_store.save(graph_builder.graph)

        return graph_builder.node_count, graph_builder.edge_count

    except Exception as error:
        logger.error(f"Error building knowledge graph: {str(error)}")
        return 0, 0
