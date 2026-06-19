"""
API schema definitions for the Insurance Policy Advisor.

Defines Pydantic models for all API request and response payloads,
ensuring type safety and automatic validation for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for the /chat endpoint.

    Attributes:
        question: The customer's insurance question.
        use_graph: Whether to include GraphRAG context in retrieval.
    """

    question: str = Field(
        description="The customer's insurance question",
        min_length=1,
        max_length=2000,
        examples=["What does my homeowners policy cover?"],
    )
    use_graph: bool = Field(
        default=True,
        description="Whether to include GraphRAG knowledge graph context",
    )


class ChatResponseModel(BaseModel):
    """
    Response model for the /chat endpoint.

    Attributes:
        answer: The generated answer from the AI advisor.
        sources: List of source documents used for the answer.
        rag_context_used: Whether vector RAG context was found.
        graph_context_used: Whether graph context was found.
    """

    answer: str = Field(description="The AI-generated answer to the question")
    sources: list[str] = Field(default_factory=list, description="Source documents referenced")
    rag_context_used: bool = Field(description="Whether vector RAG context was available")
    graph_context_used: bool = Field(description="Whether graph context was available")


class IngestRequest(BaseModel):
    """
    Request model for the /ingest endpoint.

    Attributes:
        directory_path: Optional custom directory path to ingest from.
        rebuild_graph: Whether to rebuild the knowledge graph after ingestion.
    """

    directory_path: str = Field(
        default=None,
        description="Optional custom directory path. Defaults to configured path.",
    )
    rebuild_graph: bool = Field(
        default=True,
        description="Whether to rebuild the knowledge graph after ingestion",
    )


class IngestResponse(BaseModel):
    """
    Response model for the /ingest endpoint.

    Attributes:
        status: Status of the ingestion operation.
        documents_processed: Number of documents that were ingested.
        chunks_created: Total number of text chunks created.
        graph_nodes: Number of nodes in the knowledge graph.
        graph_edges: Number of edges in the knowledge graph.
    """

    status: str = Field(description="Status of the ingestion operation")
    documents_processed: int = Field(description="Number of documents ingested")
    chunks_created: int = Field(description="Total text chunks created")
    graph_nodes: int = Field(default=0, description="Number of knowledge graph nodes")
    graph_edges: int = Field(default=0, description="Number of knowledge graph edges")


class HealthResponse(BaseModel):
    """
    Response model for the /health endpoint.

    Attributes:
        status: Health status of the application.
        version: Application version string.
        vector_store_count: Number of documents in the vector store.
        graph_nodes: Number of nodes in the knowledge graph.
    """

    status: str = Field(description="Application health status")
    version: str = Field(description="Application version")
    vector_store_count: int = Field(default=0, description="Documents in vector store")
    graph_nodes: int = Field(default=0, description="Nodes in knowledge graph")
