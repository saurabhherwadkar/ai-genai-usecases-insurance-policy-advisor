"""
Chat agent module for the Insurance Policy Advisor.

Orchestrates the full RAG pipeline: receives a user question,
retrieves relevant context from both the vector store and knowledge
graph, builds a prompt, and generates an answer using Claude.
"""

from pydantic import BaseModel, Field

from src.agent.llm_client import LLMClient
from src.agent.prompt_templates import PromptTemplates
from src.graph_rag.graph_retriever import GraphRetriever
from src.rag.retriever import Retriever
from src.utils.logger import get_logger

# Module-level logger for chat agent operations
logger = get_logger(__name__)


class ChatResponse(BaseModel):
    """
    Represents the complete response from the chat agent.

    Attributes:
        answer: The generated answer text from Claude.
        sources: List of source document references used.
        rag_context_used: Whether vector RAG context was used.
        graph_context_used: Whether GraphRAG context was used.
    """

    answer: str = Field(description="The generated answer from Claude")
    sources: list[str] = Field(default_factory=list, description="Source documents referenced")
    rag_context_used: bool = Field(default=False, description="Whether RAG context was available")
    graph_context_used: bool = Field(default=False, description="Whether graph context was available")


class ChatAgent:
    """
    Orchestrates the RAG + GraphRAG pipeline for answering questions.

    Coordinates retrieval from both the vector store and knowledge graph,
    merges the context, builds a prompt using templates, and calls Claude
    to generate the final answer.
    """

    def __init__(
        self,
        retriever: Retriever = None,
        graph_retriever: GraphRetriever = None,
        llm_client: LLMClient = None,
    ) -> None:
        """
        Initialize the ChatAgent with its component services.

        Args:
            retriever: Optional vector-based Retriever instance.
            graph_retriever: Optional GraphRetriever instance.
            llm_client: Optional LLMClient instance.
        """
        # Initialize the vector-based retriever
        self._retriever = retriever or Retriever()

        # Initialize the graph-based retriever
        self._graph_retriever = graph_retriever or GraphRetriever()

        # Initialize the LLM client for Claude communication
        self._llm_client = llm_client or LLMClient()

        # Initialize prompt templates
        self._prompt_templates = PromptTemplates()

        logger.info("ChatAgent initialized with RAG and GraphRAG retrieval")

    async def answer_question(self, question: str, use_graph: bool = True) -> ChatResponse:
        """
        Generate an answer to a customer's insurance question.

        Retrieves context from vector store and optionally from the
        knowledge graph, then generates an answer using Claude.

        Args:
            question: The customer's insurance question.
            use_graph: Whether to include GraphRAG context. Defaults to True.

        Returns:
            A ChatResponse with the answer, sources, and context flags.
        """
        logger.info(f"Processing question: '{question[:80]}...'")

        # Step 1: Retrieve context from vector store (standard RAG)
        rag_context = self._get_rag_context(question)

        # Step 2: Optionally retrieve context from knowledge graph
        graph_context = ""
        if use_graph:
            graph_context = self._get_graph_context(question)

        # Step 3: Build the prompt with available context
        system_prompt = self._prompt_templates.get_system_prompt()
        user_message = self._prompt_templates.build_user_message(
            question=question,
            rag_context=rag_context,
            graph_context=graph_context,
        )

        # Step 4: Generate response from Claude
        logger.debug("Generating response from Claude")
        answer = await self._llm_client.generate_response(
            system_prompt=system_prompt,
            user_message=user_message,
        )

        # Step 5: Extract source references from retrieval results
        sources = self._extract_sources(question)

        # Build and return the response
        response = ChatResponse(
            answer=answer,
            sources=sources,
            rag_context_used=bool(rag_context),
            graph_context_used=bool(graph_context),
        )

        logger.info(
            f"Answer generated (rag_context={response.rag_context_used}, "
            f"graph_context={response.graph_context_used}, sources={len(sources)})"
        )

        return response

    def _get_rag_context(self, question: str) -> str:
        """
        Retrieve context string from the vector store.

        Args:
            question: The user's question for similarity search.

        Returns:
            Formatted context string from relevant document chunks.
        """
        try:
            context = self._retriever.get_context_string(query=question)
            if context:
                logger.debug(f"RAG context retrieved ({len(context)} chars)")
            else:
                logger.debug("No RAG context found")
            return context
        except Exception as error:
            logger.error(f"Error retrieving RAG context: {str(error)}")
            return ""

    def _get_graph_context(self, question: str) -> str:
        """
        Retrieve context string from the knowledge graph.

        Args:
            question: The user's question for graph traversal.

        Returns:
            Formatted context string from relevant graph entities.
        """
        try:
            result = self._graph_retriever.retrieve(query=question)
            if result.context_string:
                logger.debug(f"Graph context retrieved ({len(result.context_string)} chars)")
            else:
                logger.debug("No graph context found")
            return result.context_string
        except Exception as error:
            logger.error(f"Error retrieving graph context: {str(error)}")
            return ""

    def _extract_sources(self, question: str) -> list[str]:
        """
        Extract source document references from retrieval results.

        Args:
            question: The user's question to retrieve sources for.

        Returns:
            List of source document filenames.
        """
        try:
            results = self._retriever.retrieve(query=question)
            # Extract unique filenames from metadata
            sources = list(set(
                result.metadata.get("filename", "Unknown")
                for result in results
                if result.metadata.get("filename")
            ))
            return sources
        except Exception as error:
            logger.error(f"Error extracting sources: {str(error)}")
            return []
