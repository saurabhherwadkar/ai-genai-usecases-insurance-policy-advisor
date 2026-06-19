"""
Prompt templates module for the Insurance Policy Advisor.

Contains all system and user prompt templates used by the chat agent.
Templates are structured to guide Claude's responses for insurance
policy Q&A with RAG and GraphRAG context.
"""

from src.utils.logger import get_logger

# Module-level logger for prompt template operations
logger = get_logger(__name__)


# System prompt for the insurance policy advisor chatbot
SYSTEM_PROMPT = """You are an expert Insurance Policy Advisor assistant. Your role is to help customers understand their insurance policies by answering questions about coverage, exclusions, limits, and conditions.

IMPORTANT GUIDELINES:
1. Only answer based on the provided context from insurance policy documents.
2. If the context does not contain information to answer the question, clearly state that the information is not available in the provided documents.
3. Always cite which policy document or section your answer comes from.
4. Be clear and specific about what IS and IS NOT covered.
5. When discussing monetary limits, always include the exact amounts.
6. If a question is ambiguous, ask for clarification.
7. Never provide legal advice - recommend consulting an insurance professional for complex situations.
8. Use simple language that a customer can easily understand.
9. Structure your answers clearly with bullet points when listing multiple items."""

# Template for user message with RAG context only
RAG_CONTEXT_TEMPLATE = """Based on the following insurance policy document excerpts, please answer the customer's question.

DOCUMENT CONTEXT:
{rag_context}

CUSTOMER QUESTION:
{question}

Please provide a clear, accurate answer based solely on the information in the document context above."""

# Template for user message with both RAG and GraphRAG context
COMBINED_CONTEXT_TEMPLATE = """Based on the following insurance policy information, please answer the customer's question.

DOCUMENT EXCERPTS:
{rag_context}

KNOWLEDGE GRAPH CONTEXT (entity relationships):
{graph_context}

CUSTOMER QUESTION:
{question}

Please provide a clear, accurate answer using both the document excerpts and the knowledge graph relationships above. Cite specific policy sections when possible."""

# Template for user message with no context available
NO_CONTEXT_TEMPLATE = """The customer has asked a question, but no relevant insurance policy documents were found in our system.

CUSTOMER QUESTION:
{question}

Please let the customer know that you couldn't find relevant information in the available policy documents. Suggest they:
1. Rephrase their question with more specific terms
2. Contact their insurance agent for detailed policy information
3. Review their actual policy document directly"""


class PromptTemplates:
    """
    Manages prompt template formatting for the chat agent.

    Provides methods to build complete prompts by injecting context
    and questions into the appropriate templates based on available
    retrieval results.
    """

    def __init__(self) -> None:
        """Initialize PromptTemplates with the configured templates."""
        logger.debug("PromptTemplates initialized")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the insurance advisor.

        Returns:
            The system prompt string for Claude.
        """
        return SYSTEM_PROMPT

    def build_user_message(
        self, question: str, rag_context: str = "", graph_context: str = ""
    ) -> str:
        """
        Build the user message with appropriate context template.

        Selects the right template based on which context types
        are available and formats it with the provided data.

        Args:
            question: The customer's question.
            rag_context: Context from vector-based RAG retrieval.
            graph_context: Context from GraphRAG knowledge graph.

        Returns:
            The formatted user message string.
        """
        # Determine which template to use based on available context
        if rag_context and graph_context:
            # Both RAG and graph context available - use combined template
            logger.debug("Using combined context template (RAG + Graph)")
            message = COMBINED_CONTEXT_TEMPLATE.format(
                rag_context=rag_context,
                graph_context=graph_context,
                question=question,
            )
        elif rag_context:
            # Only RAG context available
            logger.debug("Using RAG-only context template")
            message = RAG_CONTEXT_TEMPLATE.format(
                rag_context=rag_context,
                question=question,
            )
        else:
            # No context available
            logger.debug("Using no-context template")
            message = NO_CONTEXT_TEMPLATE.format(question=question)

        return message
