"""
Chat endpoint for the Insurance Policy Advisor API.

Handles customer questions by routing them through the RAG
pipeline and returning AI-generated answers with source citations.
"""

from fastapi import APIRouter, HTTPException

from src.agent.chat_agent import ChatAgent
from src.api.models.schemas import ChatRequest, ChatResponseModel
from src.utils.logger import get_logger

# Module-level logger for chat endpoint
logger = get_logger(__name__)

# Create the router for chat endpoints
router = APIRouter(tags=["chat"])

# Module-level chat agent instance (initialized on first use)
_chat_agent: ChatAgent = None


def _get_chat_agent() -> ChatAgent:
    """
    Get or create the singleton ChatAgent instance.

    Lazily initializes the chat agent on first request to avoid
    loading models at import time.

    Returns:
        The ChatAgent singleton instance.
    """
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent


@router.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest) -> ChatResponseModel:
    """
    Process a customer question and return an AI-generated answer.

    Retrieves relevant context from the vector store and knowledge
    graph, then generates an answer using Claude.

    Args:
        request: ChatRequest containing the question and options.

    Returns:
        ChatResponseModel with the answer, sources, and context flags.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        logger.info(f"Chat request received: '{request.question[:80]}...'")

        # Get the chat agent instance
        agent = _get_chat_agent()

        # Generate the answer using the full RAG pipeline
        response = await agent.answer_question(
            question=request.question,
            use_graph=request.use_graph,
        )

        # Convert to API response model
        return ChatResponseModel(
            answer=response.answer,
            sources=response.sources,
            rag_context_used=response.rag_context_used,
            graph_context_used=response.graph_context_used,
        )

    except Exception as error:
        logger.error(f"Error processing chat request: {str(error)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(error)}",
        )
