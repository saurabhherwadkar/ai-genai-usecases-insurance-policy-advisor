"""
LLM client module for the Insurance Policy Advisor.

Provides an async wrapper around the Anthropic Claude API with
retry logic, error handling, and structured message formatting.
"""

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for LLM client operations
logger = get_logger(__name__)


class LLMClient:
    """
    Async wrapper for the Anthropic Claude API.

    Handles message creation with retry logic for transient errors,
    structured prompt formatting, and response parsing.
    """

    def __init__(self) -> None:
        """Initialize the LLMClient with Anthropic client and settings."""
        # Load LLM settings
        settings = get_settings()
        self._model = settings.llm.model
        self._max_tokens = settings.llm.max_tokens
        self._temperature = settings.llm.temperature

        # Initialize the Anthropic async client (uses ANTHROPIC_API_KEY env var)
        self._async_client = anthropic.AsyncAnthropic()

        # Initialize the sync client for non-async contexts
        self._sync_client = anthropic.Anthropic()

        logger.info(f"LLMClient initialized with model={self._model}, max_tokens={self._max_tokens}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
    )
    async def generate_response(self, system_prompt: str, user_message: str) -> str:
        """
        Generate a response from Claude using the provided prompts.

        Sends the system prompt and user message to Claude and returns
        the generated response text. Includes retry logic for transient errors.

        Args:
            system_prompt: The system-level instruction for Claude.
            user_message: The user's message or question with context.

        Returns:
            The generated response text from Claude.
        """
        try:
            logger.debug(f"Sending request to Claude (model={self._model})")

            # Create the message with Claude API
            response = await self._async_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract the text response
            response_text = response.content[0].text

            logger.info(
                f"Received response from Claude "
                f"(input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens})"
            )

            return response_text

        except anthropic.AuthenticationError as auth_error:
            logger.error(f"Authentication failed: {str(auth_error)}")
            raise
        except anthropic.RateLimitError as rate_error:
            logger.warning(f"Rate limit hit, retrying: {str(rate_error)}")
            raise
        except anthropic.InternalServerError as server_error:
            logger.warning(f"Server error, retrying: {str(server_error)}")
            raise
        except anthropic.APIError as api_error:
            logger.error(f"API error: {str(api_error)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
    )
    def generate_response_sync(self, system_prompt: str, user_message: str) -> str:
        """
        Synchronous version of generate_response for non-async contexts.

        Sends the system prompt and user message to Claude and returns
        the generated response text. Used in ingestion and other sync flows.

        Args:
            system_prompt: The system-level instruction for Claude.
            user_message: The user's message or question with context.

        Returns:
            The generated response text from Claude.
        """
        try:
            logger.debug(f"Sending sync request to Claude (model={self._model})")

            # Create the message with Claude API (synchronous)
            response = self._sync_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract the text response
            response_text = response.content[0].text

            logger.info(
                f"Received sync response from Claude "
                f"(input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens})"
            )

            return response_text

        except anthropic.AuthenticationError as auth_error:
            logger.error(f"Authentication failed: {str(auth_error)}")
            raise
        except anthropic.RateLimitError as rate_error:
            logger.warning(f"Rate limit hit, retrying: {str(rate_error)}")
            raise
        except anthropic.InternalServerError as server_error:
            logger.warning(f"Server error, retrying: {str(server_error)}")
            raise
        except anthropic.APIError as api_error:
            logger.error(f"API error: {str(api_error)}")
            raise

    async def generate_with_history(self, system_prompt: str, messages: list[dict]) -> str:
        """
        Generate a response with conversation history.

        Supports multi-turn conversations by passing the full
        message history to Claude.

        Args:
            system_prompt: The system-level instruction for Claude.
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The generated response text from Claude.
        """
        try:
            logger.debug(f"Sending request with {len(messages)} messages to Claude")

            # Create the message with full conversation history
            response = await self._async_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=messages,
            )

            # Extract the text response
            response_text = response.content[0].text

            logger.info(
                f"Received response with history "
                f"(input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens})"
            )

            return response_text

        except anthropic.APIError as api_error:
            logger.error(f"API error with history: {str(api_error)}")
            raise
