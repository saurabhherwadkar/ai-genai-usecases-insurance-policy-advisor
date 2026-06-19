"""
Streamlit chat frontend for the Insurance Policy Advisor.

Provides a web-based chat interface that allows customers to ask
questions about their insurance policies. Communicates with the
FastAPI backend for RAG-powered answers.
"""

import streamlit as st
import requests

from src.config.settings import get_settings

# Load settings for API URL configuration
settings = get_settings()
API_BASE_URL = settings.ui.api_base_url


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state variables.

    Sets up chat history, settings, and connection state on first load.
    """
    # Initialize chat message history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize graph toggle setting
    if "use_graph" not in st.session_state:
        st.session_state.use_graph = True

    # Initialize API connection status
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False


def check_api_connection() -> bool:
    """
    Check if the FastAPI backend is reachable.

    Sends a health check request to verify the API is running.

    Returns:
        True if the API is reachable, False otherwise.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        return False


def send_chat_message(question: str, use_graph: bool) -> dict:
    """
    Send a chat message to the FastAPI backend.

    Posts the question to the /api/chat endpoint and returns
    the response containing the answer and metadata.

    Args:
        question: The customer's insurance question.
        use_graph: Whether to include GraphRAG context.

    Returns:
        Dictionary with 'answer', 'sources', and context flags.
        Returns error dict on failure.
    """
    try:
        # Send the question to the chat endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"question": question, "use_graph": use_graph},
            timeout=60,
        )

        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"Error: API returned status {response.status_code}", "sources": []}

    except requests.ConnectionError:
        return {"answer": "Error: Cannot connect to the API server. Please ensure it is running.", "sources": []}
    except requests.Timeout:
        return {"answer": "Error: Request timed out. The server may be processing a large query.", "sources": []}
    except Exception as error:
        return {"answer": f"Error: {str(error)}", "sources": []}


def trigger_ingestion(directory_path: str = None) -> dict:
    """
    Trigger document ingestion via the API.

    Sends a POST request to the /api/ingest endpoint to start
    document processing.

    Args:
        directory_path: Optional custom directory path for documents.

    Returns:
        Dictionary with ingestion results or error information.
    """
    try:
        payload = {"rebuild_graph": True}
        if directory_path:
            payload["directory_path"] = directory_path

        response = requests.post(
            f"{API_BASE_URL}/api/ingest",
            json=payload,
            timeout=300,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "detail": f"API returned status {response.status_code}"}

    except requests.ConnectionError:
        return {"status": "error", "detail": "Cannot connect to API server"}
    except requests.Timeout:
        return {"status": "error", "detail": "Ingestion timed out"}
    except Exception as error:
        return {"status": "error", "detail": str(error)}


def render_sidebar() -> None:
    """
    Render the sidebar with settings and controls.

    Displays API connection status, GraphRAG toggle, ingestion
    controls, and about information.
    """
    with st.sidebar:
        st.title("Settings")

        # API Connection Status
        st.subheader("API Status")
        api_connected = check_api_connection()
        if api_connected:
            st.success("Connected to API")
        else:
            st.error("API not connected")
            st.caption("Start the API with: `poetry run uvicorn src.main:app --reload`")

        st.divider()

        # GraphRAG Toggle
        st.subheader("Retrieval Mode")
        st.session_state.use_graph = st.toggle(
            "Enable GraphRAG",
            value=st.session_state.use_graph,
            help="Include knowledge graph context for relationship-aware answers",
        )

        if st.session_state.use_graph:
            st.caption("Using: Vector RAG + Knowledge Graph")
        else:
            st.caption("Using: Vector RAG only")

        st.divider()

        # Document Ingestion Controls
        st.subheader("Document Ingestion")
        if st.button("Ingest Documents", disabled=not api_connected):
            with st.spinner("Ingesting documents..."):
                result = trigger_ingestion()
                if result.get("status") == "completed":
                    st.success(
                        f"Ingested {result.get('documents_processed', 0)} documents, "
                        f"{result.get('chunks_created', 0)} chunks created"
                    )
                else:
                    st.error(f"Ingestion failed: {result.get('detail', 'Unknown error')}")

        st.divider()

        # Clear Chat History
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # About Section
        st.subheader("About")
        st.caption(
            "Insurance Policy Advisor uses RAG and GraphRAG "
            "to answer questions about insurance policy coverage, "
            "exclusions, and conditions."
        )


def render_chat_interface() -> None:
    """
    Render the main chat interface.

    Displays the chat history and handles new user input,
    sending questions to the API and displaying responses.
    """
    # Display page title
    st.title("Insurance Policy Advisor")
    st.caption("Ask questions about your insurance policy coverage")

    # Display existing chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"- {source}")

    # Handle new user input
    if prompt := st.chat_input("Ask about your insurance policy..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(
                    question=prompt,
                    use_graph=st.session_state.use_graph,
                )

            # Display the answer
            answer = response.get("answer", "Sorry, I couldn't generate a response.")
            st.markdown(answer)

            # Display sources if available
            sources = response.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.caption(f"- {source}")

            # Display context indicators
            rag_used = response.get("rag_context_used", False)
            graph_used = response.get("graph_context_used", False)
            context_info = []
            if rag_used:
                context_info.append("Vector RAG")
            if graph_used:
                context_info.append("GraphRAG")
            if context_info:
                st.caption(f"Context sources: {', '.join(context_info)}")

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


def main() -> None:
    """
    Main entry point for the Streamlit application.

    Configures the page layout and renders the sidebar and chat interface.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Insurance Policy Advisor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Render the sidebar with controls
    render_sidebar()

    # Render the main chat interface
    render_chat_interface()


# Run the application
if __name__ == "__main__":
    main()
