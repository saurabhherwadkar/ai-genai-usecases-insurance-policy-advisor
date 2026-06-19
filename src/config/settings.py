"""
Settings module for the Insurance Policy Advisor application.

Loads configuration from config/settings.yaml and allows overriding
via environment variables. Provides a singleton settings instance
accessible throughout the application.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
load_dotenv()


class AppSettings(BaseModel):
    """Application-level settings such as name, version, host, and port."""

    name: str = "Insurance Policy Advisor"
    version: str = "0.1.0"
    env: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")


class LLMSettings(BaseModel):
    """Settings for the Anthropic Claude LLM client."""

    model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.3)


class EmbeddingsSettings(BaseModel):
    """Settings for the sentence-transformers embedding model."""

    model: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384)


class VectorStoreSettings(BaseModel):
    """Settings for the ChromaDB vector store."""

    persist_directory: str = Field(default="./chroma_data")
    collection_name: str = Field(default="insurance_policies")


class GraphStoreSettings(BaseModel):
    """Settings for the NetworkX graph persistence."""

    persist_path: str = Field(default="./graph_data/insurance_graph.json")


class RAGSettings(BaseModel):
    """Settings for RAG chunking and retrieval parameters."""

    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=5)


class IngestionSettings(BaseModel):
    """Settings for the document ingestion pipeline."""

    supported_formats: list[str] = Field(default=[".pdf", ".docx", ".txt"])
    documents_directory: str = Field(default="./data/sample_policies")


class UISettings(BaseModel):
    """Settings for the Streamlit frontend."""

    port: int = Field(default=8501)
    api_base_url: str = Field(default="http://localhost:8000")


class Settings(BaseModel):
    """
    Root settings model that aggregates all configuration sections.

    Loads values from config/settings.yaml and overrides with
    environment variables where applicable.
    """

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph_store: GraphStoreSettings = Field(default_factory=GraphStoreSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    ui: UISettings = Field(default_factory=UISettings)


def _load_yaml_config() -> dict[str, Any]:
    """
    Load configuration from the settings.yaml file.

    Returns:
        Dictionary containing the parsed YAML configuration.
    """
    # Determine the path to the config file relative to the project root
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    # Return empty dict if config file does not exist
    if not config_path.exists():
        return {}

    # Read and parse the YAML file
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to the configuration dictionary.

    Environment variables take precedence over YAML values.
    Mapping: ENV_VAR_NAME -> config section.key

    Args:
        config: The base configuration dictionary from YAML.

    Returns:
        Updated configuration dictionary with environment overrides applied.
    """
    # Map environment variables to their config paths
    env_mappings = {
        "APP_ENV": ("app", "env"),
        "APP_LOG_LEVEL": ("app", "log_level"),
        "APP_HOST": ("app", "host"),
        "APP_PORT": ("app", "port"),
        "LLM_MODEL": ("llm", "model"),
        "CHROMA_PERSIST_DIRECTORY": ("vector_store", "persist_directory"),
        "CHROMA_COLLECTION_NAME": ("vector_store", "collection_name"),
        "GRAPH_PERSIST_PATH": ("graph_store", "persist_path"),
        "RAG_CHUNK_SIZE": ("rag", "chunk_size"),
        "RAG_CHUNK_OVERLAP": ("rag", "chunk_overlap"),
        "RAG_TOP_K": ("rag", "top_k"),
        "EMBEDDING_MODEL": ("embeddings", "model"),
    }

    # Apply each environment variable override if set
    for env_var, (section, key) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Ensure the section exists in the config
            if section not in config:
                config[section] = {}
            # Convert numeric strings to appropriate types
            if key in ("port", "chunk_size", "chunk_overlap", "top_k"):
                config[section][key] = int(value)
            else:
                config[section][key] = value

    return config


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Loads from YAML config, applies environment variable overrides,
    and returns a validated Settings instance. Results are cached
    so subsequent calls return the same instance.

    Returns:
        The application Settings instance.
    """
    # Load base configuration from YAML
    config = _load_yaml_config()

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Create and return the validated settings object
    return Settings(**config)
