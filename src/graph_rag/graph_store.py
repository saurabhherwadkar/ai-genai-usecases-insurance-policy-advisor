"""
Graph store module for the Insurance Policy Advisor GraphRAG system.

Handles persistence of the NetworkX knowledge graph to and from
JSON files. Provides serialization and deserialization methods
for saving and loading the graph state.
"""

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for graph persistence operations
logger = get_logger(__name__)


class GraphStore:
    """
    Manages persistence of the NetworkX knowledge graph.

    Serializes the graph to JSON format and stores it on disk.
    Supports saving, loading, and checking for existing graph data.
    """

    def __init__(self, persist_path: str = None) -> None:
        """
        Initialize the GraphStore with the persistence file path.

        Args:
            persist_path: Path where the graph JSON will be stored.
                         Defaults to the configured graph persist path.
        """
        # Load the persist path from settings if not provided
        settings = get_settings()
        self._persist_path = Path(persist_path or settings.graph_store.persist_path)

        logger.info(f"GraphStore initialized with persist path: {self._persist_path}")

    def save(self, graph: nx.DiGraph) -> None:
        """
        Save the knowledge graph to a JSON file.

        Serializes the NetworkX graph using node-link format and
        writes it to the configured persistence path.

        Args:
            graph: The NetworkX DiGraph to persist.
        """
        try:
            # Ensure the parent directory exists
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize the graph to node-link data format
            graph_data = json_graph.node_link_data(graph)

            # Write the serialized graph to JSON file
            with open(self._persist_path, "w", encoding="utf-8") as graph_file:
                json.dump(graph_data, graph_file, indent=2, ensure_ascii=False)

            logger.info(
                f"Graph saved successfully: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges -> {self._persist_path}"
            )

        except Exception as error:
            logger.error(f"Failed to save graph to {self._persist_path}: {str(error)}")
            raise

    def load(self) -> nx.DiGraph:
        """
        Load the knowledge graph from the JSON file.

        Reads the persisted JSON and deserializes it back into
        a NetworkX DiGraph.

        Returns:
            The loaded NetworkX DiGraph, or an empty graph if no file exists.
        """
        # Return empty graph if no persisted file exists
        if not self._persist_path.exists():
            logger.info("No persisted graph found, returning empty graph")
            return nx.DiGraph()

        try:
            # Read the JSON file
            with open(self._persist_path, "r", encoding="utf-8") as graph_file:
                graph_data = json.load(graph_file)

            # Deserialize from node-link format back to DiGraph
            graph = json_graph.node_link_graph(graph_data, directed=True)

            logger.info(
                f"Graph loaded successfully: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges <- {self._persist_path}"
            )
            return graph

        except json.JSONDecodeError as parse_error:
            logger.error(f"Failed to parse graph JSON file: {str(parse_error)}")
            return nx.DiGraph()
        except Exception as error:
            logger.error(f"Failed to load graph from {self._persist_path}: {str(error)}")
            return nx.DiGraph()

    def exists(self) -> bool:
        """
        Check if a persisted graph file exists.

        Returns:
            True if the graph file exists on disk, False otherwise.
        """
        return self._persist_path.exists()

    def delete(self) -> None:
        """
        Delete the persisted graph file.

        Used for resetting the graph during re-ingestion or testing.
        """
        if self._persist_path.exists():
            self._persist_path.unlink()
            logger.info(f"Deleted persisted graph: {self._persist_path}")
        else:
            logger.debug("No persisted graph to delete")
