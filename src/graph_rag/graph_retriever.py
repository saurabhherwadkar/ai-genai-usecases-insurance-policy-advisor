"""
Graph retriever module for the Insurance Policy Advisor GraphRAG system.

Queries the knowledge graph to find relevant entities and their
relationships based on a user's query. Provides graph-based context
that complements the vector-based RAG retrieval.
"""

import networkx as nx

from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.graph_store import GraphStore
from src.utils.logger import get_logger

# Module-level logger for graph retrieval operations
logger = get_logger(__name__)


class GraphRetrievalResult:
    """
    Represents a graph-based retrieval result.

    Contains the relevant entities, their relationships, and
    a formatted context string for LLM prompt injection.

    Attributes:
        entities: List of relevant entity dictionaries.
        relationships: List of relevant relationship descriptions.
        context_string: Formatted string representation of the graph context.
    """

    def __init__(self, entities: list[dict], relationships: list[str], context_string: str) -> None:
        """
        Initialize a GraphRetrievalResult.

        Args:
            entities: List of entity dictionaries found in the graph.
            relationships: List of relationship description strings.
            context_string: Pre-formatted context for LLM injection.
        """
        self.entities = entities
        self.relationships = relationships
        self.context_string = context_string


class GraphRetriever:
    """
    Retrieves relevant context from the knowledge graph.

    Searches the graph for entities matching query keywords and
    traverses their relationships to build comprehensive context
    about insurance policies, coverages, and exclusions.
    """

    def __init__(self, graph_builder: GraphBuilder = None, graph_store: GraphStore = None) -> None:
        """
        Initialize the GraphRetriever with graph components.

        Loads the persisted graph if available, or uses the provided
        graph builder's current state.

        Args:
            graph_builder: Optional GraphBuilder with an existing graph.
            graph_store: Optional GraphStore for loading persisted graphs.
        """
        # Initialize graph store for persistence
        self._graph_store = graph_store or GraphStore()

        # Initialize graph builder and load persisted graph if available
        self._graph_builder = graph_builder or GraphBuilder()

        # Load the graph from persistence if it exists
        if self._graph_store.exists():
            loaded_graph = self._graph_store.load()
            self._graph_builder.graph = loaded_graph
            logger.info(f"Loaded graph with {self._graph_builder.node_count} nodes for retrieval")
        else:
            logger.info("No persisted graph found, starting with empty graph")

    def retrieve(self, query: str, max_depth: int = 2) -> GraphRetrievalResult:
        """
        Retrieve graph context relevant to the query.

        Finds entities matching query keywords and traverses their
        neighborhood to build relationship context.

        Args:
            query: The user's question to find graph context for.
            max_depth: Maximum traversal depth from matching nodes.

        Returns:
            A GraphRetrievalResult with entities, relationships, and formatted context.
        """
        # Get the underlying graph
        graph = self._graph_builder.graph

        # Return empty result if graph is empty
        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, no context available")
            return GraphRetrievalResult(entities=[], relationships=[], context_string="")

        # Find nodes matching the query keywords
        matching_nodes = self._find_matching_nodes(query, graph)

        # Return empty result if no matches found
        if not matching_nodes:
            logger.info(f"No matching nodes found for query: '{query[:80]}'")
            return GraphRetrievalResult(entities=[], relationships=[], context_string="")

        # Traverse the graph from matching nodes to collect context
        entities, relationships = self._traverse_neighborhood(matching_nodes, graph, max_depth)

        # Format the context string for LLM injection
        context_string = self._format_context(entities, relationships)

        logger.info(
            f"Graph retrieval: {len(matching_nodes)} matching nodes, "
            f"{len(entities)} entities, {len(relationships)} relationships"
        )

        return GraphRetrievalResult(
            entities=entities,
            relationships=relationships,
            context_string=context_string,
        )

    def _find_matching_nodes(self, query: str, graph: nx.DiGraph) -> list[str]:
        """
        Find graph nodes whose names or descriptions match query keywords.

        Performs case-insensitive keyword matching against node names
        and descriptions.

        Args:
            query: The search query string.
            graph: The NetworkX graph to search.

        Returns:
            List of matching node IDs.
        """
        # Tokenize the query into keywords (lowercase)
        query_keywords = [word.lower() for word in query.split() if len(word) > 2]

        # Search through all nodes for keyword matches
        matching_nodes = []
        for node_id, data in graph.nodes(data=True):
            node_name = data.get("name", "").lower()
            node_description = data.get("description", "").lower()
            node_type = data.get("entity_type", "").lower()

            # Check if any query keyword matches the node's text
            for keyword in query_keywords:
                if keyword in node_name or keyword in node_description or keyword in node_type:
                    matching_nodes.append(node_id)
                    break

        logger.debug(f"Found {len(matching_nodes)} nodes matching query keywords")
        return matching_nodes

    def _traverse_neighborhood(
        self, start_nodes: list[str], graph: nx.DiGraph, max_depth: int
    ) -> tuple[list[dict], list[str]]:
        """
        Traverse the graph neighborhood around starting nodes.

        Collects entities and relationship descriptions up to the
        specified depth from each starting node.

        Args:
            start_nodes: List of node IDs to start traversal from.
            graph: The NetworkX graph to traverse.
            max_depth: Maximum traversal depth.

        Returns:
            Tuple of (entities list, relationships list).
        """
        visited_nodes = set()
        entities = []
        relationships = []

        # Traverse from each starting node using BFS
        for start_node in start_nodes:
            # Use BFS to traverse up to max_depth
            bfs_edges = nx.bfs_edges(graph, start_node, depth_limit=max_depth)

            # Add the starting node itself
            if start_node not in visited_nodes:
                visited_nodes.add(start_node)
                node_data = graph.nodes[start_node]
                entities.append({"id": start_node, **node_data})

            # Process each edge found during traversal
            for source, target in bfs_edges:
                # Add target node if not visited
                if target not in visited_nodes:
                    visited_nodes.add(target)
                    node_data = graph.nodes[target]
                    entities.append({"id": target, **node_data})

                # Add relationship description
                edge_data = graph.edges[source, target]
                rel_type = edge_data.get("relationship_type", "RELATED_TO")
                source_name = graph.nodes[source].get("name", source)
                target_name = graph.nodes[target].get("name", target)
                relationships.append(f"{source_name} --[{rel_type}]--> {target_name}")

        return entities, relationships

    def _format_context(self, entities: list[dict], relationships: list[str]) -> str:
        """
        Format entities and relationships into a context string for LLM.

        Creates a structured text representation of the graph context
        suitable for injection into a prompt.

        Args:
            entities: List of entity dictionaries.
            relationships: List of relationship description strings.

        Returns:
            Formatted context string.
        """
        # Build the entities section
        entity_lines = []
        for entity in entities:
            entity_type = entity.get("entity_type", "UNKNOWN")
            name = entity.get("name", "Unknown")
            description = entity.get("description", "")
            entity_lines.append(f"- [{entity_type}] {name}: {description}")

        # Build the relationships section
        relationship_lines = [f"- {rel}" for rel in relationships]

        # Combine into formatted context
        context_parts = []
        if entity_lines:
            context_parts.append("ENTITIES:\n" + "\n".join(entity_lines))
        if relationship_lines:
            context_parts.append("RELATIONSHIPS:\n" + "\n".join(relationship_lines))

        return "\n\n".join(context_parts)

    def reload_graph(self) -> None:
        """
        Reload the graph from persistence.

        Used after new documents are ingested and the graph is updated.
        """
        if self._graph_store.exists():
            loaded_graph = self._graph_store.load()
            self._graph_builder.graph = loaded_graph
            logger.info(f"Graph reloaded: {self._graph_builder.node_count} nodes")
        else:
            logger.warning("No persisted graph to reload")
