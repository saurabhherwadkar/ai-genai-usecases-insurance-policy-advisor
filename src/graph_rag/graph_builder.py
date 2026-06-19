"""
Graph builder module for the Insurance Policy Advisor GraphRAG system.

Builds a NetworkX knowledge graph from extracted entities and
relationships. The graph represents the insurance domain knowledge
with nodes for policies, coverages, exclusions, and edges for
their relationships.
"""

import networkx as nx

from src.graph_rag.entity_extractor import ExtractionResult, ExtractedEntity, ExtractedRelationship
from src.utils.logger import get_logger

# Module-level logger for graph building operations
logger = get_logger(__name__)


class GraphBuilder:
    """
    Builds and maintains a NetworkX knowledge graph from extracted entities.

    Creates nodes from entities and edges from relationships, building
    up a comprehensive graph of insurance policy knowledge that can
    be queried for context-aware retrieval.
    """

    def __init__(self) -> None:
        """Initialize the GraphBuilder with an empty directed graph."""
        # Create a directed graph to represent entity relationships
        self._graph = nx.DiGraph()

        logger.info("GraphBuilder initialized with empty graph")

    def build_from_extraction(self, extraction_result: ExtractionResult) -> None:
        """
        Add entities and relationships from an extraction result to the graph.

        Processes a single extraction result by adding all its entities
        as nodes and all its relationships as edges.

        Args:
            extraction_result: The ExtractionResult containing entities and relationships to add.
        """
        # Add all entities as nodes
        for entity in extraction_result.entities:
            self._add_entity_node(entity)

        # Add all relationships as edges
        for relationship in extraction_result.relationships:
            self._add_relationship_edge(relationship)

        logger.debug(
            f"Graph updated: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def build_from_extractions(self, extraction_results: list[ExtractionResult]) -> None:
        """
        Build the graph from multiple extraction results.

        Processes all extraction results sequentially, adding entities
        and relationships from each to the shared graph.

        Args:
            extraction_results: List of ExtractionResult objects to process.
        """
        logger.info(f"Building graph from {len(extraction_results)} extraction results")

        # Process each extraction result
        for result in extraction_results:
            self.build_from_extraction(result)

        logger.info(
            f"Graph build complete: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

    def _add_entity_node(self, entity: ExtractedEntity) -> None:
        """
        Add an entity as a node in the graph.

        If the node already exists, merges the description and source
        information rather than overwriting.

        Args:
            entity: The ExtractedEntity to add as a node.
        """
        node_id = entity.id

        # Check if node already exists and merge if so
        if self._graph.has_node(node_id):
            existing_data = self._graph.nodes[node_id]
            # Append source chunk reference if new
            existing_sources = existing_data.get("source_chunks", [])
            if entity.source_chunk and entity.source_chunk not in existing_sources:
                existing_sources.append(entity.source_chunk)
                self._graph.nodes[node_id]["source_chunks"] = existing_sources
            logger.debug(f"Merged existing node: {node_id}")
        else:
            # Add new node with all entity attributes
            self._graph.add_node(
                node_id,
                entity_type=entity.entity_type,
                name=entity.name,
                description=entity.description,
                source_chunks=[entity.source_chunk] if entity.source_chunk else [],
            )
            logger.debug(f"Added new node: {node_id} (type={entity.entity_type}, name={entity.name})")

    def _add_relationship_edge(self, relationship: ExtractedRelationship) -> None:
        """
        Add a relationship as a directed edge in the graph.

        Creates the edge between source and target nodes. If either
        node does not exist, creates a placeholder node.

        Args:
            relationship: The ExtractedRelationship to add as an edge.
        """
        source = relationship.source
        target = relationship.target

        # Create placeholder nodes if they don't exist
        if not self._graph.has_node(source):
            self._graph.add_node(source, entity_type="UNKNOWN", name=source, description="", source_chunks=[])
            logger.debug(f"Created placeholder node for source: {source}")

        if not self._graph.has_node(target):
            self._graph.add_node(target, entity_type="UNKNOWN", name=target, description="", source_chunks=[])
            logger.debug(f"Created placeholder node for target: {target}")

        # Add the directed edge with relationship attributes
        self._graph.add_edge(
            source,
            target,
            relationship_type=relationship.relationship_type,
            description=relationship.description,
        )

        logger.debug(f"Added edge: {source} --[{relationship.relationship_type}]--> {target}")

    @property
    def graph(self) -> nx.DiGraph:
        """
        Get the underlying NetworkX graph.

        Returns:
            The directed graph containing all entities and relationships.
        """
        return self._graph

    @graph.setter
    def graph(self, new_graph: nx.DiGraph) -> None:
        """
        Set the underlying graph (used when loading from persistence).

        Args:
            new_graph: A NetworkX DiGraph to use as the graph.
        """
        self._graph = new_graph

    @property
    def node_count(self) -> int:
        """
        Get the number of nodes in the graph.

        Returns:
            Total number of entity nodes.
        """
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """
        Get the number of edges in the graph.

        Returns:
            Total number of relationship edges.
        """
        return self._graph.number_of_edges()

    def get_nodes_by_type(self, entity_type: str) -> list[dict]:
        """
        Get all nodes of a specific entity type.

        Args:
            entity_type: The entity type to filter by (e.g., "COVERAGE", "EXCLUSION").

        Returns:
            List of node data dictionaries matching the type.
        """
        # Filter nodes by entity type attribute
        matching_nodes = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("entity_type") == entity_type:
                matching_nodes.append({"id": node_id, **data})

        logger.debug(f"Found {len(matching_nodes)} nodes of type '{entity_type}'")
        return matching_nodes

    def get_neighbors(self, node_id: str, relationship_type: str = None) -> list[dict]:
        """
        Get neighboring nodes connected to a given node.

        Optionally filters by relationship type on the connecting edge.

        Args:
            node_id: The ID of the node to find neighbors for.
            relationship_type: Optional filter for edge relationship type.

        Returns:
            List of neighbor node data dictionaries.
        """
        # Return empty if node doesn't exist
        if not self._graph.has_node(node_id):
            logger.debug(f"Node not found for neighbor lookup: {node_id}")
            return []

        # Get all successors (outgoing edges) and predecessors (incoming edges)
        neighbors = []

        # Check outgoing edges
        for successor in self._graph.successors(node_id):
            edge_data = self._graph.edges[node_id, successor]
            if relationship_type is None or edge_data.get("relationship_type") == relationship_type:
                node_data = self._graph.nodes[successor]
                neighbors.append({
                    "id": successor,
                    "direction": "outgoing",
                    "relationship": edge_data.get("relationship_type", ""),
                    **node_data,
                })

        # Check incoming edges
        for predecessor in self._graph.predecessors(node_id):
            edge_data = self._graph.edges[predecessor, node_id]
            if relationship_type is None or edge_data.get("relationship_type") == relationship_type:
                node_data = self._graph.nodes[predecessor]
                neighbors.append({
                    "id": predecessor,
                    "direction": "incoming",
                    "relationship": edge_data.get("relationship_type", ""),
                    **node_data,
                })

        logger.debug(f"Found {len(neighbors)} neighbors for node '{node_id}'")
        return neighbors
