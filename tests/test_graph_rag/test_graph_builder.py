"""
Unit tests for the GraphBuilder module.

Tests graph construction from entities and relationships,
including node merging and neighbor retrieval.
"""

import pytest

from src.graph_rag.entity_extractor import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.graph_rag.graph_builder import GraphBuilder


class TestGraphBuilder:
    """Test suite for the GraphBuilder class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.builder = GraphBuilder()

    def test_build_from_extraction_adds_nodes(self) -> None:
        """Test that entities from an extraction result become graph nodes."""
        # Create test entities
        entities = [
            ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home Policy", description="Homeowners policy"),
            ExtractedEntity(id="coverage_1", entity_type="COVERAGE", name="Dwelling", description="Dwelling coverage"),
        ]
        relationships = []
        result = ExtractionResult(entities=entities, relationships=relationships)

        # Build the graph
        self.builder.build_from_extraction(result)

        # Verify nodes were added
        assert self.builder.node_count == 2

    def test_build_from_extraction_adds_edges(self) -> None:
        """Test that relationships become graph edges."""
        entities = [
            ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description="Home policy"),
            ExtractedEntity(id="exclusion_1", entity_type="EXCLUSION", name="Flood", description="Flood exclusion"),
        ]
        relationships = [
            ExtractedRelationship(source="policy_1", target="exclusion_1", relationship_type="EXCLUDES"),
        ]
        result = ExtractionResult(entities=entities, relationships=relationships)

        self.builder.build_from_extraction(result)

        # Verify edge was added
        assert self.builder.edge_count == 1

    def test_duplicate_nodes_are_merged(self) -> None:
        """Test that adding the same node twice merges rather than duplicates."""
        entity = ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description="First", source_chunk="chunk_0")
        result1 = ExtractionResult(entities=[entity], relationships=[])

        entity2 = ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description="Second", source_chunk="chunk_1")
        result2 = ExtractionResult(entities=[entity2], relationships=[])

        self.builder.build_from_extraction(result1)
        self.builder.build_from_extraction(result2)

        # Should still be one node, not two
        assert self.builder.node_count == 1
        # Source chunks should be merged
        node_data = self.builder.graph.nodes["policy_1"]
        assert "chunk_0" in node_data["source_chunks"]
        assert "chunk_1" in node_data["source_chunks"]

    def test_get_nodes_by_type_filters_correctly(self) -> None:
        """Test that get_nodes_by_type returns only matching nodes."""
        entities = [
            ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description=""),
            ExtractedEntity(id="coverage_1", entity_type="COVERAGE", name="Dwelling", description=""),
            ExtractedEntity(id="coverage_2", entity_type="COVERAGE", name="Liability", description=""),
            ExtractedEntity(id="exclusion_1", entity_type="EXCLUSION", name="Flood", description=""),
        ]
        result = ExtractionResult(entities=entities, relationships=[])
        self.builder.build_from_extraction(result)

        # Query for COVERAGE type only
        coverage_nodes = self.builder.get_nodes_by_type("COVERAGE")

        assert len(coverage_nodes) == 2
        assert all(node["entity_type"] == "COVERAGE" for node in coverage_nodes)

    def test_get_neighbors_returns_connected_nodes(self) -> None:
        """Test that get_neighbors returns nodes connected by edges."""
        entities = [
            ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description=""),
            ExtractedEntity(id="coverage_1", entity_type="COVERAGE", name="Dwelling", description=""),
            ExtractedEntity(id="exclusion_1", entity_type="EXCLUSION", name="Flood", description=""),
        ]
        relationships = [
            ExtractedRelationship(source="policy_1", target="coverage_1", relationship_type="COVERS"),
            ExtractedRelationship(source="policy_1", target="exclusion_1", relationship_type="EXCLUDES"),
        ]
        result = ExtractionResult(entities=entities, relationships=relationships)
        self.builder.build_from_extraction(result)

        # Get neighbors of policy_1
        neighbors = self.builder.get_neighbors("policy_1")

        assert len(neighbors) == 2

    def test_get_neighbors_with_type_filter(self) -> None:
        """Test that get_neighbors can filter by relationship type."""
        entities = [
            ExtractedEntity(id="policy_1", entity_type="POLICY", name="Home", description=""),
            ExtractedEntity(id="coverage_1", entity_type="COVERAGE", name="Dwelling", description=""),
            ExtractedEntity(id="exclusion_1", entity_type="EXCLUSION", name="Flood", description=""),
        ]
        relationships = [
            ExtractedRelationship(source="policy_1", target="coverage_1", relationship_type="COVERS"),
            ExtractedRelationship(source="policy_1", target="exclusion_1", relationship_type="EXCLUDES"),
        ]
        result = ExtractionResult(entities=entities, relationships=relationships)
        self.builder.build_from_extraction(result)

        # Get only COVERS relationships
        neighbors = self.builder.get_neighbors("policy_1", relationship_type="COVERS")

        assert len(neighbors) == 1
        assert neighbors[0]["id"] == "coverage_1"

    def test_empty_graph_has_zero_counts(self) -> None:
        """Test that a new graph has zero nodes and edges."""
        assert self.builder.node_count == 0
        assert self.builder.edge_count == 0
