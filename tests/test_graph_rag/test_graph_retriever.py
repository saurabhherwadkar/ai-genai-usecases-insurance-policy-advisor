"""
Unit tests for the GraphRetriever module.

Tests graph-based retrieval with keyword matching and neighborhood traversal.
"""

import pytest
from unittest.mock import patch, MagicMock

import networkx as nx

from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.graph_retriever import GraphRetriever, GraphRetrievalResult
from src.graph_rag.entity_extractor import ExtractedEntity, ExtractedRelationship, ExtractionResult


class TestGraphRetriever:
    """Test suite for the GraphRetriever class."""

    def _build_test_graph(self) -> GraphBuilder:
        """
        Build a test graph with sample insurance entities.

        Returns:
            A GraphBuilder with pre-populated nodes and edges.
        """
        builder = GraphBuilder()
        entities = [
            ExtractedEntity(id="policy_home", entity_type="POLICY", name="Homeowners Insurance", description="Standard homeowners coverage"),
            ExtractedEntity(id="coverage_dwelling", entity_type="COVERAGE", name="Dwelling Coverage", description="Covers physical structure damage"),
            ExtractedEntity(id="exclusion_flood", entity_type="EXCLUSION", name="Flood Damage", description="Flood and water damage excluded"),
            ExtractedEntity(id="limit_350k", entity_type="LIMIT", name="$350,000 Limit", description="Maximum dwelling coverage"),
        ]
        relationships = [
            ExtractedRelationship(source="policy_home", target="coverage_dwelling", relationship_type="COVERS"),
            ExtractedRelationship(source="policy_home", target="exclusion_flood", relationship_type="EXCLUDES"),
            ExtractedRelationship(source="coverage_dwelling", target="limit_350k", relationship_type="HAS_LIMIT"),
        ]
        result = ExtractionResult(entities=entities, relationships=relationships)
        builder.build_from_extraction(result)
        return builder

    @patch("src.graph_rag.graph_retriever.GraphStore")
    def test_retrieve_finds_matching_entities(self, mock_store_class) -> None:
        """Test that retrieve finds entities matching query keywords."""
        # Set up mock graph store (no persisted graph)
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        # Build test graph
        builder = self._build_test_graph()

        retriever = GraphRetriever(graph_builder=builder, graph_store=mock_store)
        result = retriever.retrieve("flood damage")

        # Should find the flood exclusion entity
        assert isinstance(result, GraphRetrievalResult)
        assert len(result.entities) > 0
        assert any("flood" in e.get("name", "").lower() for e in result.entities)

    @patch("src.graph_rag.graph_retriever.GraphStore")
    def test_retrieve_returns_relationships(self, mock_store_class) -> None:
        """Test that retrieve returns relationship descriptions."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        builder = self._build_test_graph()
        retriever = GraphRetriever(graph_builder=builder, graph_store=mock_store)
        result = retriever.retrieve("dwelling coverage")

        # Should include relationships connected to dwelling coverage
        assert len(result.relationships) > 0

    @patch("src.graph_rag.graph_retriever.GraphStore")
    def test_retrieve_formats_context_string(self, mock_store_class) -> None:
        """Test that retrieve produces a non-empty formatted context string."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        builder = self._build_test_graph()
        retriever = GraphRetriever(graph_builder=builder, graph_store=mock_store)
        result = retriever.retrieve("homeowners insurance")

        # Should have a formatted context string
        assert result.context_string != ""
        assert "ENTITIES:" in result.context_string

    @patch("src.graph_rag.graph_retriever.GraphStore")
    def test_retrieve_no_matches_returns_empty(self, mock_store_class) -> None:
        """Test that a query with no matches returns empty result."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        builder = self._build_test_graph()
        retriever = GraphRetriever(graph_builder=builder, graph_store=mock_store)
        result = retriever.retrieve("xyz completely unrelated query")

        # No matches should return empty
        assert result.entities == []
        assert result.context_string == ""

    @patch("src.graph_rag.graph_retriever.GraphStore")
    def test_retrieve_empty_graph_returns_empty(self, mock_store_class) -> None:
        """Test that retrieval on an empty graph returns empty result."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_store_class.return_value = mock_store

        # Use empty graph builder
        empty_builder = GraphBuilder()
        retriever = GraphRetriever(graph_builder=empty_builder, graph_store=mock_store)
        result = retriever.retrieve("any query")

        assert result.entities == []
        assert result.context_string == ""
