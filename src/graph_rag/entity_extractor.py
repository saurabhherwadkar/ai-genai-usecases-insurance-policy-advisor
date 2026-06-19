"""
Entity extractor module for the Insurance Policy Advisor GraphRAG system.

Uses Claude to extract structured entities and relationships from
insurance policy document chunks. Entities include policy types,
coverages, exclusions, conditions, and monetary limits.
"""

import json
from typing import Any

import anthropic

from src.config.settings import get_settings
from src.utils.logger import get_logger

# Module-level logger for entity extraction operations
logger = get_logger(__name__)

# Prompt template for entity extraction using Claude
EXTRACTION_PROMPT = """You are an insurance policy document analyzer. Extract structured entities and relationships from the following insurance policy text.

Extract the following entity types:
- POLICY: The type of insurance policy (e.g., "Homeowners", "Auto", "Health")
- COVERAGE: Specific coverages provided (e.g., "Dwelling Coverage", "Liability Coverage")
- EXCLUSION: Things explicitly not covered (e.g., "Flood damage", "Intentional loss")
- CONDITION: Requirements or conditions for coverage (e.g., "Deductible applies", "Prior authorization required")
- LIMIT: Monetary limits or caps (e.g., "$350,000 dwelling limit", "$100,000 liability limit")
- BENEFIT: Specific benefits or services covered (e.g., "Preventive care", "Rental reimbursement")

Extract relationships between entities:
- COVERS: Policy/Coverage covers a benefit or situation
- EXCLUDES: Policy/Coverage excludes something
- REQUIRES: Coverage requires a condition
- HAS_LIMIT: Coverage has a monetary limit
- PART_OF: Coverage is part of a policy

Return the result as a JSON object with this structure:
{
    "entities": [
        {"id": "unique_id", "type": "ENTITY_TYPE", "name": "entity name", "description": "brief description"}
    ],
    "relationships": [
        {"source": "entity_id", "target": "entity_id", "type": "RELATIONSHIP_TYPE", "description": "brief description"}
    ]
}

Insurance policy text to analyze:
---
{text}
---

Return ONLY the JSON object, no other text."""


class ExtractedEntity:
    """
    Represents an entity extracted from an insurance document.

    Attributes:
        id: Unique identifier for the entity.
        entity_type: The category of entity (POLICY, COVERAGE, EXCLUSION, etc.).
        name: Human-readable name of the entity.
        description: Brief description of what this entity represents.
        source_chunk: Reference to the source chunk this was extracted from.
    """

    def __init__(self, id: str, entity_type: str, name: str, description: str, source_chunk: str = "") -> None:
        """
        Initialize an ExtractedEntity.

        Args:
            id: Unique identifier for the entity.
            entity_type: Category of entity (POLICY, COVERAGE, etc.).
            name: Human-readable name.
            description: Brief description.
            source_chunk: Reference to the source text chunk.
        """
        self.id = id
        self.entity_type = entity_type
        self.name = name
        self.description = description
        self.source_chunk = source_chunk

    def to_dict(self) -> dict[str, str]:
        """
        Convert the entity to a dictionary representation.

        Returns:
            Dictionary with all entity attributes.
        """
        return {
            "id": self.id,
            "type": self.entity_type,
            "name": self.name,
            "description": self.description,
            "source_chunk": self.source_chunk,
        }


class ExtractedRelationship:
    """
    Represents a relationship between two extracted entities.

    Attributes:
        source: ID of the source entity.
        target: ID of the target entity.
        relationship_type: Type of relationship (COVERS, EXCLUDES, etc.).
        description: Brief description of the relationship.
    """

    def __init__(self, source: str, target: str, relationship_type: str, description: str = "") -> None:
        """
        Initialize an ExtractedRelationship.

        Args:
            source: ID of the source entity.
            target: ID of the target entity.
            relationship_type: Type of relationship.
            description: Brief description of the relationship.
        """
        self.source = source
        self.target = target
        self.relationship_type = relationship_type
        self.description = description

    def to_dict(self) -> dict[str, str]:
        """
        Convert the relationship to a dictionary representation.

        Returns:
            Dictionary with all relationship attributes.
        """
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relationship_type,
            "description": self.description,
        }


class ExtractionResult:
    """
    Contains the complete extraction results from a document chunk.

    Attributes:
        entities: List of extracted entities.
        relationships: List of extracted relationships between entities.
    """

    def __init__(self, entities: list[ExtractedEntity], relationships: list[ExtractedRelationship]) -> None:
        """
        Initialize an ExtractionResult.

        Args:
            entities: List of extracted entities.
            relationships: List of extracted relationships.
        """
        self.entities = entities
        self.relationships = relationships


class EntityExtractor:
    """
    Extracts structured entities and relationships from insurance documents using Claude.

    Sends document chunks to Claude with a structured prompt and parses
    the returned JSON into entity and relationship objects for graph construction.
    """

    def __init__(self) -> None:
        """Initialize the EntityExtractor with the Anthropic client."""
        # Load settings for model configuration
        settings = get_settings()
        self._model = settings.llm.model
        self._max_tokens = settings.llm.max_tokens

        # Initialize the Anthropic client (uses ANTHROPIC_API_KEY env var)
        self._client = anthropic.Anthropic()

        logger.info(f"EntityExtractor initialized with model: {self._model}")

    def extract(self, text: str, source_reference: str = "") -> ExtractionResult:
        """
        Extract entities and relationships from a text chunk.

        Sends the text to Claude for structured extraction and parses
        the response into entity and relationship objects.

        Args:
            text: The insurance policy text chunk to analyze.
            source_reference: Reference identifier for the source chunk.

        Returns:
            An ExtractionResult containing entities and relationships.
        """
        # Skip extraction for empty text
        if not text or not text.strip():
            logger.warning("Received empty text for entity extraction")
            return ExtractionResult(entities=[], relationships=[])

        try:
            # Format the extraction prompt with the text chunk
            prompt = EXTRACTION_PROMPT.format(text=text)

            # Call Claude for entity extraction
            logger.debug(f"Sending text chunk ({len(text)} chars) to Claude for extraction")
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the response text
            response_text = response.content[0].text

            # Parse the JSON response
            extraction_data = self._parse_response(response_text)

            # Convert to typed objects
            entities = self._build_entities(extraction_data.get("entities", []), source_reference)
            relationships = self._build_relationships(extraction_data.get("relationships", []))

            logger.info(
                f"Extracted {len(entities)} entities and {len(relationships)} relationships "
                f"from chunk: {source_reference}"
            )
            return ExtractionResult(entities=entities, relationships=relationships)

        except anthropic.APIError as api_error:
            logger.error(f"Anthropic API error during extraction: {str(api_error)}")
            return ExtractionResult(entities=[], relationships=[])
        except Exception as error:
            logger.error(f"Unexpected error during entity extraction: {str(error)}")
            return ExtractionResult(entities=[], relationships=[])

    def extract_batch(self, texts: list[str], source_references: list[str] = None) -> list[ExtractionResult]:
        """
        Extract entities from multiple text chunks.

        Processes each chunk sequentially to stay within API rate limits.

        Args:
            texts: List of text chunks to analyze.
            source_references: Optional list of source references matching each text.

        Returns:
            List of ExtractionResult objects, one per input text.
        """
        # Create default source references if not provided
        references = source_references or [f"chunk_{i}" for i in range(len(texts))]

        # Process each chunk and collect results
        results = []
        for i, (text, reference) in enumerate(zip(texts, references)):
            logger.debug(f"Extracting entities from chunk {i + 1}/{len(texts)}")
            result = self.extract(text=text, source_reference=reference)
            results.append(result)

        logger.info(f"Batch extraction complete: processed {len(texts)} chunks")
        return results

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """
        Parse the JSON response from Claude.

        Handles potential formatting issues like markdown code blocks
        around the JSON output.

        Args:
            response_text: Raw text response from Claude.

        Returns:
            Parsed dictionary from the JSON response.
        """
        # Strip markdown code block markers if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        # Parse the JSON
        try:
            return json.loads(cleaned_text.strip())
        except json.JSONDecodeError as parse_error:
            logger.error(f"Failed to parse extraction response as JSON: {str(parse_error)}")
            return {"entities": [], "relationships": []}

    def _build_entities(self, raw_entities: list[dict], source_reference: str) -> list[ExtractedEntity]:
        """
        Convert raw entity dictionaries to ExtractedEntity objects.

        Args:
            raw_entities: List of entity dictionaries from Claude's response.
            source_reference: Source chunk reference to attach to each entity.

        Returns:
            List of ExtractedEntity objects.
        """
        entities = []
        for raw in raw_entities:
            entity = ExtractedEntity(
                id=raw.get("id", ""),
                entity_type=raw.get("type", "UNKNOWN"),
                name=raw.get("name", ""),
                description=raw.get("description", ""),
                source_chunk=source_reference,
            )
            entities.append(entity)
        return entities

    def _build_relationships(self, raw_relationships: list[dict]) -> list[ExtractedRelationship]:
        """
        Convert raw relationship dictionaries to ExtractedRelationship objects.

        Args:
            raw_relationships: List of relationship dictionaries from Claude's response.

        Returns:
            List of ExtractedRelationship objects.
        """
        relationships = []
        for raw in raw_relationships:
            relationship = ExtractedRelationship(
                source=raw.get("source", ""),
                target=raw.get("target", ""),
                relationship_type=raw.get("type", "RELATED_TO"),
                description=raw.get("description", ""),
            )
            relationships.append(relationship)
        return relationships
