"""
Shared test fixtures for the Insurance Policy Advisor test suite.

Provides reusable fixtures for sample documents, text chunks,
mock services, and test configuration used across all test modules.
"""

import pytest

from src.ingestion.document_loader import Document
from src.ingestion.text_splitter import TextChunk


@pytest.fixture
def sample_document() -> Document:
    """
    Create a sample Document for testing.

    Returns:
        A Document instance with realistic insurance policy content.
    """
    return Document(
        content=(
            "HOMEOWNERS INSURANCE POLICY\n\n"
            "Coverage A - Dwelling\n"
            "This policy covers the dwelling on the residence premises. "
            "The dwelling coverage limit is $350,000.\n\n"
            "Coverage includes physical damage from covered perils, "
            "built-in appliances, and permanently installed fixtures.\n\n"
            "EXCLUSIONS\n"
            "This policy does NOT cover flood damage, earthquake, "
            "or intentional loss caused by the insured."
        ),
        metadata={
            "filename": "test_policy.txt",
            "format": ".txt",
            "path": "/test/test_policy.txt",
            "size_bytes": 500,
        },
    )


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    """
    Create sample TextChunks for testing.

    Returns:
        A list of TextChunk instances representing split document content.
    """
    return [
        TextChunk(
            content=(
                "Coverage A - Dwelling. This policy covers the dwelling "
                "on the residence premises. The dwelling coverage limit is $350,000."
            ),
            metadata={"filename": "home_policy.txt", "chunk_index": 0, "char_count": 120},
            chunk_index=0,
        ),
        TextChunk(
            content=(
                "EXCLUSIONS. This policy does NOT cover flood damage, "
                "earthquake, or intentional loss caused by the insured."
            ),
            metadata={"filename": "home_policy.txt", "chunk_index": 1, "char_count": 110},
            chunk_index=1,
        ),
        TextChunk(
            content=(
                "Personal Liability coverage up to $100,000 per occurrence "
                "for bodily injury to others on the premises."
            ),
            metadata={"filename": "home_policy.txt", "chunk_index": 2, "char_count": 100},
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_extraction_response() -> dict:
    """
    Create a sample entity extraction response for testing.

    Returns:
        A dictionary simulating Claude's extraction response.
    """
    return {
        "entities": [
            {
                "id": "policy_homeowners",
                "type": "POLICY",
                "name": "Homeowners Insurance",
                "description": "A homeowners insurance policy covering dwelling and personal property",
            },
            {
                "id": "coverage_dwelling",
                "type": "COVERAGE",
                "name": "Dwelling Coverage",
                "description": "Covers physical damage to the dwelling structure",
            },
            {
                "id": "exclusion_flood",
                "type": "EXCLUSION",
                "name": "Flood Damage",
                "description": "Flood and surface water damage is not covered",
            },
            {
                "id": "limit_dwelling",
                "type": "LIMIT",
                "name": "$350,000 Dwelling Limit",
                "description": "Maximum coverage for dwelling damage",
            },
        ],
        "relationships": [
            {
                "source": "policy_homeowners",
                "target": "coverage_dwelling",
                "type": "COVERS",
                "description": "Homeowners policy provides dwelling coverage",
            },
            {
                "source": "policy_homeowners",
                "target": "exclusion_flood",
                "type": "EXCLUDES",
                "description": "Homeowners policy excludes flood damage",
            },
            {
                "source": "coverage_dwelling",
                "target": "limit_dwelling",
                "type": "HAS_LIMIT",
                "description": "Dwelling coverage has a $350,000 limit",
            },
        ],
    }


@pytest.fixture
def tmp_documents_dir(tmp_path) -> str:
    """
    Create a temporary directory with sample document files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the temporary directory containing test documents.
    """
    # Create a sample text file
    txt_file = tmp_path / "test_policy.txt"
    txt_file.write_text(
        "SAMPLE INSURANCE POLICY\n\n"
        "This policy provides coverage for property damage.\n"
        "Coverage limit: $100,000.\n\n"
        "Exclusions: This policy does not cover flood or earthquake."
    )

    return str(tmp_path)
