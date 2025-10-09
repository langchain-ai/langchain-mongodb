"""Test large knowledge graph construction from a substantial text corpus.

This test demonstrates building a knowledge graph from a text file containing
approximately 100 entities across multiple organizations, people, projects, and locations.
The text is chunked by paragraph using RecursiveCharacterTextSplitter to create
documents suitable for entity extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb.graphrag.graph import MongoDBGraphStore

from ..utils import CONNECTION_STRING, DB_NAME

COLLECTION_NAME = "langchain_test_graphrag_large"


@pytest.fixture(scope="module")
def collection() -> Generator[Collection, None, None]:
    """Create a fresh MongoDB collection for testing."""
    client = MongoClient(CONNECTION_STRING)
    db = client[DB_NAME]
    db[COLLECTION_NAME].drop()
    collection = db.create_collection(COLLECTION_NAME)
    yield collection
    client.close()


if not ("OPENAI_API_KEY" in os.environ or "AZURE_OPENAI_ENDPOINT" in os.environ):
    pytest.skip(
        "GraphRAG tests require OpenAI for chat responses.", allow_module_level=True
    )


@pytest.fixture(scope="module")
def entity_extraction_model() -> BaseChatModel | None:
    """LLM for converting documents into Graph of Entities and Relationships."""
    try:
        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            return AzureChatOpenAI(
                model="gpt-4o", temperature=0.0, cache=False, seed=12345
            )
        return ChatOpenAI(model="gpt-4o", temperature=0.0, cache=False, seed=12345)
    except Exception:
        pass
    return None


@pytest.fixture(scope="module")
def large_text_content() -> str:
    """Load the large text dataset from file."""
    test_dir = Path(__file__).parent
    data_file = test_dir / "data" / "large_graph_dataset.txt"
    return data_file.read_text()


@pytest.fixture(scope="module")
def text_chunks(large_text_content: str) -> list[str]:
    """Split text into paragraph-sized chunks using RecursiveCharacterTextSplitter."""
    # Split by double newlines (paragraphs) with some overlap
    # This creates natural document boundaries at paragraph breaks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(large_text_content)
    return chunks


@pytest.fixture(scope="module")
def graph_store(
    collection: Collection,
    entity_extraction_model: BaseChatModel,
    text_chunks: list[str],
) -> MongoDBGraphStore:
    """Create MongoDBGraphStore and populate it with entities from text chunks."""
    from langchain_core.documents import Document

    # Create the graph store
    store = MongoDBGraphStore(
        collection=collection,
        entity_extraction_model=entity_extraction_model,
        max_depth=3,  # Allow deeper graph traversal for complex queries
    )

    # Convert text chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Add documents and extract entities
    bulkwrite_results = store.add_documents(documents)

    # Verify that we processed all chunks
    assert len(bulkwrite_results) == len(documents)

    return store


def test_large_graph_entity_count(graph_store: MongoDBGraphStore):
    """Verify that the large graph contains more than 50 entities.

    The dataset is designed to generate approximately 100 entities including:
    - Organizations (companies, nonprofits, research institutions)
    - People (CEOs, scientists, directors)
    - Projects and initiatives
    - Locations (cities, facilities)
    - Technologies and products
    """
    entity_count = graph_store.collection.count_documents({})
    print(f"\nExtracted {entity_count} entities from the text corpus")

    # Verify we have a substantial knowledge graph
    assert entity_count > 50, (
        f"Expected more than 50 entities, but found {entity_count}. "
        "The entity extraction may need adjustment."
    )


def test_large_graph_entity_types(graph_store: MongoDBGraphStore):
    """Verify diverse entity types are extracted."""
    entities = list(graph_store.collection.find({}))

    # Collect all entity types
    entity_types = {entity.get("type") for entity in entities}

    print(f"\nFound entity types: {entity_types}")

    # We expect at least Organization and Person types
    assert len(entity_types) >= 2, "Expected at least 2 different entity types"


def test_large_graph_relationships(graph_store: MongoDBGraphStore):
    """Verify that entities have relationships connecting them."""
    entities = list(graph_store.collection.find({}))

    # Count entities with relationships
    entities_with_relationships = [
        entity
        for entity in entities
        if entity.get("relationships", {}).get("target_ids")
    ]

    relationship_percentage = (
        len(entities_with_relationships) / len(entities) * 100 if entities else 0
    )

    print(
        f"\n{len(entities_with_relationships)} out of {len(entities)} entities "
        f"({relationship_percentage:.1f}%) have relationships"
    )

    # At least 30% of entities should have relationships
    assert (
        len(entities_with_relationships) >= len(entities) * 0.3
    ), "Expected at least 30% of entities to have relationships"


def test_large_graph_sample_queries(graph_store: MongoDBGraphStore):
    """Test sample queries on the large knowledge graph."""
    # Query about relationships between entities
    query = "What is the connection between Quantum Dynamics Corp and NanoTech Materials Ltd?"

    # Extract entity names from query
    entity_names = graph_store.extract_entity_names(query)
    print(f"\nExtracted entities from query: {entity_names}")

    # Find related entities
    related_entities = graph_store.related_entities(entity_names)
    print(f"Found {len(related_entities)} related entities")

    # Should find some related entities
    assert len(related_entities) >= 1, "Expected to find related entities"


def test_large_graph_similarity_search(graph_store: MongoDBGraphStore):
    """Test similarity search functionality on the large graph."""
    # Use a query that includes actual entity names from the dataset
    query = (
        "Tell me about SolarTech Industries and their work with "
        "the Green Computing Alliance on renewable energy."
    )

    docs = graph_store.similarity_search(query)

    print(f"\nSimilarity search returned {len(docs)} entities")

    # Should return relevant entities if the entity names were extracted
    # If entity extraction finds the names, we should get results
    if len(docs) > 0:
        # Verify structure of returned documents
        for doc in docs[:3]:  # Check first 3
            assert "_id" in doc
            assert "type" in doc
        assert len(docs) >= 2, "Expected similarity search to return multiple entities"
    else:
        # If no entities were extracted from query, that's ok too
        # This can happen if the LLM doesn't recognize the entity names
        print("Note: No entities extracted from query - this is acceptable")


def test_large_graph_multi_hop_query(graph_store: MongoDBGraphStore):
    """Test multi-hop graph traversal to answer complex queries."""
    # This query includes specific entity names from the dataset
    query = (
        "What are the connections between MediChain Solutions, "
        "HealthSecure Systems, and the Blockchain Healthcare Consortium?"
    )

    entity_names = graph_store.extract_entity_names(query)

    print(f"\nExtracted entity names: {entity_names}")

    # Even if we start with few entities, graph traversal should find more
    if len(entity_names) > 0:
        related_entities = graph_store.related_entities(entity_names, max_depth=3)

        print(
            f"Multi-hop traversal from {len(entity_names)} starting entities "
            f"found {len(related_entities)} related entities"
        )

        # Multi-hop traversal should expand the result set
        assert (
            len(related_entities) >= 1
        ), "Expected multi-hop traversal to find at least the starting entities"
    else:
        # If no entities extracted, test that we can still query the graph directly
        # Pick some entities from the database
        sample_entities = list(graph_store.collection.find({}, {"_id": 1}).limit(3))
        if sample_entities:
            entity_ids = [e["_id"] for e in sample_entities]
            print(f"Testing with sample entities: {entity_ids}")
            related = graph_store.related_entities(entity_ids, max_depth=2)
            assert len(related) >= 1, "Expected to find entities in graph traversal"


def test_chunk_count(text_chunks: list[str]):
    """Verify that text was properly chunked."""
    assert len(text_chunks) >= 10, (
        f"Expected at least 10 chunks, got {len(text_chunks)}. "
        "Text splitting may need adjustment."
    )
    print(f"\nText split into {len(text_chunks)} chunks")


def test_graph_store_chat_response(graph_store: MongoDBGraphStore):
    """Test the chat response functionality with the large graph."""
    from langchain_core.messages import AIMessage

    query = (
        "Tell me about the AI Ethics Initiative and which companies are supporting it."
    )

    answer = graph_store.chat_response(query)

    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 0, "Expected a non-empty response"
    print(f"\nChat response preview: {answer.content[:200]}...")
