from __future__ import annotations

import os
from typing import Generator

import pytest
from flaky import flaky  # type:ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.results import BulkWriteResult

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_mongodb.graphrag.prompts import entity_prompt, query_prompt

from ..utils import CONNECTION_STRING, DB_NAME

COLLECTION_NAME = "langchain_test_graphrag"


@pytest.fixture(scope="module")
def collection() -> Generator[Collection]:
    client = MongoClient(CONNECTION_STRING)
    db = client[DB_NAME]
    db[COLLECTION_NAME].drop()
    collection = db.create_collection(COLLECTION_NAME)
    yield collection
    client.close()


if "OPENAI_API_KEY" not in os.environ:
    pytest.skip(
        "GraphRAG tests require OpenAI for chat responses.", allow_module_level=True
    )


@pytest.fixture(scope="module")
def entity_extraction_model() -> BaseChatModel | None:
    """LLM for converting documents into Graph of Entities and Relationships"""
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0.0, cache=False)
    except Exception:
        pass
    return None


@pytest.fixture(scope="module")
def documents():
    return [
        Document(
            page_content="""
ACME Corporation Expands Renewable Energy Efforts

New York, NY — ACME Corporation, a leading player in renewable energy and logistics,
has announced a partnership with GreenTech Ltd. to launch the SolarGrid Initiative.
This ambitious project aims to expand solar panel networks in rural areas,
addressing energy inequity while supporting sustainable development.

John Doe, ACME’s Chief Technology Officer, emphasized the importance of collaboration in tackling climate change.
"GreenTech’s expertise in solar technology makes them an ideal partner," Doe said,
adding that the initiative builds on the companies’ successful partnership, which began in 2021.
"""
        ),
        Document(
            page_content="""
GreenTech Ltd. Leads SolarGrid Initiative

San Francisco, CA — GreenTech Ltd. has emerged as a leader in renewable energy projects with the SolarGrid Initiative,
a collaboration with ACME Corporation. Jane Smith, the project’s Lead Manager, highlighted its ambitious goal:
providing affordable solar energy to underserved communities.

GreenTech, headquartered in San Francisco, has worked closely with ACME since their partnership began in May 2021.
"Together, we’re combining innovation and impact," Smith stated.

The project is set to expand across multiple regions, marking a milestone in the renewable energy sector.
"""
        ),
    ]


@pytest.fixture(scope="module")
def entity_example():
    return """
Input:
Best Practices
Style - Follow the Python Style Guide (PEP 8) wherever possible.
We are planning to add auto code formatting in: PYTHON-1834.
Documentation - Document public APIs using docstrings and examples whenever possible.
Testing - New functionality should be accompanied with Integration Tests when specification tests aren’t provided. Bug-fixes should be accompanied by regression testing.
The Node Team has a comprehensive guide to best practices: https://wiki.corp.mongodb.com/display/DRIVERS/Node+Team+Practices.

Output:
{{
  "_id": "Best Practices",
  "type": "Guideline",
  "attributes": {{
    "style": "Follow the Python Style Guide (PEP 8)",
    "documentation": "Document public APIs using docstrings and examples",
    "testing": "Integration Tests for new functionality, regression tests for bug-fixes"
  }},
  "relationships": {{
    "target_ids": [
      "PYTHON-1834",
      "Node Team Practices"

    ]
    "types": [
      "plannedFeature",
      "reference"
    ],
    "attributes": [
      {{"description": "Auto code formatting"}},
      {{"url": "https://wiki.corp.mongodb.com/display/DRIVERS/Node+Team+Practices"}}
    ]
  }}
}}
"""


@pytest.fixture(scope="module")
def graph_store(collection, entity_extraction_model, documents) -> MongoDBGraphStore:
    store = MongoDBGraphStore(
        collection=collection,
        entity_extraction_model=entity_extraction_model,
        entity_prompt=entity_prompt,
        query_prompt=query_prompt,
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    assert isinstance(bulkwrite_results[0], BulkWriteResult)
    return store


@pytest.fixture
def query_connection():
    return "How are Jane Smith and John Doe related?"


def test_add_docs_store(graph_store):
    # Add entities to the collection by extracting from documents
    extracted_entities = list(graph_store.collection.find({}))
    assert 4 <= len(extracted_entities) < 8


@flaky
def test_extract_entity_names(graph_store, query_connection):
    query_entity_names = graph_store.extract_entity_names(query_connection)
    assert set(query_entity_names) == {"John Doe", "Jane Smith"}

    no_names = graph_store.extract_entity_names("")
    assert isinstance(no_names, list)
    assert len(no_names) == 0


@flaky
def test_related_entities(graph_store):
    entity_names = ["John Doe", "Jane Smith"]
    related_entities = graph_store.related_entities(entity_names)
    assert len(related_entities) >= 4

    no_entities = graph_store.related_entities([])
    assert isinstance(no_entities, list)
    assert len(no_entities) == 0


@flaky
def test_additional_entity_examples(entity_extraction_model, entity_example, documents):
    # First, create one client just to drop any existing collections
    client = MongoClient(CONNECTION_STRING)
    clxn_name = f"{COLLECTION_NAME}_addl_examples"
    client[DB_NAME][clxn_name].drop()
    # Test additional examples
    store_with_addl_examples = MongoDBGraphStore(
        connection_string=CONNECTION_STRING,
        database_name=DB_NAME,
        collection_name=clxn_name,
        entity_extraction_model=entity_extraction_model,
        entity_prompt=entity_prompt,
        query_prompt=query_prompt,
        entity_examples=entity_example,
    )
    store_with_addl_examples.collection.drop()

    store_with_addl_examples.add_documents(documents)
    entity_names = ["ACME Corporation", "GreenTech Ltd."]
    new_entities = store_with_addl_examples.related_entities(entity_names)
    assert len(new_entities) >= 2


@flaky
def test_chat_response(graph_store, query_connection):
    """Displays querying an existing Knowledge Graph Database"""
    answer = graph_store.chat_response(query_connection)
    assert isinstance(answer, AIMessage)
    assert "acme corporation" in answer.content.lower()


@flaky
def test_similarity_search(graph_store, query_connection):
    docs = graph_store.similarity_search(query_connection)
    assert len(docs) >= 4
    assert all({"_id", "type", "relationships"}.issubset(set(d.keys())) for d in docs)
    assert any("depth" in d.keys() for d in docs)
    assert any("attributes" in d.keys() for d in docs)


@flaky
def test_validator(documents, entity_extraction_model):
    # Case 1. No existing collection.
    client = MongoClient(CONNECTION_STRING)
    clxn_name = f"{COLLECTION_NAME}_validation"
    client[DB_NAME][clxn_name].drop()
    # now we call with validation that can be added without db admin privileges
    store = MongoDBGraphStore(
        connection_string=CONNECTION_STRING,
        database_name=DB_NAME,
        collection_name=clxn_name,
        entity_extraction_model=entity_extraction_model,
        validate=True,
        validation_action="error",
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    entities = store.collection.find({}).to_list()
    # Using subset because SolarGrid Initiative is not always considered an entity
    assert {"Person", "Organization"}.issubset(set(e["type"] for e in entities))
    client.close()

    # Case 2: Existing collection with a validator
    client = MongoClient(CONNECTION_STRING)
    clxn_name = f"{COLLECTION_NAME}_validation"
    collection = client[DB_NAME][clxn_name]
    collection.delete_many({})

    store = MongoDBGraphStore(
        collection=collection,
        entity_extraction_model=entity_extraction_model,
        validate=True,
        validation_action="error",
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    collection.drop()
    client.close()

    # Case 3: Existing collection without a validator
    client = MongoClient(CONNECTION_STRING)
    clxn_name = f"{COLLECTION_NAME}_validation"
    collection = client[DB_NAME].create_collection(clxn_name)
    store = MongoDBGraphStore(
        collection=collection,
        entity_extraction_model=entity_extraction_model,
        validate=True,
        validation_action="error",
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    client.close()


@flaky
def test_allowed_entity_types(documents, entity_extraction_model):
    """Add allowed_entity_types. Use the validator to confirm behaviour."""
    allowed_entity_types = ["Person"]
    # drop collection
    client = MongoClient(CONNECTION_STRING)
    collection_name = f"{COLLECTION_NAME}_allowed_entity_types"
    client[DB_NAME][collection_name].drop()
    # create knowledge graph with only allowed_entity_types
    # this changes the schema at runtime
    store = MongoDBGraphStore(
        allowed_entity_types=allowed_entity_types,
        validate=True,
        validation_action="error",
        connection_string=CONNECTION_STRING,
        database_name=DB_NAME,
        collection_name=collection_name,
        entity_extraction_model=entity_extraction_model,
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    entities = store.collection.find({}).to_list()
    assert set(e["type"] for e in entities) == {"Person"}
    all([len(e["relationships"].get("target_ids", [])) == 0 for e in entities])
    all([len(e["relationships"].get("types", [])) == 0 for e in entities])
    all([len(e["relationships"].get("attributes", [])) == 0 for e in entities])


@flaky
def test_allowed_relationship_types(documents, entity_extraction_model):
    # drop collection
    client = MongoClient(CONNECTION_STRING)
    clxn_name = f"{COLLECTION_NAME}_allowed_relationship_types"
    client[DB_NAME][clxn_name].drop()
    collection = client[DB_NAME].create_collection(clxn_name)
    collection.drop()
    # create knowledge graph with only allowed_relationship_types=["partner"]
    # this changes the schema at runtime
    store = MongoDBGraphStore(
        allowed_relationship_types=["partner"],
        validate=True,
        validation_action="error",
        connection_string=CONNECTION_STRING,
        database_name=DB_NAME,
        collection_name=clxn_name,
        entity_extraction_model=entity_extraction_model,
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    relationships = set()
    for ent in store.collection.find({}):
        relationships.update(set(ent.get("relationships", {}).get("types", [])))
    assert relationships == {"partner"}
