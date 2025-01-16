import os

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.results import BulkWriteResult

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_mongodb.graphrag.prompts import entity_prompt, query_prompt

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_graphrag"


@pytest.fixture(scope="module")
def collection() -> Collection:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.delete_many({})
    return collection


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI for chat responses."
)
@pytest.fixture(scope="module")
def entity_extraction_model() -> BaseChatModel:
    """LLM for converting documents into Graph of Entities and Relationships"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


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
  "ID": "Best Practices",
  "type": "Guideline",
  "attributes": {{
    "style": "Follow the Python Style Guide (PEP 8)",
    "documentation": "Document public APIs using docstrings and examples",
    "testing": "Integration Tests for new functionality, regression tests for bug-fixes"
  }},
  "relationships": {{
    "plannedFeature": [
      {{
        "target": "PYTHON-1834",
        "attributes": {{
          "description": "Auto code formatting"
        }}
      }}
    ],
    "reference": [
      {{
        "target": "Node Team Practices",
        "attributes": {{
          "url": "https://wiki.corp.mongodb.com/display/DRIVERS/Node+Team+Practices"
        }}
      }}
    ]
  }}
}}
"""


@pytest.fixture(scope="module")
def graph_store(collection, entity_extraction_model, documents) -> MongoDBGraphStore:
    store = MongoDBGraphStore(
        collection, entity_extraction_model, entity_prompt, query_prompt
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
    assert isinstance(bulkwrite_results[0], BulkWriteResult)
    return store


@pytest.fixture(scope="module")
def query_connection():
    return "What is the connection between ACME Corporation and GreenTech Ltd.?"


def test_add_docs_store(graph_store):
    # Add entities to the collection by extracting from documents
    extracted_entities = list(graph_store.collection.find({}))
    assert 4 <= len(extracted_entities) < 8


def test_extract_entity_names(graph_store, query_connection):
    query_entity_names = graph_store.extract_entity_names(query_connection)
    assert set(query_entity_names) == {"ACME Corporation", "GreenTech Ltd."}

    no_names = graph_store.extract_entity_names("")
    assert isinstance(no_names, list)
    assert len(no_names) == 0


def test_related_entities(graph_store):
    entity_names = ["ACME Corporation", "GreenTech Ltd."]
    related_entities = graph_store.related_entities(entity_names)
    assert len(related_entities) >= 4

    no_entities = graph_store.related_entities([])
    assert isinstance(no_entities, list)
    assert len(no_entities) == 0


def test_additional_entity_examples(entity_extraction_model, entity_example, documents):
    # Test additional examples
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[f"{COLLECTION_NAME}_addl_examples"]
    collection.delete_many({})
    store_with_addl_examples = MongoDBGraphStore(
        collection, entity_extraction_model, entity_examples=entity_example
    )
    store_with_addl_examples.add_documents(documents)
    entity_names = ["ACME Corporation", "GreenTech Ltd."]
    new_entities = store_with_addl_examples.related_entities(entity_names)
    assert len(new_entities) >= 2


def test_chat_response(graph_store, query_connection):
    """Displays querying an existing Knowledge Graph Database"""
    answer = graph_store.chat_response(query_connection)
    assert isinstance(answer, AIMessage)
    assert "partner" in answer.content.lower()


def test_similarity_search(graph_store, query_connection):
    docs = graph_store.similarity_search(query_connection)
    assert len(docs) >= 4
    assert all(
        set(d.keys()) == {"ID", "type", "relationships", "attributes"} for d in docs
    )


def test_validator(documents, entity_extraction_model):
    client = MongoClient(MONGODB_URI)
    clxn = client[DB_NAME]["langchain_test_validated_entities"]
    clxn.delete_many({})
    store = MongoDBGraphStore(clxn, entity_extraction_model, validate=True)
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)
