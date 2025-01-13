import os

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
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


def test_add_docs_store(graph_store, documents, query_connection):
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


def test_similarity_search(graph_store, query_connection):
    pass


def test_respond_to_query(graph_store, query_connection):
    pass

    # answer = graph_store.respond_to_query(query_connection)
    # assert isinstance(answer, str)
    # assert "partner" in answer.lower()


@pytest.mark.skip("TODO")
def test_validator(documents, entity_extraction_model):
    clxn_name = "langchain_test_validated_entities"
    client = MongoClient(MONGODB_URI)
    clxn = client[DB_NAME][clxn_name]
    store = MongoDBGraphStore(
        clxn, entity_extraction_model, entity_prompt, query_prompt, validate=True
    )
    bulkwrite_results = store.add_documents(documents)
    assert len(bulkwrite_results) == len(documents)


def test_multihop_questions(graph_store, documents):
    questions = [
        "What is the connection between ACME Corporation and GreenTech Ltd.?",
        "Who is leading the SolarGrid Initiative, and what is their role?",
        "Which organizations are participating in the SolarGrid Initiative?",
        "What is John Doe’s role in ACME’s renewable energy projects?",
        "Which company is headquartered in San Francisco and involved in the SolarGrid Initiative?",
    ]
    assert questions
