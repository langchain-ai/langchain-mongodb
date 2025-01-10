import os

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection

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
def graph_store(collection, entity_extraction_model) -> MongoDBGraphStore:
    return MongoDBGraphStore(
        collection, entity_extraction_model, entity_prompt, query_prompt
    )


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
“GreenTech’s expertise in solar technology makes them an ideal partner,” Doe said,
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
“Together, we’re combining innovation and impact,” Smith stated.

The project is set to expand across multiple regions, marking a milestone in the renewable energy sector.
"""
        ),
    ]


@pytest.fixture(scope="module")
def query():
    return "What is the connection between ACME Corporation and GreenTech Ltd.?"


def test_graph_store(graph_store, documents, query):
    # Add entities to the collection by extracting from documents
    graph_store.add_documents(documents)
    extracted_entities = list(graph_store.collection.find({}))
    assert len(extracted_entities) > 2

    query_entity_names = graph_store.extract_entity_names(query)
    assert set(query_entity_names) == {"ACME Corporation", "GreenTech Ltd."}

    related_entities = graph_store.related_entities(query_entity_names)
    assert len(related_entities) >= 4

    answer = graph_store.respond_to_query(query, related_entities)
    assert isinstance(answer, str)
    assert "partner" in answer.lower()
