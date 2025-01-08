import json
import os
from typing import Dict, List

import pytest

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_mongodb.graphrag.prompts import entity_prompt

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_graphrag"


@pytest.fixture(scope="module")
def raw_document():
    return Document(page_content="""
Mohammed Akbar, Casey Sneddon, and Dara Achariya live in adjacent homes on 86th street in Rivertown. Mohammed is a data scientist who works remotely for ACME, an investment bank. Known for his love of board games, he often hosts game nights where the trio gathers to relax.
Casey is a backend engineer at Sprockets LLC. His garage doubles as a workshop where tech gadgets and tennis racquets share space. Casey and Dara are married. Dara, an HR specialist also at Sprockets, has a knack for organizing both work teams and weekend tennis matches.
Despite being single, Mohammed is an integral part of the group’s activities. He will often play doubles tennis together with Casey, his wife, and their mutual friend Njeri, a local university student studying Computer Science.
""")


@pytest.fixture(scope="module")
def query():
    return "Sprockets LLC is looking to hire a new employee in Rivertown.  Casey Sneddon is an employee there. Does he have any friends that might be good candidates?"

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
    return MongoDBGraphStore(collection, entity_extraction_model, entity_prompt)


def test_query_prompt(graph_store, query):
    query_entities = graph_store.extract_entities_from_query(query)
    assert len(query_entities) == 2
    assert set(query_entities["Casey Sneddon"]) == {'friend', 'employee'}

def test_graph_store(graph_store, raw_document, query):
    graph_store.add_documents(raw_document)
    extracted_entities = list(graph_store.collection.find({}))
    assert len(extracted_entities) > 1

    # TODO - Update API of this to similarity_search(query, **)
    query_entities = graph_store.extract_entities_from_query(query)
    assert query_entities is not None

    related = {}
    for entity, relationships in query_entities.items():
        related.update(graph_store.related_entities(entity, relationships))
    assert len(related) > 0
