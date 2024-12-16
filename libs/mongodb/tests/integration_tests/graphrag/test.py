import os
import json
import pytest
from typing import List, Dict

from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb.graphrag.graph import MongoDBGraphStore
from langchain_mongodb.graphrag.prompts import entity_prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_graphrag"

@pytest.fixture(scope="module")
def raw_document():
    return """
Mohammed Akbar, Casey Sneddon, and Dara Achariya live in adjacent homes on 86th street in Rivertown. Mohammed is a data scientist who works remotely for ACME, an investment bank. Known for his love of board games, he often hosts game nights where the trio gathers to relax.
Casey is a backend engineer at Sprockets LLC. His garage doubles as a workshop where tech gadgets and tennis racquets share space. Casey and Dara are both married. Dara, an HR specialist also at Sprockets, has a knack for organizing both work teams and weekend tennis matches.
Despite being single, Mohammed is an integral part of the group’s activities. He will often play doubles tennis together with Casey, his wife, and their mutual friend Njeri, a local university student passionate about community engagement.
"""

@pytest.fixture(scope="module")
def collection() -> Collection:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.delete_many({})
    return collection


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Requires OpenAI for chat responses."
)
@pytest.fixture(scope="module")
def entity_extraction_model() -> BaseChatModel:
    """LLM for converting documents into Graph of Entities and Relationships"""
    return ChatOpenAI(model="gpt-4o", temperature=0.0)


@pytest.fixture(scope="module")
def graph_store(collection) -> MongoDBGraphStore:
    return MongoDBGraphStore(collection=collection)

def test_graph_store(graph_store, raw_document):
    graph_store.add_documents(raw_document)

    query = "MongoDB is looking to hire a new in Rivertown. Might Mohammed Akbar know anyone?"
    query_graph = graph_store.extract_entities(query)
    assert query_graph is not None




