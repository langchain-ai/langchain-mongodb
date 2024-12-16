"""Test MongoDBAtlasVectorSearch.from_documents."""

from __future__ import annotations

import os
from typing import Dict, Generator, List

import pytest  # type: ignore[import-not-found]
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    create_vector_search_index,
)

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_from_texts"
INDEX_NAME = "langchain-test-index-from-texts"
DIMENSIONS = 5


@pytest.fixture(scope="module")
def collection() -> Collection:
    client: MongoClient = MongoClient(CONNECTION_STRING)

    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if not any([INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        create_vector_search_index(
            collection=clxn,
            index_name=INDEX_NAME,
            dimensions=DIMENSIONS,
            path="embedding",
            filters=["c"],
            similarity="cosine",
            wait_until_complete=60,
        )

    return clxn


@pytest.fixture(scope="module")
def texts() -> List[str]:
    return [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]


@pytest.fixture(scope="module")
def metadatas() -> List[Dict]:
    return [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]


@pytest.fixture(scope="module")
def embeddings() -> Embeddings:
    return ConsistentFakeEmbeddings(DIMENSIONS)


@pytest.fixture(scope="module")
def vectorstore(
    collection: Collection,
    texts: List[str],
    embeddings: Embeddings,
    metadatas: List[dict],
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """VectorStore created with a few documents and a trivial embedding model.

    Note: PatchedMongoDBAtlasVectorSearch is MongoDBAtlasVectorSearch in all
    but one important feature. It waits until all documents are fully indexed
    before returning control to the caller.
    """
    vectorstore_from_texts = PatchedMongoDBAtlasVectorSearch.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection=collection,
        index_name=INDEX_NAME,
    )
    yield vectorstore_from_texts

    vectorstore_from_texts.collection.delete_many({})


def test_search_with_metadatas_and_pre_filter(
    vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Confirm the presence of metadata in output
    output = vectorstore.similarity_search("Sandwich", k=1)
    assert len(output) == 1
    metakeys = [list(d.keys())[0] for d in metadatas]
    assert any([key in output[0].metadata for key in metakeys])


def test_search_filters_all(
    vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Test filtering out
    does_not_match_filter = vectorstore.similarity_search(
        "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
    )
    assert does_not_match_filter == []


def test_search_pre_filter(
    vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Test filtering with expected output
    matches_filter = vectorstore.similarity_search(
        "Sandwich", k=3, pre_filter={"c": {"$gt": 0}}
    )
    assert len(matches_filter) == 1
