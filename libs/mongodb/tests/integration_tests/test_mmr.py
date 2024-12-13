"""Test max_marginal_relevance_search."""

from __future__ import annotations

import os

import pytest  # type: ignore[import-not-found]
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb.index import (
    create_vector_search_index,
)

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_vectorstores"
INDEX_NAME = "langchain-test-index-vectorstores"
DIMENSIONS = 5


@pytest.fixture()
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
            dimensions=5,
            path="embedding",
            filters=["c"],
            similarity="cosine",
            wait_until_complete=60,
        )

    return clxn


@pytest.fixture
def embeddings() -> Embeddings:
    return ConsistentFakeEmbeddings(DIMENSIONS)


def test_mmr(embeddings: Embeddings, collection: Collection) -> None:
    texts = ["foo", "foo", "fou", "foy"]
    collection.delete_many({})
    vectorstore = PatchedMongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME,
    )
    query = "foo"
    output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
    assert len(output) == len(texts)
    assert output[0].page_content == "foo"
    assert output[1].page_content != "foo"
