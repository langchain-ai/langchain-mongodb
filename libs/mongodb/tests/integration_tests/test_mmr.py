"""Test max_marginal_relevance_search."""

from __future__ import annotations

import os

import pytest  # type: ignore[import-not-found]
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_vectorstores"
INDEX_NAME = "langchain-test-index-vectorstores"
DIMENSIONS = 5


@pytest.fixture()
def collection() -> Collection:
    test_client: MongoClient = MongoClient(CONNECTION_STRING)
    return test_client[DB_NAME][COLLECTION_NAME]


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
