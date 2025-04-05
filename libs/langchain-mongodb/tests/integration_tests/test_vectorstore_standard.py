"""Test MongoDBAtlasVectorSearch.from_documents."""

from __future__ import annotations

import pytest  # type: ignore[import-not-found]
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb.index import (
    create_vector_search_index,
)

from ..utils import DB_NAME, PatchedMongoDBAtlasVectorSearch

COLLECTION_NAME = "langchain_test_standard"
INDEX_NAME = "langchain-test-index-standard"
DIMENSIONS = 6


@pytest.fixture
def collection(client: MongoClient) -> Collection:
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


class TestMongoDBAtlasVectorSearch(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self, collection) -> VectorStore:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = PatchedMongoDBAtlasVectorSearch(
            collection, self.get_embeddings(), index_name=INDEX_NAME
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        return store
