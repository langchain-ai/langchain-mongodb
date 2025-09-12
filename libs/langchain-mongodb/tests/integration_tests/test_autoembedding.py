"""Test MongoDB Atlas Vector Search on Collections with Auto-embedding indexes."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest  # type: ignore[import-not-found]
from bson import ObjectId
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.vectorstores import AutoEmbeddingVectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from langchain_mongodb.utils import oid_to_str

from ..utils import DB_NAME, ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

INDEX_NAME = "langchain-test-index-autoembedding"
COLLECTION_NAME = "langchain_test_autoembedding"

@pytest.fixture
def collection(client: MongoClient) -> Collection:
    clx = client[DB_NAME][COLLECTION_NAME]
    clx.delete_many({})
    return clx

@pytest.fixture(scope="module")
def texts() -> List[str]:
    return [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]

@pytest.fixture(scope="module")
def metadatas() -> List[Dict[str, Any]]:
     return [
        {"a": 1},
        {"b": 1},
        {"c": 1},
        {"d": 1, "e": 2},
    ]



def test_autoembedding(collection: Collection, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
    vectorstore = AutoEmbeddingVectorStore(
        collection=collection,
        index_name=INDEX_NAME,
        text_key="text",
        model="voyage-3-large",
        auto_create_index=True,
        auto_index_timeout=60
    )

    assert any([ix["name"] == INDEX_NAME for ix in collection.list_search_indexes()])

    vectorstore.collection.delete_many({})

    n_docs = len(texts)
    documents = [
        Document(page_content=texts[i], metadata=metadatas[i]) for i in range(n_docs)
    ]
    result_ids = vectorstore.add_documents(documents)
    assert len(result_ids) == n_docs

    found = vectorstore.similarity_search_with_score("Animals", k=2)
    assert len(found) == 2
    assert all(res[0].page_content in ["Dogs are tough.", "Cats have fluff."] for res in found)



'''
# TODO - Run standard test suite (See test_vectorstore_standard.py)
class TestMongoDBAtlasVectorSearch(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self, collection) -> VectorStore:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = AutoEmbeddingVectorStore(
            collection, index_name=INDEX_NAME, text_key="text", model="voyage-3-large"
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        return store
'''


# TODO
#   1. Test add_texts. We have this Embeddings arg to deal with.