"""Test MongoDBAtlasVectorSearch.from_documents."""

from __future__ import annotations

from typing import Dict, Generator, List

import pytest  # type: ignore[import-not-found]
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.embeddings import AutoEmbedding
from langchain_mongodb.index import (
    create_autoembedded_vector_search_index
)

from ..utils import DB_NAME, ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

AUTOEMBED_COLLECTION_NAME = "langchain_test_from_texts-autoEmbed"
AUTOEMBED_IDX_NAME = "langchain-test-index-from-texts-autoEmbed"
DIMENSIONS = 5


@pytest.fixture(scope="module")
def collection(client: MongoClient) -> Collection:
    if AUTOEMBED_COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(AUTOEMBED_COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][AUTOEMBED_COLLECTION_NAME]

    clxn.delete_many({})

    if not any([AUTOEMBED_IDX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        create_autoembedded_vector_search_index(
            collection=clxn,
            index_name=AUTOEMBED_IDX_NAME,
            dimensions=DIMENSIONS,
            path="text",
            embedding=AutoEmbedding(model_name = "voyage-4"),
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
def autoembeddings() -> Embeddings:
    return AutoEmbedding(model_name="voyage-4")

@pytest.fixture(scope="module")
def autoembedded_vectorstore(
    collection: Collection,
    texts: List[str],
    autoembeddings: AutoEmbedding,
    metadatas: List[dict],
) -> Generator[MongoDBAtlasVectorSearch]:
    """VectorStore created with a few documents and a trivial embedding model.

    Note: PatchedMongoDBAtlasVectorSearch is MongoDBAtlasVectorSearch in all
    but one important feature. It waits until all documents are fully indexed
    before returning control to the caller.
    """
    vectorstore_from_texts = PatchedMongoDBAtlasVectorSearch.from_texts(
        texts=texts,
        embedding=autoembeddings,
        metadatas=metadatas,
        collection=collection,
        index_name=AUTOEMBED_IDX_NAME,
    )

    yield vectorstore_from_texts

    vectorstore_from_texts.collection.delete_many({})


def test_auto_embedded_similarity_search(
    autoembedded_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    # Test similarity_search method for autoembedding
    query_text = "Sandwich"

    # Perform search
    output = autoembedded_vectorstore.similarity_search_with_score(query_text, k=2)

    # Should return results
    assert len(output) == 2
    # Results should be Document objects
    assert all(hasattr(doc, "page_content") for doc, _ in output)
    assert all(hasattr(doc, "metadata") for doc, _ in output)