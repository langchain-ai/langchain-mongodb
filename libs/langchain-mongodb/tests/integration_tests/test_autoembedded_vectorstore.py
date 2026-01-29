"""Test MongoDBAtlasVectorSearch.from_documents."""

from __future__ import annotations

import os
from typing import Dict, Generator, List

import pytest  # type: ignore[import-not-found]
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.embeddings import AutoEmbeddings
from langchain_mongodb.index import (
    create_vector_search_index,
)

from ..utils import (
    AUTOEMBED_COLLECTION_NAME,
    AUTOEMBED_IDX_NAME,
    AUTOEMBED_MODEL,
    DB_NAME,
    PatchedMongoDBAtlasVectorSearch,
)

DIMENSIONS = 5

COMMUNITY_WITH_SEARCH = os.environ.get("COMMUNITY_WITH_SEARCH", "")

pytestmark = pytest.mark.skipif(
    COMMUNITY_WITH_SEARCH == "", reason="Only run in COMMUNITY_WITH_SEARCH is set"
)


@pytest.fixture(scope="module")
def collection(client: MongoClient) -> Collection:
    if AUTOEMBED_COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(AUTOEMBED_COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][AUTOEMBED_COLLECTION_NAME]

    clxn.delete_many({})

    if not any([AUTOEMBED_IDX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        create_vector_search_index(
            collection=clxn,
            index_name=AUTOEMBED_IDX_NAME,
            dimensions=-1,
            path="text",
            filters=["c"],
            similarity=None,
            wait_until_complete=60,
            auto_embedding_model=AUTOEMBED_MODEL,
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
    return AutoEmbeddings(model=AUTOEMBED_MODEL)


@pytest.fixture(scope="module")
def autoembedded_vectorstore(
    collection: Collection,
    texts: List[str],
    autoembeddings: AutoEmbeddings,
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
        embedding_key=None,
        relevance_score_fn=None,
    )

    yield vectorstore_from_texts

    vectorstore_from_texts.collection.delete_many({})


def test_search_with_metadatas_and_pre_filter(
    autoembedded_vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Confirm the presence of metadata in output
    output = autoembedded_vectorstore.similarity_search("Sandwich", k=1)
    assert len(output) == 1
    metakeys = [list(d.keys())[0] for d in metadatas]
    assert any([key in output[0].metadata for key in metakeys])


def test_search_filters_all(
    autoembedded_vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Test filtering out
    does_not_match_filter = autoembedded_vectorstore.similarity_search(
        "Sandwich", k=1, pre_filter={"c": {"$lte": 0}}
    )
    assert does_not_match_filter == []


def test_search_pre_filter(
    autoembedded_vectorstore: PatchedMongoDBAtlasVectorSearch, metadatas: List[Dict]
) -> None:
    # Test filtering with expected output
    matches_filter = autoembedded_vectorstore.similarity_search(
        "Sandwich", k=3, pre_filter={"c": {"$gt": 0}}
    )
    assert len(matches_filter) == 1


def test_similarity_search(
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


def test_create_with_str(
    collection: Collection,
    texts: List[str],
    metadatas: List[dict],
) -> None:
    vs = PatchedMongoDBAtlasVectorSearch.from_texts(
        texts=texts,
        embedding=AUTOEMBED_MODEL,
        metadatas=metadatas,
        collection=collection,
        index_name=AUTOEMBED_IDX_NAME,
        embedding_key=None,
        relevance_score_fn=None,
    )
    # Test similarity_search method for autoembedding
    query_text = "Sandwich"

    # Perform search
    output = vs.similarity_search_with_score(query_text, k=2)

    # Should return results
    assert len(output) == 2
    # Results should be Document objects
    assert all(hasattr(doc, "page_content") for doc, _ in output)
    assert all(hasattr(doc, "metadata") for doc, _ in output)

    vs.collection.delete_many({})
