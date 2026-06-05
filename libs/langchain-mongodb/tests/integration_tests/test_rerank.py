"""Integration tests for $rerank in MongoDBAtlasVectorSearch.

$rerank requires:
  - A real Atlas cluster (not local Docker) running MongoDB 8.3+
  - Native Reranking enabled in Atlas Project Settings
  - A Voyage AI API key configured in Atlas

Tests are skipped automatically when MONGODB_URI points to localhost / 127.0.0.1.
"""

import os
from typing import Generator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import create_vector_search_index

from ..utils import DB_NAME, PatchedMongoDBAtlasVectorSearch

COLLECTION_NAME = "langchain_test_rerank"
INDEX_NAME = "langchain-test-index-rerank"
BREAD_COLLECTION_NAME = "langchain_test_rerank_bread"
BREAD_INDEX_NAME = "langchain-test-index-rerank-bread"
RERANK_MODEL = "rerank-2.5-lite"

pytestmark = pytest.mark.skipif(
    any(
        host in os.environ.get("MONGODB_URI", "") for host in ("localhost", "127.0.0.1")
    ),
    reason="$rerank requires a real Atlas cluster with MongoDB 8.3+ and Native Reranking enabled",
)

# General mixed corpus used for structural / output-characteristic tests.
DOCUMENTS = [
    Document(page_content="Dogs are tough.", metadata={"topic": "animals"}),
    Document(page_content="Cats have fluff.", metadata={"topic": "animals"}),
    Document(page_content="What is a sandwich?", metadata={"topic": "food"}),
    Document(page_content="That fence is purple.", metadata={"topic": "objects"}),
    Document(page_content="The weather is nice today.", metadata={"topic": "weather"}),
    Document(
        page_content="Python is a programming language.", metadata={"topic": "tech"}
    ),
]


# All documents in this corpus mention bread, so their embedding vectors land
# in a similar region of the embedding space and cosine similarities are close.
# Vector search has genuine difficulty discriminating between them.
# Only the reranker's deeper semantic comprehension can identify the club
# sandwich as the document that actually describes "food between bread".
BREAD_DOCUMENTS = [
    Document(
        page_content="A club sandwich layers turkey, bacon, lettuce, and tomato between toasted bread.",
        metadata={"type": "sandwich"},
    ),
    Document(
        page_content="Bread pudding is a dessert made from stale bread soaked in custard.",
        metadata={"type": "dessert"},
    ),
    Document(
        page_content="French toast is made by soaking bread slices in egg and frying them.",
        metadata={"type": "breakfast"},
    ),
    Document(
        page_content="Sourdough is a type of bread made through a long fermentation process.",
        metadata={"type": "bread"},
    ),
    Document(
        page_content="Breadcrumbs are used as a topping for macaroni and cheese.",
        metadata={"type": "ingredient"},
    ),
]

CLUB_SANDWICH = BREAD_DOCUMENTS[0].page_content


@pytest.fixture(scope="module")
def collection(
    client: MongoClient, dimensions: int
) -> Generator[Collection, None, None]:
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if not any(INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()):
        create_vector_search_index(
            collection=clxn,
            index_name=INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
            wait_until_complete=120,
        )

    yield clxn
    clxn.delete_many({})


@pytest.fixture(scope="module")
def vectorstore(
    collection: Collection, embedding: Embeddings
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """VectorStore pre-loaded with DOCUMENTS and polled until fully indexed."""
    vs = PatchedMongoDBAtlasVectorSearch.from_documents(
        documents=DOCUMENTS,
        embedding=embedding,
        collection=collection,
        index_name=INDEX_NAME,
    )
    yield vs


@pytest.fixture(scope="module")
def bread_collection(
    client: MongoClient, dimensions: int
) -> Generator[Collection, None, None]:
    if BREAD_COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(BREAD_COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][BREAD_COLLECTION_NAME]

    clxn.delete_many({})

    if not any(BREAD_INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()):
        create_vector_search_index(
            collection=clxn,
            index_name=BREAD_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
            wait_until_complete=120,
        )

    yield clxn
    clxn.delete_many({})


@pytest.fixture(scope="module")
def bread_vectorstore(
    bread_collection: Collection, embedding: Embeddings
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """VectorStore pre-loaded with BREAD_DOCUMENTS and polled until fully indexed."""
    vs = PatchedMongoDBAtlasVectorSearch.from_documents(
        documents=BREAD_DOCUMENTS,
        embedding=embedding,
        collection=bread_collection,
        index_name=BREAD_INDEX_NAME,
    )
    yield vs


# ---------------------------------------------------------------------------
# Structural / output-characteristic tests
# ---------------------------------------------------------------------------


def test_rerank_returns_k_results(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """similarity_search with rerank_path returns exactly k documents."""
    results = vectorstore.similarity_search(
        "food", k=3, rerank_path="text", rerank_model=RERANK_MODEL
    )
    assert len(results) == 3
    assert all(isinstance(doc, Document) for doc in results)


def test_rerank_score_is_positive_float(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """Returned scores are positive floats when reranking is enabled."""
    results = vectorstore.similarity_search_with_score(
        "food", k=3, rerank_path="text", rerank_model=RERANK_MODEL
    )
    assert len(results) == 3
    for _doc, score in results:
        assert isinstance(score, float), f"expected float score, got {type(score)}"
        assert score > 0, f"expected positive score, got {score}"


def test_rerank_scores_are_descending(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """Results are ordered highest rerank score first."""
    results = vectorstore.similarity_search_with_score(
        "food", k=4, rerank_path="text", rerank_model=RERANK_MODEL
    )
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), (
        f"Scores not in descending order: {scores}"
    )


def test_rerank_score_in_metadata(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """rerankScore is present in document metadata and matches the returned score."""
    results = vectorstore.similarity_search_with_score(
        "food", k=3, rerank_path="text", rerank_model=RERANK_MODEL
    )
    for doc, score in results:
        assert "rerankScore" in doc.metadata, (
            f"rerankScore missing from metadata: {doc.metadata}"
        )
        assert doc.metadata["rerankScore"] == pytest.approx(score), (
            f"metadata rerankScore {doc.metadata['rerankScore']} != returned score {score}"
        )


def test_num_docs_to_rerank_still_returns_k(
    vectorstore: MongoDBAtlasVectorSearch,
) -> None:
    """Passing more candidates to the reranker than k still yields exactly k results."""
    results = vectorstore.similarity_search(
        "food",
        k=2,
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(DOCUMENTS),
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Semantic value tests
# ---------------------------------------------------------------------------


def test_rerank_semantic_ordering(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """$rerank surfaces the semantically correct document for an indirect query.

    "food between bread" is an oblique description of a sandwich that may not
    be the closest vector-space neighbour to "What is a sandwich?", but the
    Voyage reranker should correctly identify it as the most relevant document.
    """
    results = vectorstore.similarity_search(
        "food between bread", k=1, rerank_path="text", rerank_model=RERANK_MODEL
    )
    assert results[0].page_content == "What is a sandwich?"


def test_rerank_produces_meaningful_scores(
    bread_vectorstore: MongoDBAtlasVectorSearch,
) -> None:
    """Reranker produces distinct scores and correct top-1 in a bread-only corpus.

    Every document in BREAD_DOCUMENTS mentions bread, so their embeddings cluster
    together and cosine similarities are close.  Two things confirm the reranker
    is doing real work rather than returning a constant:

    1. The rerank scores are not all identical (Voyage AI is actually scoring).
    2. The club sandwich — the only document describing filling between bread —
       is ranked first.
    """
    query = "two slices of bread with fillings"

    rerank_results = bread_vectorstore.similarity_search_with_score(
        query,
        k=len(BREAD_DOCUMENTS),
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(BREAD_DOCUMENTS),
    )

    rerank_order = [doc.page_content for doc, _ in rerank_results]
    rerank_scores = [score for _, score in rerank_results]

    # Scores must not all be identical — a constant score (e.g. 0.5987) indicates
    # the reranker model is not GPU-backed and is not doing real scoring.
    assert len(set(rerank_scores)) > 1, (
        f"All rerank scores are identical ({rerank_scores[0]:.4f}): the reranker "
        f"model '{RERANK_MODEL}' may not be GPU-backed. Use 'rerank-2.5-lite'."
    )

    # The reranker must surface the club sandwich as top-1.
    assert rerank_order[0] == CLUB_SANDWICH, (
        f"Expected club sandwich as reranked top-1, got: {rerank_order[0]}"
    )


def test_rerank_top_result_matches_topic(vectorstore: MongoDBAtlasVectorSearch) -> None:
    """A topic-specific query surfaces a document from the correct topic group."""
    results = vectorstore.similarity_search(
        "furry household pets",
        k=2,
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(DOCUMENTS),
    )
    top_topics = [doc.metadata.get("topic") for doc in results]
    assert "animals" in top_topics, (
        f"Expected an animal document in top-2 results, got topics: {top_topics}"
    )
