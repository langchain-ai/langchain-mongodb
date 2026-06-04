"""Integration tests for $rerank in MongoDBAtlasVectorSearch.

$rerank requires:
  - A real Atlas cluster (not local Docker) running MongoDB 8.3+
  - Native Reranking enabled in Atlas Project Settings
  - A Voyage AI API key configured in Atlas

Tests are skipped automatically when MONGODB_URI points to localhost / 127.0.0.1.
"""

from typing import Generator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import create_vector_search_index

from ..utils import CONNECTION_STRING, DB_NAME, PatchedMongoDBAtlasVectorSearch

COLLECTION_NAME = "langchain_test_rerank"
INDEX_NAME = "langchain-test-index-rerank"
RERANK_MODEL = "rerank-2.5"

pytestmark = pytest.mark.skipif(
    any(host in CONNECTION_STRING for host in ("localhost", "127.0.0.1")),
    reason="$rerank requires a real Atlas cluster with MongoDB 8.3+ and Native Reranking enabled",
)

# Corpus chosen so that a semantically precise query ("food between bread")
# is not an obvious vector-space nearest-neighbour of "What is a sandwich?"
# but the Voyage reranker should surface it as the top result.
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


def test_rerank_changes_ordering_vs_vector_search(
    vectorstore: MongoDBAtlasVectorSearch,
) -> None:
    """Reranked results differ in ordering from pure vector-similarity results.

    This test demonstrates that $rerank adds value beyond the raw vector score:
    the top-1 reranked document for a precise semantic query should rank the
    most relevant result higher than vector search alone might.
    """
    query = "food between bread"

    vector_results = vectorstore.similarity_search_with_score(query, k=len(DOCUMENTS))
    rerank_results = vectorstore.similarity_search_with_score(
        query,
        k=len(DOCUMENTS),
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(DOCUMENTS),
    )

    vector_order = [doc.page_content for doc, _ in vector_results]
    rerank_order = [doc.page_content for doc, _ in rerank_results]

    # The reranked top result must be the sandwich document.
    assert rerank_order[0] == "What is a sandwich?", (
        f"Expected sandwich as reranked top-1, got: {rerank_order[0]}"
    )

    # The two orderings should differ, confirming reranking had an effect.
    assert vector_order != rerank_order, (
        "Reranked and vector-only orderings are identical — "
        "either reranking had no effect or the corpus needs adjustment."
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
