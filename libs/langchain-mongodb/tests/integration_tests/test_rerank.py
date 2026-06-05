"""Integration tests for $rerank in MongoDBAtlasVectorSearch.

$rerank requires:
  - A real Atlas cluster (not local Docker) running MongoDB 8.3+
  - Native Reranking enabled in Atlas Project Settings
  - A Voyage AI API key configured in Atlas

Tests are skipped automatically when MONGODB_URI points to localhost / 127.0.0.1.

--- Key test ---

test_rerank_changes_ordering_vs_vector_search demonstrates *why* reranking adds
value over vector search alone, using a corpus that exposes a well-known weakness
of embedding models: negation blindness.

Every document in the corpus contains the words "filling" and "two slices of
bread" — but documents 2-5 use those words in a negating context ("the filling
is NOT between them").  Embedding models are poor at negation and rank the
negating documents highly due to token overlap with the query.  The Voyage AI
reranker reads the full sentence, understands the negations, and promotes the
one document that affirmatively describes filling between bread to the top.

The corpus is deliberately adversarial for vector search.  When working on
the test or the corpus, run with -s to see the full score tables.
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
RERANK_MODEL = "rerank-2.5-lite"

pytestmark = pytest.mark.skipif(
    any(
        host in os.environ.get("MONGODB_URI", "") for host in ("localhost", "127.0.0.1")
    ),
    reason="$rerank requires a real Atlas cluster with MongoDB 8.3+ and Native Reranking enabled",
)

# Every document mentions "filling", "two slices of bread", or both.
# Documents 2-5 do so only in a negating context, which embedding models
# mis-score due to token overlap.  The reranker demotes them correctly.
DOCUMENTS = [
    Document(
        page_content="A club sandwich is filling between two slices of bread — not a dessert, not a drink."
    ),
    Document(
        page_content="Bread pudding is not filling between two slices of bread — it is bread soaked in custard."
    ),
    Document(
        page_content="French toast uses two slices of bread but the filling is not between them — egg coats the outside."
    ),
    Document(
        page_content="Sourdough bread is not a filling — it is the bread itself, leavened by wild fermentation."
    ),
    Document(
        page_content="Breadcrumbs are not two slices of bread — they are dried bread crushed into small pieces."
    ),
]

CLUB_SANDWICH = DOCUMENTS[0].page_content
FRENCH_TOAST = DOCUMENTS[2].page_content


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

    num_docs_to_rerank must be larger than k: the reranker can only reorder the
    candidates that vector search returns, so fetching just k=1 would give the
    reranker a single document with nothing to reorder.
    """
    results = vectorstore.similarity_search(
        "filling between two slices of bread",
        k=1,
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(DOCUMENTS),
    )
    assert results[0].page_content == CLUB_SANDWICH


def test_rerank_changes_ordering_vs_vector_search(
    vectorstore: MongoDBAtlasVectorSearch,
) -> None:
    """Reranking corrects vector search's negation blindness.

    Every document in DOCUMENTS contains the words "filling", "two slices of
    bread", or both — but documents 2-5 use those words in a negating context
    ("the filling is NOT between them", "is NOT filling between two slices...").
    Embedding models are poor at negation and rank those documents highly due to
    token overlap.  The reranker reads the full sentence and demotes them.

    Concretely, vector search puts French toast first because "two slices of bread
    but the filling is not between them" has maximum token overlap with the query
    despite meaning the opposite.  The reranker correctly promotes the club
    sandwich — the only document that affirmatively describes filling between bread.

    Note: this test makes specific claims about model behaviour that could be
    sensitive to embedding or reranker model updates; treat a failure as a signal
    to re-examine the corpus rather than an infrastructure problem.
    """
    query = "filling between two slices of bread"

    vector_results = vectorstore.similarity_search_with_score(query, k=len(DOCUMENTS))
    rerank_results = vectorstore.similarity_search_with_score(
        query,
        k=len(DOCUMENTS),
        rerank_path="text",
        rerank_model=RERANK_MODEL,
        num_docs_to_rerank=len(DOCUMENTS),
    )

    vector_top = vector_results[0][0].page_content
    rerank_top = rerank_results[0][0].page_content
    rerank_scores = [score for _, score in rerank_results]

    # Scores must not all be identical — a constant score (e.g. 0.5987) means
    # the reranker model is not GPU-backed and is not doing real scoring.
    assert len(set(rerank_scores)) > 1, (
        f"All rerank scores are identical ({rerank_scores[0]:.4f}): "
        f"model '{RERANK_MODEL}' may not be GPU-backed."
    )

    # Vector search is fooled by negation: "filling is NOT between them" scores
    # high because it shares tokens with the query.
    assert vector_top == FRENCH_TOAST, (
        f"Expected French toast as vector top-1 (negation-confused), got: {vector_top}"
    )

    # The reranker understands negation and surfaces the correct answer.
    assert rerank_top == CLUB_SANDWICH, (
        f"Expected club sandwich as reranked top-1, got: {rerank_top}"
    )

    # The top picks differ — confirming reranking changed the outcome.
    assert vector_top != rerank_top
