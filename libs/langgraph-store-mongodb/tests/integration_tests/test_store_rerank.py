"""Integration tests for $rerank in MongoDBStore.search().

$rerank requires (all three must be satisfied — missing any one causes constant
scores of ~0.5987 rather than an error):
  - A real Atlas cluster (not local Docker) running MongoDB 8.3+
  - Native Reranking enabled in Atlas Project Settings
  - A Voyage AI API key configured in Atlas Project Settings

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
"""

import os
from collections.abc import Callable, Generator
from time import monotonic, sleep

import pytest
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langgraph.store.base import PutOp, SearchItem
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from langgraph.store.mongodb import (
    MongoDBStore,
    RerankConfig,
    create_vector_index_config,
)

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "rerank_search"
INDEX_NAME = "rerank_vector_index"
RERANK_MODEL = "rerank-2.5-lite"
TIMEOUT, INTERVAL = 120, 1

pytestmark = pytest.mark.skipif(
    any(host in MONGODB_URI for host in ("localhost", "127.0.0.1")),
    reason="$rerank requires a real Atlas cluster with MongoDB 8.3+ and Native Reranking enabled",
)

# Every document mentions "filling", "two slices of bread", or both.
# Documents 2-5 do so only in a negating context, which embedding models
# mis-score due to token overlap.  The reranker demotes them correctly.
ITEMS = [
    (
        "club_sandwich",
        "A club sandwich is filling between two slices of bread — not a dessert, not a drink.",
    ),
    (
        "bread_pudding",
        "Bread pudding is not filling between two slices of bread — it is bread soaked in custard.",
    ),
    (
        "french_toast",
        "French toast uses two slices of bread but the filling is not between them — egg coats the outside.",
    ),
    (
        "sourdough",
        "Sourdough bread is not a filling — it is the bread itself, leavened by wild fermentation.",
    ),
    (
        "breadcrumbs",
        "Breadcrumbs are not two slices of bread — they are dried bread crushed into small pieces.",
    ),
]

CLUB_SANDWICH = ITEMS[0][1]
FRENCH_TOAST = ITEMS[2][1]
NAMESPACE = ("food", "descriptions")


def get_embedding() -> OpenAIEmbeddings | AzureOpenAIEmbeddings:
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIEmbeddings(model="text-embedding-3-small")
    return OpenAIEmbeddings(model="text-embedding-3-small")


def wait_until(
    predicate: Callable, timeout: int = TIMEOUT, interval: int = INTERVAL
) -> None:
    start = monotonic()
    while monotonic() - start < timeout:
        if predicate():
            return
        sleep(interval)
    raise TimeoutError(f"Timed out waiting for predicate: {predicate}")


@pytest.fixture(scope="module")
def collection() -> Generator[Collection, None, None]:
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)
    coll = db.create_collection(COLLECTION_NAME)
    try:
        coll.drop_search_index(INDEX_NAME)
    except OperationFailure:
        pass
    wait_until(lambda: len(coll.list_search_indexes().to_list()) == 0)
    yield coll
    db.drop_collection(COLLECTION_NAME)
    client.close()


@pytest.fixture(scope="module")
def store(collection: Collection) -> Generator[MongoDBStore, None, None]:
    """Store with rerank enabled, pre-loaded with ITEMS and polled until indexed."""
    index_config = create_vector_index_config(
        name=INDEX_NAME,
        dims=1536,
        fields=["text"],
        embed=get_embedding(),
    )
    rerank_config: RerankConfig = {
        "model": RERANK_MODEL,
        "num_docs_to_rerank": len(ITEMS),
    }
    s = MongoDBStore(
        collection,
        index_config=index_config,
        rerank_config=rerank_config,
        auto_index_timeout=TIMEOUT,
    )
    s.batch(
        [
            PutOp(namespace=NAMESPACE, key=key, value={"text": text})
            for key, text in ITEMS
        ]
    )
    wait_until(lambda: len(s.search(NAMESPACE, query="bread")) == len(ITEMS))
    yield s


# ---------------------------------------------------------------------------
# Structural / output-characteristic tests
# ---------------------------------------------------------------------------


def test_rerank_returns_limit_results(store: MongoDBStore) -> None:
    """search() with rerank_config returns exactly limit items."""
    results = store.search(NAMESPACE, query="bread", limit=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchItem) for r in results)


def test_rerank_score_is_positive_float(store: MongoDBStore) -> None:
    """Rerank scores are positive floats."""
    results = store.search(NAMESPACE, query="bread", limit=3)
    for item in results:
        assert isinstance(item.score, float), (
            f"expected float score, got {type(item.score)}"
        )
        assert item.score > 0, f"expected positive score, got {item.score}"


def test_rerank_scores_are_descending(store: MongoDBStore) -> None:
    """Results are ordered highest rerank score first."""
    results = store.search(NAMESPACE, query="bread", limit=5)
    scores = [r.score for r in results if r.score is not None]
    assert scores == sorted(scores, reverse=True), (
        f"Scores not in descending order: {scores}"
    )


def test_rerank_scores_are_not_constant(store: MongoDBStore) -> None:
    """Rerank scores differ across documents — confirming Voyage AI is scoring.

    A constant score (e.g. 0.5987) indicates the reranker model is not GPU-backed.
    Only 'rerank-2.5-lite' is currently GPU-backed; other models return constants.
    """
    results = store.search(NAMESPACE, query="bread", limit=len(ITEMS))
    scores = [r.score for r in results]
    assert len(set(scores)) > 1, (
        f"All rerank scores are identical ({scores[0]:.4f}): "
        f"model '{RERANK_MODEL}' may not be GPU-backed."
    )


def test_rerank_num_docs_to_rerank_still_returns_limit(store: MongoDBStore) -> None:
    """Passing num_docs_to_rerank > limit still returns exactly limit items."""
    results = store.search(NAMESPACE, query="bread", limit=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Semantic value tests
# ---------------------------------------------------------------------------


def test_rerank_semantic_ordering(store: MongoDBStore) -> None:
    """$rerank surfaces the semantically correct item for an indirect query.

    num_docs_to_rerank must be > limit so the reranker has a real pool to
    reorder — fetching only limit=1 would leave the reranker with the single
    vector top-1 result and nothing to promote.
    """
    results = store.search(
        NAMESPACE, query="filling between two slices of bread", limit=1
    )
    assert results[0].value["text"] == CLUB_SANDWICH


def test_rerank_changes_ordering_vs_vector_search(
    store: MongoDBStore, collection: Collection
) -> None:
    """Reranking corrects vector search's negation blindness.

    Every document contains the words "filling", "two slices of bread", or both —
    but documents 2-5 use those words in a negating context.  Embedding models are
    poor at negation and rank those documents highly due to token overlap.  The
    reranker reads the full sentence and demotes them.

    Concretely, vector search puts French toast first because "two slices of bread
    but the filling is not between them" has maximum token overlap with the query
    despite meaning the opposite.  The reranker correctly promotes the club
    sandwich — the only document that affirmatively describes filling between bread.

    Note: this test makes specific claims about model behaviour that could be
    sensitive to embedding or reranker model updates; treat a failure as a signal
    to re-examine the corpus rather than an infrastructure problem.
    """
    query = "filling between two slices of bread"

    # Vector-only store for comparison: same collection, no rerank_config.
    vector_store = MongoDBStore(
        collection,
        index_config=create_vector_index_config(
            name=INDEX_NAME,
            dims=1536,
            fields=["text"],
            embed=get_embedding(),
        ),
    )

    vector_results = vector_store.search(NAMESPACE, query=query, limit=len(ITEMS))
    rerank_results = store.search(NAMESPACE, query=query, limit=len(ITEMS))

    vector_top = vector_results[0].value["text"]
    rerank_top = rerank_results[0].value["text"]

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
