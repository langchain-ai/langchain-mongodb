"""Shared fixtures for the langgraph-store-mongodb integration tests."""

import os
from functools import lru_cache

import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongo_search_utils import create_vector_search_index

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
AUTOEMBED_MODEL = "voyage-4"
AUTOEMBED_PROBE_COLLECTION = "langgraph_test_autoembed_probe"
AUTOEMBED_PROBE_IDX_NAME = "langgraph-test-index-autoembed-probe"

# Substring of the server error raised when the deployment has no embedding
# model registered.  The accompanying code is a generic ``UnknownError`` (8),
# so the message is the only thing that distinguishes this from a real fault.
_MODEL_NOT_REGISTERED = "not registered yet"


@lru_cache(maxsize=1)
def autoembedding_available() -> bool:
    """Whether this deployment can build an ``autoEmbed`` vector search index.

    Auto-embedding is evaluated entirely server-side: the deployment calls
    Voyage AI itself, so there is no client-side key and nothing in the
    connection string reveals whether a model is registered.  A deployment
    without one rejects the index with ``CanonicalModel: <model> not
    registered yet, supported models are: []``.

    The probe creates and drops a throwaway index.  It deliberately does not
    wait for READY: an unregistered model is refused by ``createSearchIndexes``
    itself, so a single round trip settles the question.
    """
    client: MongoClient = MongoClient(MONGODB_URI)
    clxn = client[DB_NAME][AUTOEMBED_PROBE_COLLECTION]
    try:
        create_vector_search_index(
            collection=clxn,
            index_name=AUTOEMBED_PROBE_IDX_NAME,
            path="text",
            dimensions=-1,
            similarity=None,
            auto_embedding_model=AUTOEMBED_MODEL,
        )
    except OperationFailure as exc:
        if _MODEL_NOT_REGISTERED in str(exc):
            return False
        raise
    else:
        return True
    finally:
        clxn.drop()
        client.close()


@pytest.fixture(scope="session")
def autoembedding_or_skip() -> None:
    """Skip the requesting test unless the deployment supports auto-embedding.

    Deployments expected to support it should set ``AUTOEMBED_REQUIRED``, which
    turns the skip into a failure.  Without that, a deployment that quietly
    loses its registered model leaves the suite green while covering nothing.
    """
    if autoembedding_available():
        return
    reason = (
        f"Deployment has no '{AUTOEMBED_MODEL}' embedding model registered, "
        "so autoEmbed indexes cannot be created"
    )
    if os.environ.get("AUTOEMBED_REQUIRED"):
        pytest.fail(f"{reason} (AUTOEMBED_REQUIRED is set)")
    pytest.skip(reason)
