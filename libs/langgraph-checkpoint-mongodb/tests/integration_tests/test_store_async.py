import os
from collections.abc import Generator
from typing import Union

import pytest
from pymongo import MongoClient

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    PutOp,
    SearchOp,
    TTLConfig,
)
from langgraph.store.mongodb import (
    MongoDBStore,
)

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "async_store"


@pytest.fixture
def store() -> Generator:
    """Create a simple store following that in base's test_list_namespaces_basic"""
    client: MongoClient = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.drop_indexes()

    yield MongoDBStore(
        collection,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    )

    if client:
        client.close()


async def test_batch_async(store: MongoDBStore) -> None:
    N = 100
    M = 5
    ops: list[Union[PutOp, GetOp, ListNamespacesOp, SearchOp]] = []
    for m in range(M):
        for i in range(N):
            ops.append(
                PutOp(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            ops.append(
                GetOp(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )
            ops.append(
                ListNamespacesOp(
                    match_conditions=None,
                    max_depth=m + 1,
                )
            )
            ops.append(
                SearchOp(
                    ("test",),
                )
            )
            ops.append(
                PutOp(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            ops.append(
                PutOp(("test", "foo", "bar", "baz", str(m % 2)), f"key{i}", None)
            )

    results = await store.abatch(ops)
    assert len(results) == M * N * 6
