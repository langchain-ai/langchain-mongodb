import os
from collections.abc import AsyncGenerator
from datetime import datetime

import pytest_asyncio
from pymongo import MongoClient

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    TTLConfig,
)
from langgraph.store.mongodb import (
    MongoDBStore,
)

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "long_term_memory_aio"


t0 = (datetime(2025, 4, 7, 17, 29, 10, 0),)


@pytest_asyncio.fixture
async def store() -> AsyncGenerator:
    """Create a simple store following that in base's test_list_namespaces_basic"""
    client: MongoClient = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.drop_indexes()

    mdbstore = MongoDBStore(
        collection,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    )

    namespaces = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
        ("users", "123"),
        ("users", "456", "settings"),
        ("admin", "users", "789"),
    ]
    for i, ns in enumerate(namespaces):
        await mdbstore.aput(
            namespace=ns, key=f"id_{i}", value={"data": f"value_{i:02d}"}
        )

    yield mdbstore

    if client:
        client.close()


async def test_alist_namespaces(store: MongoDBStore) -> None:
    result = await store.alist_namespaces(prefix=("a", "b"))
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(suffix=("f",))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("a",), suffix=("f",))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(
        prefix=("a",),
        suffix=(
            "b",
            "f",
        ),
    )
    expected = [("a", "b", "f")]
    assert sorted(result) == sorted(expected)

    # Test max_depth and deduplication
    result = await store.alist_namespaces(prefix=("a", "b"), max_depth=3)
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "b", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("a", "*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("*", "*", "f"))
    expected = [("a", "c", "f"), ("b", "a", "f"), ("a", "b", "f")]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(suffix=("*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("a", "b"), suffix=("d", "i"))
    expected = [("a", "b", "d", "i")]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("a", "b"), suffix=("i",))
    expected = [("a", "b", "d", "i")]
    assert sorted(result) == sorted(expected)

    result = await store.alist_namespaces(prefix=("nonexistent",))
    assert result == []

    result = await store.alist_namespaces()
    assert len(result) == store.collection.count_documents({})


async def test_aget(store: MongoDBStore) -> None:
    result = store.get(namespace=("a", "b", "d", "i"), key="id_2")
    assert isinstance(result, Item)
    assert result.updated_at > result.created_at
    assert result.value == {"data": f"value_{2:02d}"}

    result = await store.aget(namespace=("a", "b", "d", "i"), key="id-2")
    assert result is None

    result = await store.aget(namespace=tuple(), key="id_2")
    assert result is None

    result = await store.aget(namespace=("a", "b", "d", "i"), key="")
    assert result is None

    # Test case: refresh_ttl is False
    result = store.collection.find_one(dict(namespace=["a", "b", "d", "i"], key="id_2"))
    assert result is not None
    expected_updated_at = result["updated_at"]

    result = await store.aget(
        namespace=("a", "b", "d", "i"), key="id_2", refresh_ttl=False
    )
    assert result is not None
    assert result.updated_at == expected_updated_at


async def test_ttl() -> None:
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}

    # refresh_on_read is True
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME + "-ttl",
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        store.collection.delete_many({})
        await store.aput(namespace=namespace, key=key, value=value)
        res = store.collection.find_one({})
        assert res is not None
        orig_updated_at = res["updated_at"]
        res = await store.aget(namespace=namespace, key=key)
        assert res is not None
        found = store.collection.find_one({})
        assert found is not None
        new_updated_at = found["updated_at"]
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is False
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME + "-ttl",
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=False),
    ) as store:
        store.collection.delete_many({})
        await store.aput(namespace=namespace, key=key, value=value)
        found = store.collection.find_one({})
        assert found is not None
        orig_updated_at = found["updated_at"]
        res = await store.aget(namespace=namespace, key=key)
        assert res is not None
        found = store.collection.find_one({})
        assert found is not None
        new_updated_at = found["updated_at"]
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at

    # ttl_config is None
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME + "-ttl",
        ttl_config=None,
    ) as store:
        store.collection.delete_many({})
        await store.aput(namespace=namespace, key=key, value=value)
        found = store.collection.find_one({})
        assert found is not None
        orig_updated_at = found["updated_at"]
        res = await store.aget(namespace=namespace, key=key)
        assert res is not None
        found = store.collection.find_one({})
        assert found is not None
        new_updated_at = found["updated_at"]
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is True but refresh_ttl=False in get()
    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME + "-ttl",
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        store.collection.delete_many({})
        await store.aput(namespace=namespace, key=key, value=value)
        found = store.collection.find_one({})
        assert found is not None
        orig_updated_at = found["updated_at"]
        res = await store.aget(refresh_ttl=False, namespace=namespace, key=key)
        assert res is not None
        found = store.collection.find_one({})
        assert found is not None
        new_updated_at = found["updated_at"]
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at


async def test_aput(store: MongoDBStore) -> None:
    n = store.collection.count_documents({})
    await store.aput(namespace=("a",), key=f"id_{n}", value={"data": f"value_{n:02d}"})
    assert store.collection.count_documents({}) == n + 1

    # include index kwarg
    await store.aput(("a",), "idx", {"data": "val"}, index=["data"])
    assert store.collection.count_documents({}) == n + 2


async def test_adelete(store: MongoDBStore) -> None:
    n_items = store.collection.count_documents({})
    await store.adelete(namespace=("a", "b", "c"), key="id_0")
    assert store.collection.count_documents({}) == n_items - 1


async def test_abatch() -> None:
    """Simple demonstration of order of batch operations.

    Read operations, regardless of their order in the list of operations,
    act on the state of the database at the beginning of the batch.
    These include GetOp SearchOp, and ListNamespacesOp.

    Write operations are applied only *after* reads!

    Cases:
    PutOp
    GetOp
    ListNameSpaces after PutOp
    PutOp as delete after PutOp

    raises:
    match_condition stuff

    - check state after ops in different order
    """
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}

    op_put = PutOp(namespace=namespace, key=key, value=value)
    op_del = PutOp(namespace=namespace, key=key, value=None)
    op_get = GetOp(namespace=namespace, key=key)
    cond_pre = MatchCondition(match_type="prefix", path=("a", "b"))
    cond_suf = MatchCondition(match_type="suffix", path=("d", "e"))
    op_list = ListNamespacesOp(match_conditions=(cond_pre, cond_suf))

    with MongoDBStore.from_conn_string(
        conn_string=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        # 1. Put 1, read it, list namespaces, and delete one item.
        #   => not any(results)
        store.collection.delete_many({})
        n_ops = 4
        results = await store.abatch([op_put, op_get, op_list, op_del])
        assert store.collection.count_documents({}) == 0
        assert len(results) == n_ops
        assert not any(results)

        # 2. delete, put, get
        # => not any(results)
        n_ops = 3
        results = await store.abatch([op_get, op_del, op_put])
        assert store.collection.count_documents({}) == 1
        assert len(results) == n_ops
        assert not any(results)

        # 3. delete, put, get
        # => get sees item from put in previous batch
        n_ops = 2
        results = await store.abatch([op_del, op_get, op_list])
        assert results[0] is None
        assert isinstance(results[1], Item)
        assert isinstance(results[2], list) and isinstance(results[2][0], tuple)


async def test_asearch_basic(store: MongoDBStore) -> None:
    result = await store.asearch(("a", "b"))
    assert len(result) == 4
    assert all(isinstance(res, Item) for res in result)

    namespace = ("a", "b", "c")
    await store.aput(namespace=namespace, key="id_foo", value={"data": "value_foo"})
    result = await store.asearch(namespace, filter={"data": "value_foo"})
    assert len(result) == 1
