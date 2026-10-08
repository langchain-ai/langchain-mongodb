import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.base import empty_checkpoint
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure

from langgraph.checkpoint.mongodb import aio
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "async_checkpoints"


@pytest_asyncio.fixture
async def saver() -> AsyncGenerator:
    client: AsyncMongoClient = AsyncMongoClient(MONGODB_URI)
    db = client[DB_NAME]
    for clxn in await db.list_collection_names():
        await db.drop_collection(clxn)
    await client.close()
    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME
    ) as checkpointer:
        yield checkpointer


async def test_asearch(input_data: dict[str, Any], saver: AsyncMongoDBSaver) -> None:
    await saver.aput(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    await saver.aput(
        input_data["config_2"], input_data["chkpnt_2"], input_data["metadata_2"], {}
    )
    await saver.aput(
        input_data["config_3"], input_data["chkpnt_3"], input_data["metadata_3"], {}
    )

    query_1 = {"source": "input"}  # search by 1 key
    query_2 = {"step": 1, "writes": {"foo": "bar"}}  # search by multiple keys
    query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
    query_4 = {"source": "update", "step": 1}  # no match

    search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata["source"] == "input"

    search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata["source"] == "loop"

    search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
    assert len(search_results_3) == 3

    search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
    assert len(search_results_4) == 0

    search_results_5 = [
        c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(search_results_5) == 2
    assert {
        search_results_5[0].config["configurable"]["checkpoint_ns"],
        search_results_5[1].config["configurable"]["checkpoint_ns"],
    } == {"", "inner"}


async def test_aget_tuple(input_data: dict[str, Any], saver: AsyncMongoDBSaver) -> None:
    assert await saver.aget_tuple(input_data["config_1"]) is None

    config = await saver.aput(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    await saver.aput_writes(config, [("channel_1", "value_1")], "task-1")

    tuple_ = await saver.aget_tuple(config)
    assert tuple_ is not None
    assert tuple_.checkpoint["id"] == input_data["chkpnt_1"]["id"]
    assert tuple_.metadata["source"] == "input"
    assert tuple_.pending_writes == [("task-1", "channel_1", "value_1")]


async def test_aput_writes_are_listed(
    input_data: dict[str, Any], saver: AsyncMongoDBSaver
) -> None:
    config = await saver.aput(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    await saver.aput_writes(config, [("channel_1", "value_1")], "task-1")

    listed = [c async for c in saver.alist(input_data["config_1"])]
    assert len(listed) == 1
    assert listed[0].pending_writes == [("task-1", "channel_1", "value_1")]


async def test_adelete_thread(
    input_data: dict[str, Any], saver: AsyncMongoDBSaver
) -> None:
    config = await saver.aput(
        input_data["config_1"], input_data["chkpnt_1"], input_data["metadata_1"], {}
    )
    await saver.aput_writes(config, [("channel_1", "value_1")], "task-1")
    assert await saver.aget_tuple(config) is not None

    await saver.adelete_thread(input_data["config_1"]["configurable"]["thread_id"])

    assert await saver.aget_tuple(config) is None
    assert await saver.writes_collection.count_documents({}) == 0


async def test_injected_client_creates_indexes() -> None:
    """A client the caller owns is used as-is, and setup() adds the indexes."""
    client: AsyncMongoClient = AsyncMongoClient(MONGODB_URI)
    checkpoint_coll = "async_checkpoints_test"
    writes_coll = "async_writes_test"
    db = client[DB_NAME]
    await db.drop_collection(checkpoint_coll)
    await db.drop_collection(writes_coll)

    ttl = 100
    saver = AsyncMongoDBSaver(client, DB_NAME, checkpoint_coll, writes_coll, ttl=ttl)
    assert saver.client is client
    await saver.setup()

    def has_index(index_info: Any, keys: list[tuple[str, int]]) -> bool:
        return any(info.get("key") == keys for info in index_info.values())

    cp_indexes = await saver.checkpoint_collection.index_information()
    wr_indexes = await saver.writes_collection.index_information()

    assert has_index(
        cp_indexes, [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)]
    )
    assert has_index(cp_indexes, [("created_at", 1)])
    assert has_index(
        wr_indexes,
        [
            ("thread_id", 1),
            ("checkpoint_ns", 1),
            ("checkpoint_id", -1),
            ("task_id", 1),
            ("idx", 1),
        ],
    )
    assert has_index(wr_indexes, [("created_at", 1)])

    await db.drop_collection(checkpoint_coll)
    await db.drop_collection(writes_coll)
    await client.close()


async def test_setup_runs_once(
    saver: AsyncMongoDBSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """setup() builds indexes once, so every method may await it."""
    collections = []

    async def counting_create(
        collection: Any, compound_index: Any, ttl: Any = None
    ) -> None:
        collections.append(collection.name)

    monkeypatch.setattr(aio, "_create_saver_indexes", counting_create)
    saver.is_setup = False

    await asyncio.gather(saver.setup(), saver.setup())
    await saver.setup()

    assert len(collections) == 2  # one pass over the two collections
    assert saver.is_setup


async def test_setup_failure_is_retryable(
    saver: AsyncMongoDBSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed setup is not cached as done, so the next call tries again."""

    async def failing_create(
        collection: Any, compound_index: Any, ttl: Any = None
    ) -> None:
        raise OperationFailure("index build failed")

    monkeypatch.setattr(aio, "_create_saver_indexes", failing_create)
    saver.is_setup = False

    with pytest.raises(OperationFailure):
        await saver.setup()
    assert not saver.is_setup

    monkeypatch.undo()
    await saver.setup()
    assert saver.is_setup


async def test_alist_rejects_mql_operator_keys(saver: AsyncMongoDBSaver) -> None:
    for filter_ in (
        {"user_id": {"$exists": True}},
        {"user_id": {"$ne": "alice"}},
        {"step": {"$gt": 0}},
        {"$where": "1==1"},
        {"$or": [{"source": "loop"}]},
    ):
        with pytest.raises(ValueError, match="MongoDB operator keys are not allowed"):
            [c async for c in saver.alist(None, filter=filter_)]


async def test_sync_call_from_event_loop_is_refused(
    input_data: dict[str, Any], saver: AsyncMongoDBSaver
) -> None:
    """A blocking call made on the saver's own loop would deadlock it."""
    with pytest.raises(asyncio.InvalidStateError, match="different thread"):
        saver.get_tuple(input_data["config_1"])
    with pytest.raises(asyncio.InvalidStateError, match="different thread"):
        saver.put(
            input_data["config_1"], empty_checkpoint(), input_data["metadata_1"], {}
        )
    with pytest.raises(asyncio.InvalidStateError, match="different thread"):
        list(saver.list(None))


async def test_sync_calls_from_another_thread(
    input_data: dict[str, Any], saver: AsyncMongoDBSaver
) -> None:
    """From a worker thread the sync methods marshal onto the saver's loop."""
    config = await asyncio.to_thread(
        saver.put,
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    await asyncio.to_thread(
        saver.put_writes, config, [("channel_1", "value_1")], "task-1"
    )

    tuple_ = await asyncio.to_thread(saver.get_tuple, config)
    assert tuple_ is not None
    assert tuple_.pending_writes == [("task-1", "channel_1", "value_1")]

    listed = await asyncio.to_thread(lambda: list(saver.list(input_data["config_1"])))
    assert len(listed) == 1

    await asyncio.to_thread(
        saver.delete_thread, input_data["config_1"]["configurable"]["thread_id"]
    )
    assert await saver.aget_tuple(config) is None
