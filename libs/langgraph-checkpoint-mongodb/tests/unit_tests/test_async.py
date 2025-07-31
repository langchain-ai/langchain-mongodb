import os
from collections.abc import AsyncGenerator
from typing import Any, Union

import pytest
import pytest_asyncio
from bson.errors import InvalidDocument
from pymongo import AsyncMongoClient, MongoClient

from langgraph.checkpoint.mongodb import AsyncMongoDBSaver, MongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "sync_checkpoints_aio"


@pytest_asyncio.fixture(params=["run_in_executor", "aio"])
async def async_saver(request: pytest.FixtureRequest) -> AsyncGenerator:
    if request.param == "aio":
        # Use async client and checkpointer
        aclient: AsyncMongoClient = AsyncMongoClient(MONGODB_URI)
        adb = aclient[DB_NAME]
        for clxn in await adb.list_collection_names():
            await adb.drop_collection(clxn)
        async with AsyncMongoDBSaver.from_conn_string(
            MONGODB_URI, DB_NAME, COLLECTION_NAME
        ) as checkpointer:
            yield checkpointer
        await aclient.close()
    else:
        # Use sync client and checkpointer with async methods run in executor
        client: MongoClient = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        for clxn in db.list_collection_names():
            db.drop_collection(clxn)
        with MongoDBSaver.from_conn_string(
            MONGODB_URI, DB_NAME, COLLECTION_NAME
        ) as checkpointer:
            yield checkpointer
        client.close()


@pytest.mark.asyncio
async def test_asearch(
    input_data: dict[str, Any], async_saver: Union[AsyncMongoDBSaver, MongoDBSaver]
) -> None:
    # save checkpoints
    await async_saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        input_data["metadata_1"],
        {},
    )
    await async_saver.aput(
        input_data["config_2"],
        input_data["chkpnt_2"],
        input_data["metadata_2"],
        {},
    )
    await async_saver.aput(
        input_data["config_3"],
        input_data["chkpnt_3"],
        input_data["metadata_3"],
        {},
    )

    # call method / assertions
    query_1 = {"source": "input"}  # search by 1 key
    query_2 = {
        "step": 1,
        "writes": {"foo": "bar"},
    }  # search by multiple keys
    query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
    query_4 = {"source": "update", "step": 1}  # no match

    search_results_1 = [c async for c in async_saver.alist(None, filter=query_1)]
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == input_data["metadata_1"]

    search_results_2 = [c async for c in async_saver.alist(None, filter=query_2)]
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == input_data["metadata_2"]

    search_results_3 = [c async for c in async_saver.alist(None, filter=query_3)]
    assert len(search_results_3) == 3

    search_results_4 = [c async for c in async_saver.alist(None, filter=query_4)]
    assert len(search_results_4) == 0

    # search by config (defaults to checkpoints across all namespaces)
    search_results_5 = [
        c async for c in async_saver.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(search_results_5) == 2
    assert {
        search_results_5[0].config["configurable"]["checkpoint_ns"],
        search_results_5[1].config["configurable"]["checkpoint_ns"],
    } == {"", "inner"}


@pytest.mark.asyncio
async def test_null_chars(
    input_data: dict[str, Any], async_saver: Union[AsyncMongoDBSaver, MongoDBSaver]
) -> None:
    """In MongoDB string *values* can be any valid UTF-8 including nulls.
    *Field names*, however, cannot contain nulls characters."""

    null_str = "\x00abc"  # string containing null character

    # 1. null string in field *value*
    null_value_cfg = await async_saver.aput(
        input_data["config_1"],
        input_data["chkpnt_1"],
        {"my_key": null_str},
        {},
    )
    null_tuple = await async_saver.aget_tuple(null_value_cfg)
    assert null_tuple.metadata["my_key"] == null_str  # type: ignore
    cps = [c async for c in async_saver.alist(None, filter={"my_key": null_str})]
    assert cps[0].metadata["my_key"] == null_str

    # 2. null string in field *name*
    with pytest.raises(InvalidDocument):
        await async_saver.aput(
            input_data["config_1"],
            input_data["chkpnt_1"],
            {null_str: "my_value"},  # type: ignore
            {},
        )
