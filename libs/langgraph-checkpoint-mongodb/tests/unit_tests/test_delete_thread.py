import os

from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient

from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

# Setup:
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
CHKPT_COLLECTION_NAME = "delete_thread_chkpts"
WRITES_COLLECTION_NAME = "delete_thread_writes"


def test_delete_thread() -> None:
    # Clear collections if they exist
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db[CHKPT_COLLECTION_NAME].delete_many({})
    db[WRITES_COLLECTION_NAME].delete_many({})

    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, CHKPT_COLLECTION_NAME, WRITES_COLLECTION_NAME
    ) as saver:
        # Thread 1 data
        chkpnt_1 = empty_checkpoint()
        thread_1_id = "thread-1"
        config_1 = RunnableConfig(
            configurable=dict(
                thread_id=thread_1_id, checkpoint_ns="", checkpoint_id=chkpnt_1["id"]
            )
        )
        metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"foo": "bar"},
        }

        # Thread 2 data
        chkpnt_2 = empty_checkpoint()
        thread_2_id = "thread-2"
        config_2 = RunnableConfig(
            configurable=dict(
                thread_id=thread_2_id, checkpoint_ns="", checkpoint_id=chkpnt_2["id"]
            )
        )
        metadata_2: CheckpointMetadata = {
            "source": "output",
            "step": 1,
            "writes": {"baz": "qux"},
        }

        # Save checkpoints for both threads
        saver.put(config_1, chkpnt_1, metadata_1, {})
        saver.put(config_2, chkpnt_2, metadata_2, {})

        # Add some writes
        saver.put_writes(config_1, [("channel1", "value1")], "task1")
        saver.put_writes(config_2, [("channel2", "value2")], "task2")

        # Verify we have data for both threads
        assert saver.get_tuple(config_1) is not None
        assert saver.get_tuple(config_2) is not None

        # Verify we have write data
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) > 0
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0

        # Delete thread 1
        saver.delete_thread(thread_1_id)

        # Verify thread 1 data is gone
        assert saver.get_tuple(config_1) is None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) == 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) == 0

        # Verify thread 2 data still exists
        assert saver.get_tuple(config_2) is not None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0


async def test_adelete_thread() -> None:
    # Clear collections if they exist
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db[CHKPT_COLLECTION_NAME].delete_many({})
    db[WRITES_COLLECTION_NAME].delete_many({})

    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, CHKPT_COLLECTION_NAME, WRITES_COLLECTION_NAME
    ) as saver:
        # Thread 1 data
        chkpnt_1 = empty_checkpoint()
        thread_1_id = "thread-1"
        config_1 = RunnableConfig(
            configurable=dict(
                thread_id=thread_1_id, checkpoint_ns="", checkpoint_id=chkpnt_1["id"]
            )
        )
        metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"foo": "bar"},
        }

        # Thread 2 data
        chkpnt_2 = empty_checkpoint()
        thread_2_id = "thread-2"
        config_2 = RunnableConfig(
            configurable=dict(
                thread_id=thread_2_id, checkpoint_ns="", checkpoint_id=chkpnt_2["id"]
            )
        )
        metadata_2: CheckpointMetadata = {
            "source": "output",
            "step": 1,
            "writes": {"baz": "qux"},
        }

        assert await saver.checkpoint_collection.count_documents({}) == 0

        # Save checkpoints for both threads
        await saver.aput(config_1, chkpnt_1, metadata_1, {})
        await saver.aput(config_2, chkpnt_2, metadata_2, {})

        # Add some writes
        await saver.aput_writes(config_1, [("channel1", "value1")], "task1")
        await saver.aput_writes(config_2, [("channel2", "value2")], "task2")

        # Verify we have data for both threads
        assert await saver.aget_tuple(config_1) is not None
        assert await saver.aget_tuple(config_2) is not None

        # Verify we have write data
        assert (
            await saver.checkpoint_collection.count_documents(
                {"thread_id": thread_1_id}
            )
            > 0
        )
        assert (
            await saver.writes_collection.count_documents({"thread_id": thread_1_id})
            > 0
        )
        assert (
            await saver.checkpoint_collection.count_documents(
                {"thread_id": thread_2_id}
            )
            > 0
        )
        assert (
            await saver.writes_collection.count_documents({"thread_id": thread_2_id})
            > 0
        )

        # Delete thread 1
        await saver.adelete_thread(thread_1_id)

        # Verify thread 1 data is gone
        assert await saver.aget_tuple(config_1) is None
        assert (
            await saver.checkpoint_collection.count_documents(
                {"thread_id": thread_1_id}
            )
            == 0
        )
        assert (
            await saver.writes_collection.count_documents({"thread_id": thread_1_id})
            == 0
        )

        # Verify thread 2 data still exists
        assert await saver.aget_tuple(config_2) is not None
        assert (
            await saver.checkpoint_collection.count_documents(
                {"thread_id": thread_2_id}
            )
            > 0
        )
        assert (
            await saver.writes_collection.count_documents({"thread_id": thread_2_id})
            > 0
        )
