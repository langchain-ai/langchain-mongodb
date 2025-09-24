import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Interrupt

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true"
)
DB_NAME: str = "test_langgraph_db"
COLLECTION_NAME: str = "checkpoints_interrupts"
WRITES_COLLECTION_NAME: str = "writes_interrupts"
TTL: int = 60 * 60


@pytest_asyncio.fixture()
async def async_saver(request: pytest.FixtureRequest) -> AsyncGenerator:
    # Use sync client and checkpointer with async methods run in executor
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    for clxn in db.list_collection_names():
        db.drop_collection(clxn)
    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME, WRITES_COLLECTION_NAME, TTL
    ) as checkpointer:
        yield checkpointer
    client.close()


async def test_put_writes_on_interrupt(async_saver: MongoDBSaver) -> None:
    """Test that no error is raised when interrupted workflow updates writes."""
    config: RunnableConfig = {
        "configurable": {
            "checkpoint_id": "check1",
            "thread_id": "thread1",
            "checkpoint_ns": "",
        }
    }
    task_id = "task_id"
    task_path = "~__pregel_pull, human_feedback"

    writes1 = [
        (
            "__interrupt__",
            (
                Interrupt(
                    value="please provide input",
                ),
            ),
        )
    ]
    await async_saver.aput_writes(config, writes1, task_id, task_path)

    writes2 = [("__interrupt__", (Interrupt(value="please provide another input"),))]
    await async_saver.aput_writes(config, writes2, task_id, task_path)
