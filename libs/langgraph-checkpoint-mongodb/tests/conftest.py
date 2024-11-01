import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

import pytest
from langchain_core import __version__ as core_version
from packaging import version
from pytest_mock import MockerFixture

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .memory_assert import MemorySaverAssertImmutable

IS_LANGCHAIN_CORE_030_OR_GREATER = version.parse(core_version) >= version.parse(
    "0.3.0.dev0"
)
SHOULD_CHECK_SNAPSHOTS = IS_LANGCHAIN_CORE_030_OR_GREATER


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


# checkpointer fixtures


@pytest.fixture(scope="function")
def checkpointer_memory():
    yield MemorySaverAssertImmutable()


@pytest.fixture(scope="function")
def checkpointer_mongodb():
    with MongoDBSaver.from_conn_string("mongodb://localhost:27017") as checkpointer:
        # TODO This was where strange path-dependent behavior showed up.
        #  Whether we dropped the db, or restarted the container.
        #   Something funny going on with run_manager's uuids?
        # checkpointer.client.drop_database("checkpointing_db")
        yield checkpointer


@asynccontextmanager
async def _checkpointer_mongodb_aio():
    async with AsyncMongoDBSaver.from_conn_string(
        "mongodb://localhost:27017"
    ) as checkpointer:
        checkpointer.client.drop_database("checkpointing_db")
        yield checkpointer


@asynccontextmanager
async def awith_checkpointer(
    checkpointer_name: Optional[str],
) -> AsyncIterator[BaseCheckpointSaver]:
    if checkpointer_name is None:
        yield None
    elif checkpointer_name == "memory":
        yield MemorySaverAssertImmutable()
    elif checkpointer_name == "mongodb_aio":
        async with _checkpointer_mongodb_aio() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


# TODO - DO WE NEED TO ADD A MongoDBStore? (see PostgresStore)


@pytest.fixture(scope="function")
def store_in_memory():
    yield InMemoryStore()


@asynccontextmanager
async def awith_store(store_name: Optional[str]) -> AsyncIterator[BaseStore]:
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        yield InMemoryStore()
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


ALL_CHECKPOINTERS_SYNC = [
    "memory",
    "mongodb",
    # "sqlite",
    # "postgres",
    # "postgres_pipe",
    # "postgres_pool",
]
ALL_CHECKPOINTERS_ASYNC = [
    "memory",
    "mongodb_aio",
    # "sqlite_aio",
    # "postgres_aio",
    # "postgres_aio_pipe",
    # "postgres_aio_pool",
]
ALL_CHECKPOINTERS_ASYNC_PLUS_NONE = [
    *ALL_CHECKPOINTERS_ASYNC,
    None,
]
ALL_STORES_SYNC = ["in_memory"]
ALL_STORES_ASYNC = ["in_memory"]
