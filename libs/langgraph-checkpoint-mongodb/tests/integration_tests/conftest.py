from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Generator, Optional
from uuid import UUID

import pytest
from langchain_core import __version__ as core_version
from packaging import version
from pytest_mock import MockerFixture

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.store.memory import InMemoryStore

pytest.register_assert_rewrite("tests.memory_assert")

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5442/"
# TODO: fix this once core is released
IS_LANGCHAIN_CORE_030_OR_GREATER = version.parse(core_version) >= version.parse(
    "0.3.0.dev0"
)
SHOULD_CHECK_SNAPSHOTS = IS_LANGCHAIN_CORE_030_OR_GREATER


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


# checkpointer fixtures


@pytest.fixture(scope="function")
def checkpointer_memory() -> Generator[BaseCheckpointSaver, None]:
    from .memory_assert import MemorySaverAssertImmutable

    yield MemorySaverAssertImmutable()


@pytest.fixture
def checkpointer_mongodb() -> Generator[BaseCheckpointSaver, None]:
    """Fresh checkpointer without any memories."""
    with MongoDBSaver.from_conn_string(
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        os.environ.get("DATABASE_NAME", "langchain_checkpoints_db"),
    ) as checkpointer:
        checkpointer.checkpoint_collection.delete_many({})
        checkpointer.writes_collection.delete_many({})
        yield checkpointer


@asynccontextmanager
async def _checkpointer_mongodb_aio() -> AsyncGenerator[AsyncMongoDBSaver, None]:
    async with AsyncMongoDBSaver.from_conn_string(
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        os.environ.get("DATABASE_NAME", "langchain_checkpoints_db"),
    ) as checkpointer:
        await checkpointer.checkpoint_collection.delete_many({})
        await checkpointer.writes_collection.delete_many({})
        yield checkpointer


@asynccontextmanager
async def awith_checkpointer(
    checkpointer_name: Optional[str],
) -> Any:
    if checkpointer_name is None:
        yield None
    elif checkpointer_name == "memory":
        from .memory_assert import MemorySaverAssertImmutable

        yield MemorySaverAssertImmutable()
    elif checkpointer_name == "mongodb_aio":
        async with _checkpointer_mongodb_aio() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


@pytest.fixture(scope="function")
def store_in_memory() -> Generator[InMemoryStore]:
    yield InMemoryStore()


@asynccontextmanager
async def awith_store(store_name: Optional[str]) -> Any:
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        yield InMemoryStore()
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


ALL_CHECKPOINTERS_SYNC = [
    # "memory",
    "mongodb",
]
ALL_CHECKPOINTERS_ASYNC = [
    # "memory",
    "mongodb_aio",
]

ALL_CHECKPOINTERS_ASYNC_PLUS_NONE = [
    *ALL_CHECKPOINTERS_ASYNC,
    None,
]
ALL_STORES_SYNC = ["in_memory"]
ALL_STORES_ASYNC = ["in_memory"]
