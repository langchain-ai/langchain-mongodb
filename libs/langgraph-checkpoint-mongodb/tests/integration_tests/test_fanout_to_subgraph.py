"""

Based on LangGraph's Benchmarking script,
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/bench/fanout_to_subgraph.py

"""

import operator
import os
import random
import time
from collections.abc import Generator
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, Connection
from psycopg.rows import dict_row

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver, MongoDBSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START, Send
from langgraph.graph import END, StateGraph

# --- Configuration ---
MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
CHECKPOINT_CLXN_NAME = "fanout_checkpoints"
WRITES_CLXN_NAME = "fanout_writes"

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5441/"


@pytest.fixture(scope="function")
def checkpointer_memory() -> Generator[InMemorySaver, None, None]:
    yield InMemorySaver()


@pytest.fixture(scope="function")
def checkpointer_mongodb() -> Generator[MongoDBSaver, None, None]:
    with MongoDBSaver.from_conn_string(
        MONGODB_URI,
        db_name=DB_NAME,
        checkpoint_collection_name=CHECKPOINT_CLXN_NAME,
        writes_collection_name=WRITES_CLXN_NAME,
    ) as checkpointer:
        checkpointer.checkpoint_collection.delete_many({})
        checkpointer.writes_collection.delete_many({})
        yield checkpointer
        checkpointer.checkpoint_collection.drop()
        checkpointer.writes_collection.drop()


@pytest.fixture(scope="function")
async def checkpointer_mongodb_async() -> Generator[AsyncMongoDBSaver, None, None]:
    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI,
        db_name=DB_NAME,
        checkpoint_collection_name=CHECKPOINT_CLXN_NAME + "_async",
        writes_collection_name=WRITES_CLXN_NAME + "_async",
    ) as checkpointer:
        checkpointer.checkpoint_collection.delete_many({})
        checkpointer.writes_collection.delete_many({})
        yield checkpointer
        checkpointer.checkpoint_collection.drop()
        checkpointer.writes_collection.drop()


@pytest.fixture(scope="function")
def checkpointer_postgres() -> Generator[PostgresSaver, None, None]:
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with Connection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
async def checkpointer_postgres_async() -> Generator[AsyncPostgresSaver, None, None]:
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(autouse=True)
def disable_langsmith():
    """Disable LangSmith tracing for all tests"""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""


@pytest.fixture
def input():
    return {
        "subjects": [
            random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(1000)
        ]
    }


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


class JokeInput(TypedDict):
    subject: str


class JokeOutput(TypedDict):
    jokes: list[str]


class JokeState(JokeInput, JokeOutput): ...


def test_sync(
    input, checkpointer_memory, checkpointer_mongodb, checkpointer_postgres
) -> None:
    checkpointers = {
        "in_memory": checkpointer_memory,
        "mongodb": checkpointer_mongodb,
        "postgres": checkpointer_postgres,
    }

    def fanout_to_subgraph() -> StateGraph:
        def generate(state: JokeInput):
            return {"jokes": [f"Joke about {state['subject']}"]}

        def edit(state: JokeInput):
            subject = state["subject"]
            return {"subject": f"{subject} - hohoho"}

        def bump(state: JokeOutput):
            return {"jokes": [state["jokes"][0] + " a"]}

        def bump_loop(state: JokeOutput):
            return END if state["jokes"][0].endswith(" a" * 10) else "bump"

        def continue_to_jokes(state: OverallState):
            return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

        subgraph = StateGraph(JokeState, input=JokeInput, output=JokeOutput)
        subgraph.add_node("edit", edit)
        subgraph.add_node("generate", generate)
        subgraph.add_node("bump", bump)
        subgraph.set_entry_point("edit")
        subgraph.add_edge("edit", "generate")
        subgraph.add_edge("generate", "bump")
        subgraph.add_conditional_edges("bump", bump_loop)
        subgraph.set_finish_point("generate")
        subgraphc = subgraph.compile()

        # parent graph
        builder = StateGraph(OverallState)
        builder.add_node("generate_joke", subgraphc)
        builder.add_conditional_edges(START, continue_to_jokes)
        builder.add_edge("generate_joke", END)
        return builder

    print("\n\nBegin test_sync")
    for cname, checkpointer in checkpointers.items():
        assert isinstance(checkpointer, BaseCheckpointSaver)

        graph = fanout_to_subgraph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": cname}}
        start = time.time()
        len([c for c in graph.stream(input, config=config)])
        end = time.time()
        print(f"{cname}: {end - start:.4f} seconds")


async def test_async(
    input, checkpointer_memory, checkpointer_mongodb_async, checkpointer_postgres_async
) -> None:
    import bson

    print(f"{bson.has_c()=}")

    checkpointers = {
        "postgres_async": checkpointer_postgres_async,
        "in_memory_async": checkpointer_memory,
        "mongodb_async": checkpointer_mongodb_async,
    }

    async def fanout_to_subgraph() -> StateGraph:
        async def continue_to_jokes(state: OverallState):
            return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

        async def bump(state: JokeOutput):
            return {"jokes": [state["jokes"][0] + " a"]}

        async def generate(state: JokeInput):
            return {"jokes": [f"Joke about {state['subject']}"]}

        async def edit(state: JokeInput):
            subject = state["subject"]
            return {"subject": f"{subject} - hohoho"}

        async def bump_loop(state: JokeOutput):
            return END if state["jokes"][0].endswith(" a" * 10) else "bump"

        subgraph = StateGraph(JokeState, input=JokeInput, output=JokeOutput)
        subgraph.add_node("edit", edit)
        subgraph.add_node("generate", generate)
        subgraph.add_node("bump", bump)
        subgraph.set_entry_point("edit")
        subgraph.add_edge("edit", "generate")
        subgraph.add_edge("generate", "bump")
        subgraph.add_conditional_edges("bump", bump_loop)
        subgraph.set_finish_point("generate")
        subgraphc = subgraph.compile()

        # parent graph
        supergraph = StateGraph(OverallState)
        supergraph.add_node("generate_joke", subgraphc)
        supergraph.add_conditional_edges(START, continue_to_jokes)
        supergraph.add_edge("generate_joke", END)
        return supergraph

    print("\n\nBegin test_async")
    for cname, checkpointer in checkpointers.items():
        assert isinstance(checkpointer, BaseCheckpointSaver)

        graph = (await fanout_to_subgraph()).compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": cname}}
        start = time.time()
        len([c async for c in graph.astream(input, config=config)])
        end = time.time()
        print(f"{cname}: {end - start:.4f} seconds")
