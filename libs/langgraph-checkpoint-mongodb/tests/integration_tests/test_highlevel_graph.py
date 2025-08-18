"""
Based on LangGraph's Benchmarking script,
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/bench/fanout_to_subgraph.py,
this pattern of joke generation is used often in the examples.
The fanout here is performed by the list comprehension of [class:~langgraph.types.Send] calls.
The effect of this is a map (fanout) workflow where the graph invokes
the same node multiple times in parallel.
The node here is a subgraph.
The subgraph is linear, with a conditional edge 'bump_loop' that repeatably calls
the node 'bump' until a condition is met.
This test can be used for benchmarking.
It also demonstrates the high-level API of subgraphs, add_conditional_edges, and Send.
"""

import operator
import os
import time
from collections.abc import AsyncGenerator, Generator
from typing import Annotated

import pytest
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver, MongoDBSaver
from langgraph.constants import START, Send
from langgraph.graph import END, StateGraph

# --- Configuration ---
MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
CHECKPOINT_CLXN_NAME = "fanout_checkpoints"
WRITES_CLXN_NAME = "fanout_writes"

N_SUBJECTS = 10  # increase for benchmarking


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


class JokeInput(TypedDict):
    subject: str


class JokeOutput(TypedDict):
    jokes: list[str]


class JokeState(JokeInput, JokeOutput): ...


def fanout_to_subgraph() -> StateGraph:
    # Subgraph nodes create a joke.
    def edit(state: JokeOutput) -> JokeOutput:
        return {"jokes": [f"{state['jokes'][0]}... and cats!"]}

    def generate(state: JokeInput) -> JokeOutput:
        return {"jokes": [f"Joke about the year {state['subject']}"]}

    def bump(state: JokeOutput) -> dict[str, list[str]]:
        return {"jokes": [state["jokes"][0] + " and the year before"]}

    def bump_loop(state: JokeOutput) -> JokeOutput:
        return (
            "edit" if state["jokes"][0].endswith(" and the year before" * 3) else "bump"
        )

    subgraph = StateGraph(JokeState)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_node("bump_loop", bump_loop)
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("edit")
    subgraphc = subgraph.compile()

    # Parent graph maps the joke-generating subgraph.
    def fanout(state: OverallState) -> list:
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    parentgraph = StateGraph(OverallState)
    parentgraph.add_node("generate_joke", subgraphc)  # type: ignore[arg-type]
    parentgraph.add_conditional_edges(START, fanout)
    parentgraph.add_edge("generate_joke", END)
    return parentgraph


@pytest.fixture
def joke_subjects() -> OverallState:
    years = [str(2025 - 10 * i) for i in range(N_SUBJECTS)]
    return {"subjects": years}


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
async def checkpointer_mongodb_async() -> AsyncGenerator[AsyncMongoDBSaver, None]:
    async with AsyncMongoDBSaver.from_conn_string(
        MONGODB_URI,
        db_name=DB_NAME,
        checkpoint_collection_name=CHECKPOINT_CLXN_NAME + "_async",
        writes_collection_name=WRITES_CLXN_NAME + "_async",
    ) as checkpointer:
        await checkpointer.checkpoint_collection.delete_many({})
        await checkpointer.writes_collection.delete_many({})
        yield checkpointer
        await checkpointer.checkpoint_collection.drop()
        await checkpointer.writes_collection.drop()


@pytest.fixture(autouse=True)
def disable_langsmith() -> None:
    """Disable LangSmith tracing for all tests"""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""


async def test_fanout(
    joke_subjects: OverallState,
    checkpointer_mongodb: MongoDBSaver,
    checkpointer_mongodb_async: AsyncMongoDBSaver,
    checkpointer_memory: InMemorySaver,
) -> None:
    checkpointers = {
        "mongodb": checkpointer_mongodb,
        "mongodb_async": checkpointer_mongodb_async,
        "in_memory": checkpointer_memory,
        "in_memory_async": checkpointer_memory,
    }

    for cname, checkpointer in checkpointers.items():
        assert isinstance(checkpointer, BaseCheckpointSaver)
        print(f"\n\nBegin test of {cname}")
        graphc = (fanout_to_subgraph()).compile(checkpointer=checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": cname}}
        start = time.monotonic()
        if "async" in cname:
            out = [c async for c in graphc.astream(joke_subjects, config=config)]  # type: ignore[arg-type]
        else:
            out = [c for c in graphc.stream(joke_subjects, config=config)]  # type: ignore[arg-type]
        assert len(out) == N_SUBJECTS
        assert isinstance(out[0], dict)
        assert out[0].keys() == {"generate_joke"}
        assert set(out[0]["generate_joke"].keys()) == {"jokes"}
        assert all(
            res["generate_joke"]["jokes"][0].endswith(
                f'{" and the year before" * 3}... and cats!'
            )
            for res in out
        )
        end = time.monotonic()
        print(f"{cname}: {end - start:.4f} seconds")
