"""Follows https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel"""

import os
from collections.abc import Generator
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

# --- Configuration ---
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
CHECKPOINT_CLXN_NAME = "interrupts_checkpoints"
WRITES_CLXN_NAME = "interrupts_writes"


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


ALL_CHECKPOINTERS_SYNC = [
    "checkpointer_memory",
    "checkpointer_mongodb",
]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(checkpointer_name)
    assert isinstance(checkpointer, BaseCheckpointSaver)

    # --- State Definition ---
    class State(TypedDict):
        value: int
        step: int

    # --- Node Definitions ---
    def node_inc(state: State) -> State:
        """Increments value and step by 1"""
        current_step = state.get("step", 0)
        return {"value": state["value"] + 1, "step": current_step + 1}

    def node_double(state: State) -> State:
        """Doubles value and increments step by 1"""
        current_step = state.get("step", 0)
        return {"value": state["value"] * 2, "step": current_step + 1}

    # --- Graph Construction ---
    builder = StateGraph(State)
    builder.add_node("increment", node_inc)
    builder.add_node("double", node_double)
    builder.set_entry_point("increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)

    # --- Compile Graph (with Interruption) ---
    # Using sync for simplicity in this demo
    graph: CompiledGraph = builder.compile(
        checkpointer=checkpointer, interrupt_after=["increment"]
    )

    # --- Configure  ---
    config: RunnableConfig = {"configurable": {"thread_id": "thread_#1"}}
    initial_input = {"value": 10, "step": 0}

    # --- 1st invoke, with Interruption
    interrupted_state = graph.invoke(initial_input, config=config)
    assert interrupted_state == {"value": 10 + 1, "step": 1}
    state_history = list(graph.get_state_history(config))
    assert len(state_history) == 3
    # The states are returned in reverse chronological order.
    assert state_history[0].next == ("double",)

    # --- 2nd invoke, with input=None, and original config ==> continues from point of interruption
    final_state = graph.invoke(None, config=config)
    assert final_state == {"value": (10 + 1) * 2, "step": 2}
    state_history = list(graph.get_state_history(config))
    assert len(state_history) == 4
    assert state_history[0].next == ()
    assert state_history[-1].next == ("__start__",)

    # --- 3rd invoke, but with an input ===> the CompiledGraph is restarted.
    new_input = {"value": 100, "step": -100}
    third_state = graph.invoke(new_input, config=config)
    assert third_state == {"value": 101, "step": -99}

    # The entire state history is preserved however
    state_history = list(graph.get_state_history(config))
    assert len(state_history) == 7
    assert state_history[0].next == ("double",)
    assert state_history[2].next == ("__start__",)

    # --- Upstate state and continue from interrupt
    updated_state = {"value": 1000, "step": 1000}
    updated_config = graph.update_state(config, updated_state)
    final_state = graph.invoke(input=None, config=updated_config)
    assert final_state == {"value": 2000, "step": 1001}
