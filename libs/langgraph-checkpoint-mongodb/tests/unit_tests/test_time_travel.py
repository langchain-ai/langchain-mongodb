import os
from collections.abc import Generator
from operator import add
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from pymongo import MongoClient
from typing_extensions import NotRequired

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StateSnapshot

# Test configuration
MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true"
)


class ExpenseState(TypedDict):
    amount: NotRequired[int]
    version: NotRequired[int]
    approved: NotRequired[bool]
    messages: Annotated[list[str], add]


def add_expense_node(state: ExpenseState) -> dict[str, Any]:
    """Node adds expense and a message"""
    return dict(amount=100, version=1, approved=False, messages=["Added new expense"])


def validate_expense_node(state: ExpenseState) -> dict[str, Any]:
    """Node that processes data based on current state"""
    if state.get("amount") == 200:
        return dict(approved=True, messages=["expense approved"])
    else:
        return dict(approved=False, messages=["expense denied"])


@pytest.fixture(
    params=[None, 60 * 60],
    ids=["ttl_none", "ttl_3600"],
)
def checkpointer(request: Any) -> Generator[MongoDBSaver]:
    db_name = "langgraph-test"
    checkpoint_collection_name = "timetravel_checkpoints"
    writes_collection_name = "timetravel_writes"

    # Initialize MongoDB checkpointer
    client: MongoClient = MongoClient(MONGODB_URI)

    saver = MongoDBSaver(
        client=client,
        db_name=db_name,
        collection_name=checkpoint_collection_name,
        WRITES_COLLECTION_NAME=writes_collection_name,
        ttl=request.param,
    )

    # Can use this to compare
    # saver = InMemorySaver()

    yield saver

    client[db_name].drop_collection(checkpoint_collection_name)
    client[db_name].drop_collection(writes_collection_name)
    client.close()


def test(checkpointer: MongoDBSaver) -> None:
    """Test ability to use checkpointer to update exact state of graph.

    In this simple example, we assume an initial state has been set incorrectly.
    To fix this, instead of rerunning from start,
    we find the incorrect node, update_state, and continue (by passing None to invoke or stream).

    This example does not use interrupt/resume as one might, for example,
    in an expense report approval workflow.
    """
    initial_state: ExpenseState = dict(
        amount=0, version=0, approved=False, messages=["Initial state"]
    )
    config: RunnableConfig = dict(configurable=dict(thread_id="test-time-travel"))

    # Create the graph, which should be a 2-step procedure
    workflow = StateGraph(ExpenseState)
    workflow.add_node("add_expense", add_expense_node)
    workflow.add_node("validate_expense", validate_expense_node)
    workflow.add_edge(START, "add_expense")
    workflow.add_edge("validate_expense", END)
    workflow.add_edge("add_expense", "validate_expense")
    graph = workflow.compile(checkpointer=checkpointer)

    # Run the graph
    graph.invoke(input=initial_state, config=config)  # type:ignore[arg-type]

    # Check to see whether the final state is approved
    final_state = graph.get_state(config=config)

    # It is not approved.
    assert not final_state.values["approved"]

    # Let's use time-travel to find the checkpoint before "add_expense"
    checkpoints: list[StateSnapshot] = list(graph.get_state_history(config))
    # checkpoints: list[CheckpointTuple] = list(checkpointer.list(config))
    print(f"\nFound {len(checkpoints)} checkpoints")

    target_checkpoint = None
    for checkpoint in checkpoints:
        # Look for checkpoint after increment but before final processing
        if (
            checkpoint.metadata and checkpoint.metadata.get("step") == 1
        ):  # Before validate node
            target_checkpoint = checkpoint
            break

    for state in checkpoints:
        if state.metadata:
            print(f"\nstep: {state.metadata['step']}")
        print(f"next: {state.next}")
        print(f"checkpoint_id: {state.config['configurable']['checkpoint_id']}")
        print(f"values: {state.values}")

    # Get state at that checkpoint
    assert target_checkpoint
    past_state = graph.get_state(target_checkpoint.config)

    # Update the expense amount to 200 that validate amounts
    updated_state = dict(**past_state.values)
    # updated_state = {}
    updated_state["amount"] = 200
    updated_state["version"] = 2
    updated_state["messages"] += ["Updated state"]

    updated_config = graph.update_state(
        config=target_checkpoint.config, values=updated_state
    )

    # Continue from the checkpoint
    print("\nContinuing execution with stream(None, config)...")
    final_step = None
    for step in graph.stream(None, updated_config):
        print(f"Continuation step: {step}")
        final_step = step

    # Verify the final result
    assert isinstance(final_step, dict)
    assert final_step["validate_expense"]["approved"]
    # Note that all values are not in the final step
    assert "amount" not in final_step["validate_expense"]
    # They ARE available from graph.get_state
    final_state = graph.get_state(updated_config)
    assert final_state.values["amount"] == 200
    assert set(final_state.values.keys()) == {
        "amount",
        "version",
        "messages",
        "approved",
    }
