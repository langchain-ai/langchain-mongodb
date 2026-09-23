# LangGraph Checkpoint MongoDB

Implementation of LangGraph CheckpointSaver that uses MongoDB.

# Installation

```bash
pip install -U langgraph-checkpoint-mongodb
```

## Usage

For more detailed usage examples and documentation, please refer to the [MongoDB LangGraph documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/).

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "checkpoint_example"

with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Async methods of `MongoDBSaver`

The synchronous saver exposes async methods that run the blocking driver in a thread executor.

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "checkpoint_example"

with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```

### `AsyncMongoDBSaver`

For applications that are asynchronous end to end, `AsyncMongoDBSaver` uses pymongo's
`AsyncMongoClient` directly, so no thread is blocked on database I/O.

```python
import asyncio

from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langgraph.graph import StateGraph

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "checkpoint_example"


async def main() -> None:
    builder = StateGraph(int)
    builder.add_node("add_one", lambda x: x + 1)
    builder.set_entry_point("add_one")
    builder.set_finish_point("add_one")

    async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "1"}}

        print(await graph.ainvoke(3, config))
        print([c async for c in checkpointer.alist(config)])


asyncio.run(main())
```

The saver can also be given a client the application already owns. Its indexes are then
created on first use, or eagerly by awaiting `setup()`:

```python
from pymongo import AsyncMongoClient

from langgraph.checkpoint.mongodb import AsyncMongoDBSaver

client = AsyncMongoClient(MONGODB_URI)
checkpointer = AsyncMongoDBSaver(client, DB_NAME)
await checkpointer.setup()
```

`AsyncMongoDBSaver` writes to the `checkpoints_aio` and `checkpoint_writes_aio` collections
by default, so it does not share collections with `MongoDBSaver` unless you name them.

Its synchronous methods (`get_tuple`, `list`, `put`, `put_writes`, `delete_thread`) block on
their async counterparts, and so may only be called from a thread other than the one running
the saver's event loop — calling them from the loop itself raises `asyncio.InvalidStateError`.
Where the entry point is synchronous, use `MongoDBSaver` instead.
