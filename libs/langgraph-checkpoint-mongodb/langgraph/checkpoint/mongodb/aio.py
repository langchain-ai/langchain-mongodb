import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import (
    Any,
    Optional,
    cast,
)

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import ASCENDING, AsyncMongoClient, UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from .utils import (
    DRIVER_METADATA,
    _append_client_metadata,
    _validate_filter,
    dumps_metadata,
    loads_metadata,
)

__all__ = ["AsyncMongoDBSaver"]


async def _create_saver_indexes(
    collection: AsyncCollection,
    compound_index: list[tuple[str, int]],
    ttl: Optional[int] = None,
) -> None:
    """Create indexes for the saver collections.

    This helper function creates the given compound index and TTL index (if required)
    for the given collection.

    Args:
        collection (AsyncCollection): The MongoDB collection to create indexes on.
        compound_index (list[tuple[str, int]]): The compound index to create.
        ttl (int, optional): Time to live in seconds for the TTL index. Defaults to None.
    """

    def index_key_list(index: Any) -> list[tuple[str, int]]:
        return list((k, v) for k, v in index["key"].items())

    indexes = await (await collection.list_indexes()).to_list()
    index_keys = [index_key_list(idx) for idx in indexes]
    if compound_index not in index_keys:
        await collection.create_index(compound_index, unique=True)
    if ttl is not None:
        ttl_index = [("created_at", ASCENDING)]
        found = False
        for idx in indexes:
            if (
                index_key_list(idx) == tuple(ttl_index)
                and idx.get("expireAfterSeconds") == ttl
            ):
                found = True
                break
        if not found:
            await collection.create_index(ttl_index, expireAfterSeconds=ttl)


class AsyncMongoDBSaver(BaseCheckpointSaver):
    """A checkpointer that stores StateGraph checkpoints in MongoDB asynchronously.

    This is the asynchronous counterpart of
    [class:~langgraph.checkpoint.mongodb.MongoDBSaver]. It talks to MongoDB through
    an [class:~pymongo.AsyncMongoClient], so no thread is blocked on database I/O,
    and an application that already owns an asynchronous client can pass it in
    rather than opening a second connection pool.

    A compound index as shown below will be added to each of the collections
    backing the saver (checkpoints, pending writes). If the collections pre-exist,
    and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True,

    Indexes cannot be created in ``__init__``, as that is not a coroutine. They are
    created on first use instead, or eagerly by awaiting
    [meth:~langgraph.checkpoint.mongodb.aio.AsyncMongoDBSaver.setup].

    The synchronous methods are provided for compatibility with the synchronous
    LangGraph API, and may only be called from a thread other than the one running
    this saver's event loop. From the event loop itself, use the async interface.

    Args:
        client (AsyncMongoClient): The MongoDB connection.
        db_name (Optional[str]): Database name
        checkpoint_collection_name (Optional[str]): Name of Collection of Checkpoints
        writes_collection_name (Optional[str]): Name of Collection of intermediate writes.
        ttl (Optional[int]): Time to live in seconds. See https://www.mongodb.com/docs/manual/core/index-ttl/.

    Examples:

        >>> import asyncio
        >>> from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> async def main() -> None:
        ...     builder = StateGraph(int)
        ...     builder.add_node("add_one", lambda x: x + 1)
        ...     builder.set_entry_point("add_one")
        ...     builder.set_finish_point("add_one")
        ...     async with AsyncMongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
        ...         graph = builder.compile(checkpointer=memory)
        ...         config = {"configurable": {"thread_id": "1"}}
        ...         print(await graph.ainvoke(3, config))
        >>>
        >>> asyncio.run(main())
        4

        Passing in a client the application already owns:

        >>> from pymongo import AsyncMongoClient
        >>> client = AsyncMongoClient("mongodb://localhost:27017")
        >>> memory = AsyncMongoDBSaver(client)
    """

    client: AsyncMongoClient
    db: AsyncDatabase

    def __init__(
        self,
        client: AsyncMongoClient,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints_aio",
        writes_collection_name: str = "checkpoint_writes_aio",
        ttl: Optional[int] = None,
        serde: SerializerProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[db_name]
        self.checkpoint_collection = self.db[checkpoint_collection_name]
        self.writes_collection = self.db[writes_collection_name]
        self.ttl = ttl
        if serde is not None:
            self.serde = serde
        else:
            self.serde = JsonPlusSerializer()
        self.loop = asyncio.get_running_loop()
        self.setup_lock = asyncio.Lock()
        self.is_setup = False

        _append_client_metadata(self.client)

    async def setup(self) -> None:
        """Create the collections' indexes if they are not present, at most once."""
        if self.is_setup:
            return
        # The lock leaves a failed attempt retryable, rather than caching it as done.
        async with self.setup_lock:
            if self.is_setup:
                return
            await _create_saver_indexes(
                self.checkpoint_collection,
                [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
                self.ttl,
            )
            await _create_saver_indexes(
                self.writes_collection,
                [
                    ("thread_id", 1),
                    ("checkpoint_ns", 1),
                    ("checkpoint_id", -1),
                    ("task_id", 1),
                    ("idx", 1),
                ],
                self.ttl,
            )
            self.is_setup = True

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints_aio",
        writes_collection_name: str = "checkpoint_writes_aio",
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator["AsyncMongoDBSaver"]:
        """Asynchronous context manager to create a MongoDB checkpoint saver.

        A compound index as shown below will be added to each of the collections
        backing the saver (checkpoints, pending writes). If the collections pre-exist,
        and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True

        Args:
            conn_string: MongoDB connection string. See [class:~pymongo.AsyncMongoClient].
            db_name: Database name. It will be created if it doesn't exist.
            checkpoint_collection_name: Checkpoint Collection name. Created if it doesn't exist.
            writes_collection_name: Collection name of intermediate writes. Created if it doesn't exist.
            ttl (Optional[int]): Time to live in seconds.
        Yields: A new AsyncMongoDBSaver.
        """
        client: Optional[AsyncMongoClient] = None
        try:
            client = AsyncMongoClient(
                conn_string,
                driver=DRIVER_METADATA,
            )
            saver = AsyncMongoDBSaver(
                client,
                db_name,
                checkpoint_collection_name,
                writes_collection_name,
                ttl,
                **kwargs,
            )
            await saver.setup()
            yield saver
        finally:
            if client:
                await client.close()

    async def close(self) -> None:
        """Close the resources used by the AsyncMongoDBSaver."""
        await self.client.close()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

         This method retrieves a checkpoint tuple from the MongoDB database based on the
         provided config. If the config contains a "checkpoint_id" key, the checkpoint with
         the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
         for the given thread ID is retrieved.

         Args:
             config (RunnableConfig): The config to use for retrieving the checkpoint.

         Returns:
             Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

             Basic:
             >>> config = {"configurable": {"thread_id": "1"}}
             >>> checkpoint_tuple = await memory.aget_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)

             With checkpoint ID:
             >>> config = {
             ...    "configurable": {
             ...        "thread_id": "1",
             ...        "checkpoint_ns": "",
             ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
             ...    }
             ... }
             >>> checkpoint_tuple = await memory.aget_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.checkpoint_collection.find(
            query, sort=[("checkpoint_id", -1)], limit=1
        )
        async for doc in result:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = self.writes_collection.find(config_values)
            pending_writes = [
                (
                    wrt["task_id"],
                    wrt["channel"],
                    self.serde.loads_typed((wrt["type"], wrt["value"])),
                )
                async for wrt in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                loads_metadata(self.serde, doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )
        return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the MongoDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An iterator of checkpoint tuples.

            Examples:
            >>> from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
            >>> async with AsyncMongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            ... # Run a graph, then list the checkpoints
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = [c async for c in memory.alist(config, limit=2)]
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]
        """
        await self.setup()
        query = {}
        if config is not None:
            if "thread_id" in config["configurable"]:
                query["thread_id"] = config["configurable"]["thread_id"]
            if "checkpoint_ns" in config["configurable"]:
                query["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]

        if filter:
            _validate_filter(filter)
            for key, value in filter.items():
                query[f"metadata.{key}"] = dumps_metadata(self.serde, value)

        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        result = self.checkpoint_collection.find(
            query, limit=0 if limit is None else limit, sort=[("checkpoint_id", -1)]
        )

        async for doc in result:
            config_values = {
                "thread_id": doc["thread_id"],
                "checkpoint_ns": doc["checkpoint_ns"],
                "checkpoint_id": doc["checkpoint_id"],
            }
            serialized_writes = self.writes_collection.find(config_values)
            pending_writes = [
                (
                    wrt["task_id"],
                    wrt["channel"],
                    self.serde.loads_typed((wrt["type"], wrt["value"])),
                )
                async for wrt in serialized_writes
            ]

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads_typed((doc["type"], doc["checkpoint"])),
                metadata=loads_metadata(self.serde, doc["metadata"]),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes=pending_writes,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the MongoDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
            >>> async with AsyncMongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "data": {"key": "value"}}
            >>>     saved_config = await memory.aput(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        metadata = get_checkpoint_metadata(config, metadata)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": dumps_metadata(self.serde, metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        if self.ttl:
            doc["created_at"] = datetime.now(tz=UTC)

        await self.checkpoint_collection.update_one(
            upsert_query, {"$set": doc}, upsert=True
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the MongoDB database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        set_method = (  # Allow replacement on existing writes only if there were errors.
            "$set" if all(w[0] in WRITES_IDX_MAP for w in writes) else "$setOnInsert"
        )
        operations = []
        now = datetime.now(tz=UTC)
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
            }

            type_, serialized_value = self.serde.dumps_typed(value)

            update_doc: dict[str, Any] = {
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }

            if self.ttl:
                update_doc["created_at"] = now

            operations.append(
                UpdateOne(
                    filter=upsert_query,
                    update={set_method: update_doc},
                    upsert=True,
                )
            )
        await self.writes_collection.bulk_write(operations)

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id (str): The thread ID whose checkpoints should be deleted.
        """
        await self.setup()
        # Delete all checkpoints associated with the thread ID
        await self.checkpoint_collection.delete_many({"thread_id": thread_id})

        # Delete all writes associated with the thread ID
        await self.writes_collection.delete_many({"thread_id": thread_id})

    def _check_thread(self, method: str) -> None:
        """Refuse a blocking call made from the event loop that would deadlock it."""
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncMongoDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    f"For example, use `await checkpointer.{method}(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method blocks on [meth:AsyncMongoDBSaver.aget_tuple], and so may only be
        called from a thread other than the one running this saver's event loop.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        self._check_thread("aget_tuple")
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method blocks on [meth:AsyncMongoDBSaver.alist], and so may only be
        called from a thread other than the one running this saver's event loop.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        self._check_thread("alist")
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    cast(Any, anext(aiter_)),
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method blocks on [meth:AsyncMongoDBSaver.aput], and so may only be
        called from a thread other than the one running this saver's event loop.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        self._check_thread("aput")
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method blocks on [meth:AsyncMongoDBSaver.aput_writes], and so may only be
        called from a thread other than the one running this saver's event loop.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        self._check_thread("aput_writes")
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        This method blocks on [meth:AsyncMongoDBSaver.adelete_thread], and so may only
        be called from a thread other than the one running this saver's event loop.

        Args:
            thread_id (str): The thread ID whose checkpoints should be deleted.
        """
        self._check_thread("adelete_thread")
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()
