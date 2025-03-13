# Based on https://github.com/langchain-ai/langchain/blob/8f5e72de057bc07df19f7d7aefb7673b64fbb1b4/libs/community/langchain_community/indexes/_document_manager.py#L58
from __future__ import annotations

import functools
import warnings
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.indexing.base import RecordManager
from langchain_core.runnables.config import run_in_executor
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import OperationFailure


class MongoDBRecordManager(RecordManager):
    """A MongoDB-based implementation of the record manager."""

    def __init__(self, collection: Collection) -> None:
        """Initialize the MongoDBRecordManager.

        The record manager abstraction is used by the langchain indexing API.
        The record manager keeps track of which documents have been written into a vectorstore and when they were written.
        For more details, see the `RecordManager API Docs`_.

        Args:
            connection_string: A valid MongoDB connection URI.
            db_name: The name of the database to use.
            collection_name: The name of the collection to use.

        .. _RecordManager API Docs:
            https://python.langchain.com/api_reference/core/indexing/langchain_core.indexing.base.RecordManager.html
        """
        namespace = f"{collection.database.name}.{collection.name}"
        super().__init__(namespace=namespace)
        self._collection = collection

    @classmethod
    def from_connection_string(
        cls, connection_string: str, namespace: str
    ) -> MongoDBRecordManager:
        """Construct a RecordManager from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (in form f"{database}.{collection}")

        Returns:
            A new MongoDBRecordManager instance.
        """
        client: MongoClient = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        )
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection=collection)

    def close(self) -> None:
        """Close the resources used by the MongoDBRecordManager."""
        self._collection.database.client.close()

    def create_schema(self) -> None:
        """Create the database schema for the document manager."""
        pass

    async def acreate_schema(self) -> None:
        """Create the database schema for the document manager."""
        pass

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert documents into the MongoDB collection."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError("Number of keys does not match number of group_ids")

        for key, group_id in zip(keys, group_ids):
            self._collection.find_one_and_update(
                {"namespace": self.namespace, "key": key},
                {"$set": {"group_id": group_id, "updated_at": self.get_time()}},
                upsert=True,
            )

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Asynchronously upsert documents into the MongoDB collection."""
        func = functools.partial(
            self.update, keys, group_ids=group_ids, time_at_least=time_at_least
        )
        return await run_in_executor(None, func)

    def get_time(self) -> float:
        """Get the current server time as a timestamp."""
        try:
            server_info = self._collection.database.command("hostInfo")
            local_time = server_info["system"]["currentTime"]
            timestamp = local_time.timestamp()
        except OperationFailure:
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    "Could not get high-resolution timestamp, falling back to low-resolution",
                    stacklevel=2,
                )
            ping = self._collection.database.command("ping")
            local_time = ping["operationTime"]
            timestamp = float(local_time.time)
        return timestamp

    async def aget_time(self) -> float:
        """Asynchronously get the current server time as a timestamp."""
        func = functools.partial(self.get_time)
        return await run_in_executor(None, func)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the MongoDB collection."""
        existing_keys = {
            doc["key"]
            for doc in self._collection.find(
                {"namespace": self.namespace, "key": {"$in": keys}}, {"key": 1}
            )
        }
        return [key in existing_keys for key in keys]

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Asynchronously check if the given keys exist in the MongoDB collection."""
        func = functools.partial(self.exists, keys)
        return await run_in_executor(None, func)

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List documents in the MongoDB collection based on the provided date range."""
        query: Dict[str, Any] = {"namespace": self.namespace}
        if before:
            query["updated_at"] = {"$lt": before}
        if after:
            query["updated_at"] = {"$gt": after}
        if group_ids:
            query["group_id"] = {"$in": group_ids}

        cursor = (
            self._collection.find(query, {"key": 1}).limit(limit)
            if limit
            else self._collection.find(query, {"key": 1})
        )
        return [doc["key"] for doc in cursor]

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Asynchronously list documents in the MongoDB collection
        based on the provided date range.
        """
        func = functools.partial(
            self.list_keys, before=before, after=after, group_ids=group_ids, limit=limit
        )
        return await run_in_executor(None, func)

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete documents from the MongoDB collection."""
        self._collection.delete_many(
            {"namespace": self.namespace, "key": {"$in": keys}}
        )

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Asynchronously delete documents from the MongoDB collection."""
        func = functools.partial(self.delete_keys, keys)
        return await run_in_executor(None, func)
