from __future__ import annotations

from importlib.metadata import version
from typing import Any, Generator, Iterable, Iterator, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo

from langchain_mongodb.utils import (
    make_serializable,
)

DEFAULT_INSERT_BATCH_SIZE = 100


class MongoDBDocStore(BaseStore):
    """MongoDB Collection providing BaseStore interface.

    This is meant to be treated as a key-value store: [str, Document]


    In a MongoDB Collection, the field name _id is reserved for use as a primary key.
    Its value must be unique in the collection, is immutable,
    and may be of any type other than an array or regex.
    As this field is always indexed, it is the natural choice to hold keys.

    The value will be held simply in a field called "value".
    It can contain any valid BSON type.

    Example key value pair: {"_id": "foo", "value": "bar"}.
    """

    def __init__(self, collection: Collection, text_key: str = "page_content") -> None:
        self.collection = collection
        self._text_key = text_key

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        **kwargs: Any,
    ) -> MongoDBDocStore:
        """Construct a Key-Value Store from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (in form f"{database}.{collection}")

        Returns:
            A new MongoDBDocStore instance.
        """
        client: MongoClient = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        )
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection=collection)

    def close(self) -> None:
        """Close the resources used by the MongoDBDocStore."""
        self.collection.database.client.close()

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        """Get the values associated with the given keys.

        If a key is not found in the store, the corresponding value will be None.
        As returning None is not the default find behavior, we form a dictionary
        and loop over the keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns: List of values associated with the given keys.
        """
        found_docs = {}
        for res in self.collection.find({"_id": {"$in": keys}}):
            text = res.pop(self._text_key)
            key = res.pop("_id")
            make_serializable(res)
            found_docs[key] = Document(page_content=text, metadata=res)
        return [found_docs.get(key, None) for key in keys]

    def mset(
        self,
        key_value_pairs: Sequence[tuple[str, Document]],
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
    ) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.
        """
        keys, docs = zip(*key_value_pairs)
        n_docs = len(docs)
        start = 0
        for end in range(batch_size, n_docs + batch_size, batch_size):
            texts, metadatas = zip(
                *[(doc.page_content, doc.metadata) for doc in docs[start:end]]
            )
            self.insert_many(texts=texts, metadatas=metadatas, ids=keys[start:end])  # type: ignore
            start = end

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        self.collection.delete_many({"_id": {"$in": keys}})

    def yield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[Iterator[str], Iterator[str]]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Yields:
            Iterator[str | str]: An iterator over keys that match the given prefix.
            This method is allowed to return an iterator over either str
            depending on what makes more sense for the given store.
        """
        query = {"_id": {"$regex": f"^{prefix}"}} if prefix else {}
        for document in self.collection.find(query, {"_id": 1}):
            yield document["_id"]

    def insert_many(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: List[str],
    ) -> None:
        """Bulk insert single batch of texts, embeddings, and optionally ids.

        insert_many in PyMongo does not overwrite existing documents.
        Instead, it attempts to insert each document as a new document.
        If a document with the same _id already exists in the collection,
        an error will be raised for that specific document. However, other documents
        in the batch that do not have conflicting _ids will still be inserted.
        """
        to_insert = [
            {"_id": i, self._text_key: t, **m} for i, t, m in zip(ids, texts, metadatas)
        ]
        self.collection.insert_many(to_insert)  # type: ignore
