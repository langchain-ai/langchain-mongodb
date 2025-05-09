import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Literal, Optional, Union

from bson import SON
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_mongodb.index import create_vector_search_index
from langchain_mongodb.pipelines import vector_search_stage
from pymongo import (
    DeleteOne,
    MongoClient,
    UpdateOne,
)
from pymongo.collection import Collection, ReturnDocument
from pymongo.driver_info import DriverInfo

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.embed import (
    AEmbeddingsFunc,
    EmbeddingsFunc,
    ensure_embeddings,
    get_text_at_path,
)

logger = logging.getLogger(__name__)


class VectorIndexConfig(IndexConfig, total=False):
    """Configuration for a MongoDB Atlas Vector Index providing semantic search.


    Use the factory function, ~langgraph.store.mongodb.create_vector_index_config
    for convenient creation and sensible defaults.

    Unlike PostgreSQL, MongoDB does not require a separate package or vector store.
    Embeddings are stored and indexed within the collection that holds the data.
    For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview

    Vector Search can be Approximate Nearest Neighbor (ANN) or Exact (ENN).
    For ANN, Atlas uses HNSW (Hierarchical Navigable Small World).
    """

    name: str
    """Name of the index attached to the Collection in the Atlas Database."""

    relevance_score_fn: Literal["euclidean", "cosine", "dotProduct"]
    """Similarity scoring function used to compare vectors."""

    embedding_key: str
    """This key will contain the embedding vector for the value in fields[0].

    MongoDB does not require a separate Vector Store.
    It is designed to have one vector per document.
    """

    filters: list[str]
    """List of fields to that can filtered during search.


    Fields must be included in the index if they are to be filtered upon.
    The `namespace` field will always be included.
    These will be added to the vector search index automatically if not present.

    Fields can refer to top-level or embedded values (e.g. metadata.available)

    NOTE: The `value` key at the front of every field is implicit.
    It need not be included, although it is contained in the collection and index.
    """


def create_vector_index_config(
    dims: int,
    embed: Union[Embeddings, EmbeddingsFunc, AEmbeddingsFunc, str],
    fields: Optional[list[str]] = None,
    name: str = "vector_index",
    relevance_score_fn: Literal["euclidean", "cosine", "dotProduct"] = "cosine",
    embedding_key: str = "embedding",
    filters: Optional[list[str]] = None,
) -> VectorIndexConfig:
    """Factory function creates a VectorIndexConfig instance with sensible defaults.

    Args:
        dims: Dimensions of the embedding vectors.
        embed: Embedding model.
        fields: Field to extract text from for embedding generation (list of length 1).
        name: Arbitrary name to give to the index in Atlas.
        relevance_score_fn: Function used to establish similarity of vectors.
        embedding_key: Name of the field used in the collection to store vectors.
        filters: List of (possibly nested) fields to index allowing filtering.

    Returns: VectorIndexConfig to be passed to MongoDBStore constructor.
    """

    MongoDBStore.ensure_index_filters(filters)
    if filters and "namespace_prefix" not in filters:
        filters.append("namespace_prefix")

    return VectorIndexConfig(
        dims=dims,
        embed=embed,
        fields=fields,
        name=name,
        relevance_score_fn=relevance_score_fn,
        embedding_key=embedding_key,
        filters=filters,
    )


class MongoDBStore(BaseStore):
    """MongoDB's persistent key-value stores for long-term memory.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.

    Supports semantic search capabilities through
    an optional `index` configuration.
    Only a single embedding is permitted per item, although embedded fields
    can be indexed via ["parent.child"].
    """

    supports_ttl: bool = True

    def __init__(
        self,
        collection: Collection,
        ttl_config: Optional[TTLConfig] = None,
        index_config: Optional[VectorIndexConfig] = None,
        auto_index_timeout: int = 15,
        **kwargs: Any,
    ):
        """Construct store and its indexes.

        Semantic search is supported by an Atlas vector search index
        TTL (time to live)  is supported by a TTL index of field: updated_at.

        Args:
            collection: Collection of Items backing the store.
            ttl_config: Optionally define a TTL and whether to update on reads(get/search).
            index_config: Optionally define a VectorIndexConfig for semantic search.
            auto_index_timeout: Optional timeout for creation of indexes.

        Returns:
            Instance of MongoDBStore.
        """

        self.collection = collection
        self.ttl_config = {} if ttl_config is None else ttl_config
        self.index_config = {} if index_config is None else index_config
        self._index_name = self.index_config.get("name", "vector_index")
        self._relevance_score_fn = self.index_config.get("relevance_score_fn", "cosine")
        self._embedding_key = self.index_config.get("embedding_key", "embedding")

        # Create indexes if not present
        # Create a unique index, akin to primary key, on namespace + key
        idx_keys = [idx["key"] for idx in self.collection.list_indexes()]
        if SON([("namespace", 1), ("key", 1)]) not in idx_keys:
            self.collection.create_index(keys=["namespace", "key"], unique=True)

        # Optionally, expire values using [TTL Index](https://www.mongodb.com/docs/manual/core/index-ttl/)
        if (
            self.ttl_config
            and SON([("updated_at", 1)]) not in idx_keys
            and self.ttl_config["default_ttl"]
        ):
            self.collection.create_index(
                "updated_at", expireAfterSeconds=self.ttl_config["default_ttl"]
            )

        # If details provided, prepare vector index for semantic queries
        if self.index_config:
            self.index_field = self._ensure_index_fields(self.index_config["fields"])
            self.index_filters = self.__class__.ensure_index_filters(
                self.index_config["filters"]
            )
            self.embeddings: Embeddings = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.sep = kwargs.get("sep", "/")  # used for prefix denormalization/search

            # Create the vector index if it does not yet exist
            if not any(
                [
                    ix["name"] == self._index_name
                    for ix in collection.list_search_indexes()
                ]
            ):
                create_vector_search_index(
                    collection=collection,
                    index_name=self._index_name,
                    dimensions=self.index_config["dims"],
                    path=self._embedding_key,
                    similarity=self._relevance_score_fn,
                    filters=self.index_filters,
                    wait_until_complete=auto_index_timeout,
                )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        collection_name: str = "persistent-store",
        ttl_config: Optional[TTLConfig] = None,
        index_config: Optional[VectorIndexConfig] = None,
        **kwargs: Any,
    ) -> Iterator["MongoDBStore"]:
        """Context manager to create a persistent MongoDB key-value store.

        A unique compound index as shown below will be added to the collections
        backing the store (namespace, key). If the collection exists,
        and have indexes already, nothing will be done during initialization.

        If the `ttl` argument is provided, TTL functionality will be employed.
        This is done automatically via MongoDB's TTL Indexes, based on the
        `updated_at` field of the collection. The index will be created if it
        does not already exist.

        Args:
            conn_string: MongoDB connection string. See [class:~pymongo.MongoClient].
            db_name: Database name. It will be created if it doesn't exist.
            collection_name: Collection name backing the store. Created if it doesn't exist.
            ttl_config: Defines a TTL (in seconds) and whether to update on reads(get/search).
            index_config: Defines a VectorIndexConfig for semantic search queries.

        Yields: A new MongoDBStore.
        """

        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                conn_string,
                driver=DriverInfo(
                    name="Langgraph", version=version("langgraph-store-mongodb")
                ),
            )
            db = client[db_name]
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
            collection = client[db_name][collection_name]

            yield MongoDBStore(
                collection=collection,
                ttl_config=ttl_config,
                index_config=index_config,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        """Retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
            refresh_ttl: Whether to refresh TTLs for the returned item.
                If None (default), uses the store's default refresh_ttl setting.
                If no TTL is specified, this argument is ignored.

        Returns:
            The retrieved item or None if not found.
        """
        if refresh_ttl is False or (
            self.ttl_config and not self.ttl_config["refresh_on_read"]
        ):
            res = self.collection.find_one(
                filter={"namespace": namespace, "key": key},
            )
        else:
            res = self.collection.find_one_and_update(
                filter={"namespace": namespace, "key": key},
                update={"$set": {"updated_at": datetime.now(tz=timezone.utc)}},
                return_document=ReturnDocument.AFTER,
            )
        if res:
            return Item(
                value=res["value"],
                key=res["key"],
                namespace=tuple(res["namespace"]),
                created_at=res["created_at"],
                updated_at=res["updated_at"],
            )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
        """
        self.collection.delete_one({"namespace": list(namespace), "key": key})

    @staticmethod
    def _match_prefix(prefix: NamespacePath) -> dict[str, Any]:
        """Helper for list_namespaces."""
        if not prefix or prefix == "*":
            return {}
        if "*" not in prefix:
            return {"$eq": [{"$slice": ["$namespace", len(prefix)]}, list(prefix)]}
        matches = []
        for i, p in enumerate(prefix):
            if p != "*":
                matches.append({"$eq": [{"$arrayElemAt": ["$namespace", i]}, p]})
        return {"$and": matches}

    @staticmethod
    def _match_suffix(suffix: NamespacePath) -> dict[str, Any]:
        """Helper for list_namespaces."""
        if not suffix or suffix == "*":
            return {}
        if "*" not in suffix:
            return {"$eq": [{"$slice": ["$namespace", -1 * len(suffix)]}, list(suffix)]}
        matches = []
        for i, p in enumerate(suffix):
            if p != "*":
                matches.append(
                    {
                        "$eq": [
                            {
                                "$arrayElemAt": [
                                    "$namespace",
                                    {"$subtract": [{"$size": "$namespace"}, i]},
                                ]
                            },
                            p,
                        ]
                    }
                )
        return {"$and": matches}

    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store.

        Args:
            prefix: Filter namespaces that start with this path.
            suffix: Filter namespaces that end with this path.
            max_depth: Return namespaces up to this depth in the hierarchy.
            limit: Maximum number of namespaces to return (default 100).
            offset: Number of namespaces to skip for pagination. [Not implemented.]

        Returns: A list of namespace tuples that match the criteria.
        """
        pipeline: list[dict[str, Any]] = []
        expr = {}
        if prefix:
            precond = self._match_prefix(prefix)
            expr = {"$expr": precond}
        if suffix:
            sufcond = self._match_suffix(suffix)
            expr = {"$expr": sufcond}
        if prefix and suffix:
            expr = {"$expr": {"$and": [precond, sufcond]}}

        pipeline.append({"$match": expr})

        if max_depth:
            pipeline.append(
                {
                    "$project": {
                        "namespace": {"$slice": ["$namespace", max_depth]},
                        "_id": 0,
                    }
                }
            )
        else:
            pipeline.append({"$project": {"namespace": 1, "_id": 0}})

        if limit:
            pipeline.append({"$limit": limit})
        # Deduplicate
        pipeline.extend(
            [
                {"$group": {"_id": "$namespace"}},
                {"$project": {"_id": 0, "namespace": "$_id"}},
            ]
        )

        if offset:
            raise NotImplementedError("offset is not implemented")

        results = self.collection.aggregate(pipeline)
        return [tuple(res["namespace"]) for res in results]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.

        Get, Search, and List operations are performed on state before batch.
        Put and Delete change state. They are deduplicated and applied in order,
        but only after the read operations have completed.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
            The length of output may not match the input as PutOp returns None.
        """
        results: list[Result] = []
        dedupped_putops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        writes: list[Union[DeleteOne, UpdateOne]] = []

        for op in ops:
            if isinstance(op, PutOp):
                dedupped_putops[(op.namespace, op.key)] = op
                results.append(None)

            elif isinstance(op, GetOp):
                results.append(
                    self.get(
                        namespace=op.namespace,
                        key=op.key,
                        refresh_ttl=op.refresh_ttl,
                    )
                )

            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )

            elif isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                if op.match_conditions:
                    for cond in op.match_conditions:
                        if cond.match_type == "prefix":
                            prefix = cond.path
                        elif cond.match_type == "suffix":
                            suffix = cond.path
                        else:
                            raise ValueError(
                                f"Match type {cond.match_type} must be prefix or suffix."
                            )
                results.append(
                    self.list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
        # Apply puts and deletes in bulk
        # Extract texts to embed for each op
        if self.index_config:
            texts = self._extract_texts(list(dedupped_putops.values()))
            vectors = self.embeddings.embed_documents(texts)
            v = 0
        for op in dedupped_putops.values():
            if op.value is None:
                # mark the item for deletion.
                writes.append(
                    DeleteOne(filter={"namespace": list(op.namespace), "key": op.key})
                )
            else:
                # Add or Upsert the value
                to_set = {
                    "value": op.value,
                    "updated_at": datetime.now(tz=timezone.utc),
                }
                if self.index_config:
                    to_set[self._embedding_key] = vectors[v]
                    to_set["namespace_prefix"] = self._denormalize_path(op.namespace)
                    v += 1

                writes.append(
                    UpdateOne(
                        filter={"namespace": list(op.namespace), "key": op.key},
                        update={
                            "$set": to_set,
                            "$setOnInsert": {
                                "created_at": datetime.now(tz=timezone.utc),
                            },
                        },
                        upsert=True,
                    )
                )

        if writes:
            self.collection.bulk_write(writes)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        return await run_in_executor(None, self.batch, ops)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        Values are stored in the collection as a document of name 'value'.
        One uses dot notation to access embedded fields. For example,
        `value.text`, `value.address.city` and for arrays `value.titles.3`.

        Note that when search will not update the `last_updated` field
        and thus not affect TTL, unlike `get`.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.
            refresh_ttl: TTL is not supported for search. Use get if needed.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = store.search(
                ("docs",),
                filter={"value.type": "article", "value.status": "published"}
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Initialize store with embedding configuration
            store = YourStore( # e.g., InMemoryStore, AsyncPostgresStore
                index={
                    "dims": 1536,  # embedding dimensions
                    "embed": your_embedding_function,  # function to create embeddings
                    "fields": ["text"]  # fields to embed. Defaults to ["$"]
                }
            )

        """

        if not isinstance(namespace_prefix, tuple):
            raise TypeError("namespace_prefix must be a non-empty tuple of strings")
        if offset:
            raise NotImplementedError("offset is not implemented in MongoDBStore")
        if filter:
            if any(f.startswith("value") for f in filter):
                raise ValueError("filters should be specified without `value`")

        if query is None:
            # Case 1. $match namespace and filter
            pipeline: list[dict[str, Any]] = []
            match_cond: dict[str, Any] = {}
            if namespace_prefix:
                match_cond = {"$expr": self._match_prefix(namespace_prefix)}
            if filter:
                filter_cond = [{f"value.{k}": v} for k, v in filter.items()]
                match_cond = {"$and": [match_cond] + filter_cond}
            pipeline.append({"$match": match_cond})
            if limit:
                pipeline.append({"$limit": limit})

        else:
            # Case 2. $vectorSearch on query filtered on namespace and optional filters

            # Compute embedding
            query_vector = self.embeddings.embed_query(query)
            # Form filter condition for namespace_prefix
            filter_vec = {"namespace_prefix": self.sep.join(namespace_prefix)}
            if filter:  # and add any specified
                filter_cond = [{f"value.{k}": v} for k, v in filter.items()]
                filter_vec = {"$and": [filter_vec] + filter_cond}

            pipeline = [
                vector_search_stage(
                    query_vector=query_vector,
                    search_field=self._embedding_key,
                    index_name=self._index_name,
                    top_k=limit,
                    filter=filter_vec,
                ),
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
                {"$project": {self._embedding_key: 0}},
            ]

        results = self.collection.aggregate(pipeline)

        return [
            SearchItem(
                namespace=tuple(res["namespace"]),
                key=res["key"],
                value=res["value"],
                created_at=res["created_at"],
                updated_at=res["updated_at"],
                score=res.get("score"),
            )
            for res in results
        ]

    def _denormalize_path(self, paths: Union[tuple[str, ...], list[str]]) -> list[str]:
        """Create list of path parents, for use in $vectorSearch filter.

        ???+ example "Example"
        ```python
        namespace = ('parent', 'child', 'pet')
        prefixes=store_mdb.denormalize_path(namespace)
        assert prefixes == ['parent', 'parent/child', 'parent/child/pet']
        ```
        """
        return [self.sep.join(paths[:i]) for i in range(1, len(paths) + 1)]

    def _extract_texts(self, put_ops: Optional[list[PutOp]]) -> list[str]:
        """Extract text to embed according to index config."""
        if put_ops and self.index_config and self.embeddings:
            to_embed = []
            for op in put_ops:
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        field = self.index_field
                    else:
                        field = self._ensure_index_fields(list(op.index))
                    texts = get_text_at_path(op.value, field)
                    if texts:
                        if len(texts) > 1:
                            raise ValueError("Got multiple texts. Report as bug.")

                        else:
                            to_embed.append(texts[0])
            return to_embed
        else:
            return []

    @staticmethod
    def _ensure_index_fields(fields: Optional[list[str]]) -> str:
        """Ensure that requested fields to be indexed result in a single vector.

        We require that one document may only have one embedding vector.
        """
        if fields and (len(fields) > 1 or "*" in fields[0]):
            raise ValueError("Only one field can be indexed for queries.")
        if isinstance(fields, list):
            return fields[0]
        else:
            return fields

    @classmethod
    def ensure_index_filters(cls, filters: Optional[list[str]] = None) -> list[str]:
        """Prepare filters for Atlas indexing.

        We must ensure that `namespace_prefix` is included in the filter.
        We also must ensure that the implicit `value` field is added.
        """
        filters = [] if filters is None else filters
        if not isinstance(filters, list):
            raise ValueError(
                "Index filters must be a list. Found: ",
                type(filters),
            )
        filters = [
            f"value.{field}"
            for field in filters
            if not field.startswith("values") and field != "namespace_prefix"
        ]
        if "namespace_prefix" not in filters:
            filters.append("namespace_prefix")
        return filters
