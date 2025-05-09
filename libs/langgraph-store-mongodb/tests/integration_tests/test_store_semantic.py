import os
from collections.abc import Generator
from time import monotonic, sleep
from typing import Callable

import pytest
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore
from langgraph.store.mongodb import (
    MongoDBStore,
    create_vector_index_config,
)

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "semantic_search"
INDEX_NAME = "vector_index"
TIMEOUT, INTERVAL = 30, 1  # timeout to index new data

DIMENSIONS = 5  # Dimensions of embedding model


def wait(cond: Callable, timeout: int = 15, interval: int = 1) -> None:
    start = monotonic()
    while monotonic() - start < timeout:
        if cond():
            return
        else:
            sleep(interval)
    raise TimeoutError("timeout waiting for: ", cond)


class StaticEmbeddings(Embeddings):
    """ANN Search is not tested here. That is done in langchain-mongodb."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for txt in texts:
            vectors.append(self.embed_query(txt))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if "pears" in text:
            return [1.0] + [0.5] * (DIMENSIONS - 1)
        else:
            return [0.5] * DIMENSIONS


@pytest.fixture
def collection() -> Generator[Collection, None, None]:
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)
    collection = db.create_collection(COLLECTION_NAME)
    wait(lambda: collection.count_documents({}) == 0, TIMEOUT, INTERVAL)
    try:
        collection.drop_search_index(INDEX_NAME)
    except OperationFailure:
        pass
    wait(
        lambda: len(collection.list_search_indexes().to_list()) == 0, TIMEOUT, INTERVAL
    )

    yield collection

    client.close()


def test_filters(collection: Collection) -> None:
    """Test permutations of namespace_prefix in filter."""

    index_config = create_vector_index_config(
        name=INDEX_NAME,
        dims=DIMENSIONS,
        fields=["product"],
        embed=StaticEmbeddings(),  # embedding
        filters=["metadata.available"],
    )
    store_mdb = MongoDBStore(
        collection, index_config=index_config, auto_index_timeout=TIMEOUT
    )
    store_in_mem = InMemoryStore(index=index_config)

    namespaces = [
        ("a",),
        ("a", "b", "c"),
        ("a", "b", "c", "d"),
    ]

    products = ["apples", "oranges", "pears"]

    # Add some indexed data
    put_ops = []
    for i, ns in enumerate(namespaces):
        put_ops.append(
            PutOp(
                namespace=ns,
                key=f"id_{i}",
                value={
                    "product": products[i],
                    "metadata": {"available": bool(i % 2), "grade": "A" * (i + 1)},
                },
            )
        )

    store_mdb.batch(put_ops)
    store_in_mem.batch(put_ops)

    query = "What is the grade of our pears?"
    # Case 1: fields is a string:
    namespace_prefix = ("a",)  #  filter ("a",) catches all docs
    wait(
        lambda: len(store_mdb.search(namespace_prefix, query=query)) == len(products),
        TIMEOUT,
        INTERVAL,
    )

    result_mdb = store_mdb.search(namespace_prefix, query=query)
    assert result_mdb[0].value["product"] == "pears"  # test sorted by score

    result_mem = store_in_mem.search(namespace_prefix, query=query)
    assert len(result_mem) == len(products)

    # Case 2: filter on 2nd namespace in hierarchy
    namespace_prefix = ("a", "b")
    result_mem = store_in_mem.search(namespace_prefix, query=query)
    result_mdb = store_mdb.search(namespace_prefix, query=query)
    # filter ("a",) catches all docs
    assert len(result_mem) == len(result_mdb) == len(products) - 1
    assert result_mdb[0].value["product"] == "pears"

    # Case 3: Empty  namespace_prefix
    namespace_prefix = ("",)
    result_mem = store_in_mem.search(namespace_prefix, query=query)
    result_mdb = store_mdb.search(namespace_prefix, query=query)
    assert len(result_mem) == len(result_mdb) == 0

    # Case 4: With filter on value (nested)
    namespace_prefix = ("a",)
    available = {"metadata.available": True}
    result_mdb = store_mdb.search(namespace_prefix, query=query, filter=available)
    assert result_mdb[0].value["product"] == "oranges"
    assert len(result_mdb) == 1
