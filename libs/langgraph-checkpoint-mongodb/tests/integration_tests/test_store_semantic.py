import os
from collections.abc import Generator
from datetime import datetime
from time import monotonic, sleep
from typing import Callable

import pytest
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

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


t0 = (datetime(2025, 4, 7, 17, 29, 10, 0),)


def wait(cond: Callable, timeout: int = 15, interval: int = 1) -> None:
    start = monotonic()
    while monotonic() - start < timeout:
        if cond():
            return
        else:
            sleep(interval)
    raise TimeoutError("timeout waiting for: ", cond)


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


def test_index_top_level_key(
    collection: Collection, embedding: Embeddings, dimensions: int
) -> None:
    """
    - Test filter as well as query
    - Test embedding of value dictionary.
        - First, that it works.
        - 2nd - what forms we can take
        - is it always dict[str]
    """

    index_config = create_vector_index_config(
        name=INDEX_NAME,
        dims=dimensions,
        fields=["product"],
        embed=embedding,
        filters=["grade"],
    )
    store_mdb = MongoDBStore(
        collection, index_config=index_config, auto_index_timeout=TIMEOUT
    )
    store_in_mem = InMemoryStore(index=index_config)

    namespaces = [
        ("a",),
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
    ]

    products = ["apples", "oranges", "pears"]

    # Add some indexed data
    for i, ns in enumerate(namespaces):
        store_mdb.put(
            namespace=ns,
            key=f"id_{i}",
            value={
                "product": products[i],
                "metadata": {"available": True, "grade": "A" * (i + 1)},
            },
        )
        store_in_mem.put(
            namespace=ns,
            key=f"id_{i}",
            value={
                "product": products[i],
                "metadata": {"available": True, "grade": "A" * (i + 1)},
            },
        )

    # Case 1: fields is a string:
    namespace_prefix = ("a",)  #  filter ("a",) catches all docs
    query = "What is the grade of our pears?"

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
    query = "What is the grade of our pears?"
    namespace_prefix = ("a", "b")
    result_mem = store_in_mem.search(namespace_prefix, query=query)
    result_mdb = store_mdb.search(namespace_prefix, query=query)
    # filter ("a",) catches all docs
    assert len(result_mem) == len(result_mdb) == len(products) - 1
    assert result_mdb[0].value["product"] == "pears"

    # Case 3: Empty  namespace_prefix
    query = "What is the grade of our pears?"
    namespace_prefix = ("",)
    result_mem = store_in_mem.search(namespace_prefix, query=query)
    result_mdb = store_mdb.search(namespace_prefix, query=query)
    assert len(result_mem) == len(result_mdb) == 0
