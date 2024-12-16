import os
from typing import Generator, List, Optional

import pytest
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch, index

from ..utils import ConsistentFakeEmbeddings

DB_NAME = "langchain_test_index_db"
COLLECTION_NAME = "test_index"
VECTOR_INDEX_NAME = "vector_index"

TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture
def collection() -> Generator:
    """Depending on uri, this could point to any type of cluster."""
    uri = os.environ.get("MONGODB_URI")
    client: MongoClient = MongoClient(uri)
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]
    clxn = client[DB_NAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


def test_search_index_drop_add_delete_commands(collection: Collection) -> None:
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: Optional[List[str]] = None
    wait_until_complete = TIMEOUT

    for index_info in collection.list_search_indexes():
        index.drop_vector_search_index(
            collection, index_info["name"], wait_until_complete=wait_until_complete
        )

    assert len(list(collection.list_search_indexes())) == 0

    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    index.drop_vector_search_index(
        collection, index_name, wait_until_complete=wait_until_complete
    )

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 0


@pytest.mark.skip("collection.update_vector_search_index requires [CLOUDP-275518]")
def test_search_index_update_vector_search_index(collection: Collection) -> None:
    index_name = "INDEX_TO_UPDATE"
    similarity_orig = "cosine"
    similarity_new = "euclidean"

    # Create another index
    index.create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_orig,
        wait_until_complete=TIMEOUT,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_orig

    # Update the index and test new similarity
    index.update_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_new,
        wait_until_complete=TIMEOUT,
    )

    assert index._is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_new


def test_vectorstore_create_vector_search_index(collection: Collection) -> None:
    """Tests vectorstore wrapper around index command."""

    # Set up using the index module's api
    if len(list(collection.list_search_indexes())) != 0:
        index.drop_vector_search_index(
            collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
        )

    # Test MongoDBAtlasVectorSearch's API
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=ConsistentFakeEmbeddings(),
        index_name=VECTOR_INDEX_NAME,
    )

    vectorstore.create_vector_search_index(
        dimensions=DIMENSIONS, wait_until_complete=TIMEOUT
    )

    assert index._is_index_ready(collection, VECTOR_INDEX_NAME)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == VECTOR_INDEX_NAME

    # Tear down using the index module's api
    index.drop_vector_search_index(
        collection, VECTOR_INDEX_NAME, wait_until_complete=TIMEOUT
    )
