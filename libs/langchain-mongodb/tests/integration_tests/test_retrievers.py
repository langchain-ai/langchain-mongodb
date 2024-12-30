from time import sleep, time
from typing import Generator, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    create_fulltext_search_index,
    create_vector_search_index,
)
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
)

from ..utils import PatchedMongoDBAtlasVectorSearch

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_retrievers"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
PAGE_CONTENT_FIELD = "text"
SEARCH_INDEX_NAME = "text_index"

TIMEOUT = 60.0
INTERVAL = 0.5


@pytest.fixture(scope="module")
def example_documents() -> List[Document]:
    return [
        Document(page_content="In 2023, I visited Paris"),
        Document(page_content="In 2022, I visited New York"),
        Document(page_content="In 2021, I visited New Orleans"),
        Document(page_content="Sandwiches are beautiful. Sandwiches are fine."),
    ]


@pytest.fixture(scope="module")
def collection(client: MongoClient, dimensions: int) -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    clxn.delete_many({})

    if not any([VECTOR_INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        create_vector_search_index(
            collection=clxn,
            index_name=VECTOR_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
            wait_until_complete=TIMEOUT,
        )

    if not any([SEARCH_INDEX_NAME == ix["name"] for ix in clxn.list_search_indexes()]):
        create_fulltext_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest.fixture(scope="module")
def indexed_vectorstore(
    collection: Collection,
    example_documents: List[Document],
    embedding: Embeddings,
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    vectorstore.add_documents(example_documents)

    yield vectorstore

    vectorstore.collection.delete_many({})


def test_vector_retriever(indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test VectorStoreRetriever"""
    retriever = indexed_vectorstore.as_retriever()

    query1 = "When did I visit France?"
    results = retriever.invoke(query1)
    assert len(results) == 4
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_hybrid_retriever(indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        top_k=3,
    )

    query1 = "What did I visit France?"
    results = retriever.invoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = retriever.invoke(query2)
    assert "New Orleans" in results[0].page_content


def test_fulltext_retriever(
    indexed_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test result of performing fulltext search.

    The Retriever is independent of the VectorStore.
    We use it here only to get the Collection, which we know to be indexed.
    """

    collection: Collection = indexed_vectorstore.collection

    retriever = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name=SEARCH_INDEX_NAME,
        search_field=PAGE_CONTENT_FIELD,
    )

    # Wait for the search index to complete.
    search_content = dict(
        index=SEARCH_INDEX_NAME,
        wildcard=dict(query="*", path=PAGE_CONTENT_FIELD, allowAnalyzedField=True),
    )
    n_docs = collection.count_documents({})
    t0 = time()
    while True:
        if (time() - t0) > TIMEOUT:
            raise TimeoutError(
                f"Search index {SEARCH_INDEX_NAME} did not complete in {TIMEOUT}"
            )
        cursor = collection.aggregate([{"$search": search_content}])
        if len(list(cursor)) == n_docs:
            break
        sleep(INTERVAL)

    query = "When was the last time I visited new orleans?"
    results = retriever.invoke(query)
    assert "New Orleans" in results[0].page_content
    assert "score" in results[0].metadata
