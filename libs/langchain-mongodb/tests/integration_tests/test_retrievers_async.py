from time import sleep, time
from typing import AsyncGenerator, Generator, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient, AsyncMongoClient
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
import pytest_asyncio
from ..utils import DB_NAME, PatchedMongoDBAtlasVectorSearch, CONNECTION_STRING

COLLECTION_NAME = "langchain_test_retrievers"
COLLECTION_NAME_NESTED = "langchain_test_retrievers_nested"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_FIELD = "embedding"
PAGE_CONTENT_FIELD = "text"
PAGE_CONTENT_FIELD_NESTED = "title.text"
SEARCH_INDEX_NAME = "text_index"
SEARCH_INDEX_NAME_NESTED = "text_index_nested"

TIMEOUT = 60.0
INTERVAL = 0.5

@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncMongoClient, None]:
    client = AsyncMongoClient(CONNECTION_STRING)
    yield client
    await client.close()

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
def collection_nested(client: MongoClient, dimensions: int) -> Collection:
    """A Collection with both a Vector and a Full-text Search Index"""
    if COLLECTION_NAME_NESTED not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME_NESTED)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME_NESTED]

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

    if not any(
        [SEARCH_INDEX_NAME_NESTED == ix["name"] for ix in clxn.list_search_indexes()]
    ):
        create_fulltext_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME_NESTED,
            field=PAGE_CONTENT_FIELD_NESTED,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest.fixture(scope="function")
def indexed_vectorstore(
    collection: Collection,
    async_client: AsyncMongoClient,
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

    # yield vectorstore with async intialization
    clxn = async_client[DB_NAME][COLLECTION_NAME]
    async_vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=clxn, # type: ignore[arg-type]
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    yield async_vectorstore

    vectorstore.collection.delete_many({})


@pytest.fixture(scope="function")
def indexed_nested_vectorstore(
    collection_nested: Collection,
    async_client: AsyncMongoClient,
    example_documents: List[Document],
    embedding: Embeddings,
) -> Generator[MongoDBAtlasVectorSearch, None, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection_nested,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD_NESTED,
    )

    vectorstore.add_documents(example_documents)

    # yield vectorstore with async intialization
    clxn = async_client[DB_NAME][COLLECTION_NAME_NESTED]
    async_vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=clxn, # type: ignore[arg-type]
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD_NESTED,
    )

    yield async_vectorstore

    vectorstore.collection.delete_many({})



@pytest.mark.asyncio
async def test_hybrid_retriever_async(indexed_vectorstore: PatchedMongoDBAtlasVectorSearch) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        k=3,
    )

    query1 = "When did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content

@pytest.mark.asyncio
async def test_hybrid_retriever_deprecated_top_k_async(
    indexed_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        top_k=3,
    )

    query1 = "When did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content

@pytest.mark.asyncio
async def test_hybrid_retriever_nested_async(
    indexed_nested_vectorstore: PatchedMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_nested_vectorstore,
        search_index_name=SEARCH_INDEX_NAME_NESTED,
        k=3,
    )

    query1 = "What did I visit France?"
    results = await retriever.ainvoke(query1)
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content


