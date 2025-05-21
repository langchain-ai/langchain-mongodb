from time import sleep, time
from typing import AsyncGenerator, Generator, List
from pymongo.asynchronous.collection import AsyncCollection
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo import AsyncMongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    create_fulltext_search_index,
    create_vector_search_index,
)
from pymongo import AsyncMongoClient
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
    AsyncMongoDBAtlasHybridSearchRetriever,
)
import pytest_asyncio
from ..utils import DB_NAME, PatchedAsyncMongoDBAtlasVectorSearch
from ..utils import CONNECTION_STRING
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

@pytest_asyncio.fixture(scope="module")
def client() -> Generator[AsyncMongoClient, None, None]:
    client = AsyncMongoClient(CONNECTION_STRING)
    yield client
    client.close()


@pytest.fixture(scope="module")
def example_documents() -> List[Document]:
    return [
        Document(page_content="In 2023, I visited Paris"),
        Document(page_content="In 2022, I visited New York"),
        Document(page_content="In 2021, I visited New Orleans"),
        Document(page_content="Sandwiches are beautiful. Sandwiches are fine."),
    ]


@pytest_asyncio.fixture(scope="module")
async def collection(client: AsyncMongoClient, dimensions: int) -> AsyncCollection:
    """A Collection with both a Vector and a Full-text Search Index"""
    if COLLECTION_NAME not in await client[DB_NAME].list_collection_names():
        clxn = await client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]

    await clxn.delete_many({})
    search_indices = await clxn.list_search_indexes()
    search_indices_list = [ix async for ix in search_indices]
    if not any([VECTOR_INDEX_NAME == ix["name"] for ix in search_indices_list]):
        await create_vector_search_index(
            collection=clxn,
            index_name=VECTOR_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
            wait_until_complete=TIMEOUT,
        )
    search_indices = await clxn.list_search_indexes()
    search_indices_list = [ix async for ix in search_indices]
    if not any([SEARCH_INDEX_NAME == ix["name"] for ix in search_indices_list]):
        await create_fulltext_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest_asyncio.fixture(scope="module")
async def collection_nested(client: AsyncMongoClient, dimensions: int) -> AsyncCollection:
    """A Collection with both a Vector and a Full-text Search Index"""
    if COLLECTION_NAME_NESTED not in await client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME_NESTED)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME_NESTED]

    await clxn.delete_many({})

    vector_indices = await clxn.list_search_indexes()
    search_indices_list = [ix async for ix in vector_indices]
    if not any([VECTOR_INDEX_NAME == ix["name"] for ix in search_indices_list]):
        create_vector_search_index(
            collection=clxn,
            index_name=VECTOR_INDEX_NAME,
            dimensions=dimensions,
            path="embedding",
            similarity="cosine",
            wait_until_complete=TIMEOUT,
        )
    vector_indices = await clxn.list_search_indexes()
    search_indices_list = [ix async for ix in vector_indices]
    if not any(
        [SEARCH_INDEX_NAME_NESTED == ix["name"] for ix in search_indices_list]
    ):
        create_fulltext_search_index(
            collection=clxn,
            index_name=SEARCH_INDEX_NAME_NESTED,
            field=PAGE_CONTENT_FIELD_NESTED,
            wait_until_complete=TIMEOUT,
        )

    return clxn


@pytest_asyncio.fixture(scope="module")
async def indexed_vectorstore(
    collection: AsyncCollection,
    example_documents: List[Document],
    embedding: Embeddings,
) -> AsyncGenerator[MongoDBAtlasVectorSearch, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedAsyncMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
    )

    await vectorstore.add_documents(example_documents)

    yield vectorstore

    await vectorstore.collection.delete_many({})


@pytest_asyncio.fixture(scope="module")
async def indexed_nested_vectorstore(
    collection_nested: AsyncCollection,
    example_documents: List[Document],
    embedding: Embeddings,
) -> AsyncGenerator[MongoDBAtlasVectorSearch, None]:
    """Return a VectorStore with example document embeddings indexed."""

    vectorstore = PatchedAsyncMongoDBAtlasVectorSearch(
        collection=collection_nested,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD_NESTED,
    )

    await vectorstore.add_documents(example_documents)

    yield vectorstore

    await vectorstore.collection.delete_many({})



@pytest.mark.asyncio
async def test_async_hybrid_retriever(indexed_vectorstore: PatchedAsyncMongoDBAtlasVectorSearch) -> None:
    """Test basic usage of AsyncMongoDBAtlasHybridSearchRetriever"""
    retriever = AsyncMongoDBAtlasHybridSearchRetriever(
        vectorstore=indexed_vectorstore,
        search_index_name=SEARCH_INDEX_NAME,
        k=3,
    )

    query1 = "When did I visit France?"
    results = await retriever.ainvoke(query1)
    print(indexed_nested_vectorstore)
    print(SEARCH_INDEX_NAME)
    print(retriever.k)
    
    assert len(results) == 3
    assert "Paris" in results[0].page_content

    query2 = "When was the last time I visited new orleans?"
    results = await retriever.ainvoke(query2)
    assert "New Orleans" in results[0].page_content

@pytest.mark.asyncio
async def test_async_hybrid_retriever_deprecated_top_k(
    indexed_vectorstore: PatchedAsyncMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of AsyncMongoDBAtlasHybridSearchRetriever"""
    retriever = AsyncMongoDBAtlasHybridSearchRetriever(
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
async def test_async_hybrid_retriever_nested(
    indexed_nested_vectorstore: PatchedAsyncMongoDBAtlasVectorSearch,
) -> None:
    """Test basic usage of MongoDBAtlasHybridSearchRetriever"""
    retriever = AsyncMongoDBAtlasHybridSearchRetriever(
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



