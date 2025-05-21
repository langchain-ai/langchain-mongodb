from typing import Type
import pytest
from langchain_core.documents import Document
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)
from langchain_core.retrievers import BaseRetriever
from pymongo import MongoClient, AsyncMongoClient
from pymongo.collection import Collection
from pymongo.asynchronous.collection import AsyncCollection
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    create_fulltext_search_index,
)
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
    AsyncMongoDBAtlasHybridSearchRetriever
)

from ..utils import (
    CONNECTION_STRING,
    DB_NAME,
    TIMEOUT,
    ConsistentFakeEmbeddings,
    PatchedMongoDBAtlasVectorSearch,
    PatchedAsyncMongoDBAtlasVectorSearch
)

DIMENSIONS = 5
COLLECTION_NAME = "langchain_test_retrievers_standard"
VECTOR_INDEX_NAME = "vector_index"
PAGE_CONTENT_FIELD = "text"
SEARCH_INDEX_NAME = "text_index"


def setup_test() -> tuple[Collection, MongoDBAtlasVectorSearch]:
    client = MongoClient(CONNECTION_STRING)
    coll = client[DB_NAME][COLLECTION_NAME]

    # Set up the vector search index and add the documents if needed.
    vs = PatchedMongoDBAtlasVectorSearch(
        coll,
        embedding=ConsistentFakeEmbeddings(DIMENSIONS),
        dimensions=DIMENSIONS,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
        auto_index_timeout=TIMEOUT,
    )

    if coll.count_documents({}) == 0:
        vs.add_documents(
            [
                Document(page_content="In 2023, I visited Paris"),
                Document(page_content="In 2022, I visited New York"),
                Document(page_content="In 2021, I visited New Orleans"),
                Document(page_content="Sandwiches are beautiful. Sandwiches are fine."),
            ]
        )

    # Set up the search index if needed.
    if not any([ix["name"] == SEARCH_INDEX_NAME for ix in coll.list_search_indexes()]):
        create_fulltext_search_index(
            collection=coll,
            index_name=SEARCH_INDEX_NAME,
            field=PAGE_CONTENT_FIELD,
            wait_until_complete=TIMEOUT,
        )

    client = AsyncMongoClient(CONNECTION_STRING)
    coll = client[DB_NAME][COLLECTION_NAME]

    vs = PatchedMongoDBAtlasVectorSearch(
        coll,
        auto_create_index=False,
        embedding=ConsistentFakeEmbeddings(DIMENSIONS),
        dimensions=DIMENSIONS,
        index_name=VECTOR_INDEX_NAME,
        text_key=PAGE_CONTENT_FIELD,
        auto_index_timeout=TIMEOUT,
    )


    return coll, vs


class TestAsyncMongoDBAtlasHybridSearchRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[AsyncMongoDBAtlasHybridSearchRetriever]:
        """Get a retriever for integration tests."""
        return AsyncMongoDBAtlasHybridSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        coll, vs = setup_test()
        return {
            "vectorstore": vs,
            "collection": coll,
            "search_index_name": SEARCH_INDEX_NAME,
            "search_field": PAGE_CONTENT_FIELD,
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "When was the last time I visited new orleans?"


    @pytest.mark.xfail(reason="Async HybridSearchRetriever does not support invoke and needs to be used with await")
    async def test_k_constructor_param(self) -> None:
        """
        Test that the retriever constructor accepts a k parameter, representing
        the number of documents to return.

        .. dropdown:: Troubleshooting

            If this test fails, either the retriever constructor does not accept a k
            parameter, or the retriever does not return the correct number of documents
            (`k`) when it is set.

            For example, a retriever like

            .. code-block:: python

                    MyRetriever(k=3).invoke("query")

            should return 3 documents when invoked with a query.
        """
        retriver_params = self.retriever_constructor_params
        params = {
            k: v for k, v in retriver_params.items() if k != "k"
        }
        params_3 = {**params, "k": 3}
        retriever_3 = self.retriever_constructor(**params_3)
        result_3 = await retriever_3.ainvoke(self.retriever_query_example)
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

        params_1 = {**params, "k": 1}
        retriever_1 = self.retriever_constructor(**params_1)
        result_1 = await retriever_1.ainvoke(self.retriever_query_example)
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)
    @pytest.mark.xfail(reason="Async HybridSearchRetriever does not support invoke and needs to be used with await")
    async def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        """
        Test that the invoke method accepts a k parameter, representing the number of
        documents to return.

        .. dropdown:: Troubleshooting

            If this test fails, the retriever's invoke method does not accept a k
            parameter, or the retriever does not return the correct number of documents
            (`k`) when it is set.

            For example, a retriever like

            .. code-block:: python

                MyRetriever().invoke("query", k=3)

            should return 3 documents when invoked with a query.
        """
        result_1 = await retriever.ainvoke(self.retriever_query_example, k=1)
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)

        result_3 = await retriever.ainvoke(self.retriever_query_example, k=3)
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

    @pytest.mark.xfail(reason="Async HybridSearchRetriever does not support invoke and needs to be used with await")
    async def test_invoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """
        If invoked with the example params, the retriever should return a list of
        Documents.

        .. dropdown:: Troubleshooting

            If this test fails, the retriever's invoke method does not return a list of
            `langchain_core.document.Document` objects. Please confirm that your
            `_get_relevant_documents` method returns a list of `Document` objects.
        """
        result = await retriever.ainvoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)