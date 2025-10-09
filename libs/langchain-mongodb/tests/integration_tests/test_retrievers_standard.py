from typing import Type

from langchain_core.documents import Document
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    create_fulltext_search_index,
)
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
)

from ..utils import (
    CONNECTION_STRING,
    DB_NAME,
    TIMEOUT,
    ConsistentFakeEmbeddings,
    PatchedMongoDBAtlasVectorSearch,
)

DIMENSIONS = 5
COLLECTION_NAME = "langchain_test_retrievers_standard"
VECTOR_INDEX_NAME = "vector_index"
PAGE_CONTENT_FIELD = "text"
SEARCH_INDEX_NAME = "text_index"


def get_collection() -> Collection:
    client = MongoClient(CONNECTION_STRING)
    return client[DB_NAME][COLLECTION_NAME]


def setup_test(coll: Collection) -> MongoDBAtlasVectorSearch:
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

    return vs


class TestMongoDBAtlasFullTextSearchRetriever(RetrieversIntegrationTests):
    _coll: Collection

    @classmethod
    def setup_class(cls):
        cls._coll = get_collection()

    @classmethod
    def teardown_class(cls):
        cls._coll.database.client.close()

    @property
    def retriever_constructor(self) -> Type[MongoDBAtlasFullTextSearchRetriever]:
        """Get a retriever for integration tests."""
        return MongoDBAtlasFullTextSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        setup_test(self._coll)
        return {
            "collection": self._coll,
            "search_index_name": SEARCH_INDEX_NAME,
            "search_field": PAGE_CONTENT_FIELD,
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "When was the last time I visited new orleans?"


class TestMongoDBAtlasHybridSearchRetriever(RetrieversIntegrationTests):
    _coll: Collection

    @classmethod
    def setup_class(cls):
        cls._coll = get_collection()

    @classmethod
    def teardown_class(cls):
        cls._coll.database.client.close()

    @property
    def retriever_constructor(self) -> Type[MongoDBAtlasHybridSearchRetriever]:
        """Get a retriever for integration tests."""
        return MongoDBAtlasHybridSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        vs = setup_test(self._coll)
        return {
            "vectorstore": vs,
            "collection": self._coll,
            "search_index_name": SEARCH_INDEX_NAME,
            "search_field": PAGE_CONTENT_FIELD,
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "When was the last time I visited new orleans?"
