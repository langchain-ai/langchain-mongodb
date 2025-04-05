import pytest

from langchain_mongodb.docstores import MongoDBDocStore
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
    MongoDBAtlasParentDocumentRetriever,
)
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from ..utils import ConsistentFakeEmbeddings, MockCollection


@pytest.fixture()
def collection() -> MockCollection:
    return MockCollection()


@pytest.fixture()
def embeddings() -> ConsistentFakeEmbeddings:
    return ConsistentFakeEmbeddings()


def test_full_text_search(collection):
    search = MongoDBAtlasFullTextSearchRetriever(
        collection=collection, search_index_name="foo", search_field="bar"
    )
    search.close()
    assert collection.is_closed


def test_hybrid_search(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings)
    search = MongoDBAtlasHybridSearchRetriever(vectorstore=vs, search_index_name="foo")
    search.close()
    assert collection.is_closed


def test_parent_retriever(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings)
    ds = MongoDBDocStore(collection)
    retriever = MongoDBAtlasParentDocumentRetriever(vectorstore=vs, docstore=ds)
    retriever.close()
    assert collection.is_closed
