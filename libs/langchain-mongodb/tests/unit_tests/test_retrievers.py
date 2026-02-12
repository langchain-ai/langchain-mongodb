import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    assert collection.database.client.is_closed


def test_hybrid_search(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings)
    search = MongoDBAtlasHybridSearchRetriever(vectorstore=vs, search_index_name="foo")
    search.close()
    assert collection.database.client.is_closed


def test_parent_retriever(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings)
    ds = MongoDBDocStore(collection)
    cs = RecursiveCharacterTextSplitter(chunk_size=400)
    retriever = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vs, docstore=ds, child_splitter=cs
    )
    retriever.close()
    assert collection.database.client.is_closed


def test_full_text_search_retriever_auto_create_index(collection):
    assert len(collection._search_indexes) == 0
    _ = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name="foo",
        search_field="bar",
    )
    assert len(collection._search_indexes) == 1

    collection._search_indexes = []
    _ = MongoDBAtlasFullTextSearchRetriever(
        collection=collection,
        search_index_name="foo",
        search_field="bar",
        auto_create_index=False,
    )
    assert len(collection._search_indexes) == 0


def test_hybrid_search_retriever_auto_create_index(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings, auto_create_index=False)
    assert len(collection._search_indexes) == 0
    _ = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vs,
        search_index_name="foo",
    )
    assert len(collection._search_indexes) == 1

    # With auto_create_index=False, does not create
    collection._search_indexes = []
    vs2 = MongoDBAtlasVectorSearch(collection, embeddings, auto_create_index=False)
    _ = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vs2,
        search_index_name="foo",
        auto_create_index=False,
    )
    assert len(collection._search_indexes) == 0


def test_parent_document_retriever_auto_create_index(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings, auto_create_index=False)
    ds = MongoDBDocStore(collection)
    cs = RecursiveCharacterTextSplitter(chunk_size=400)
    assert len(collection._search_indexes) == 0
    _ = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vs,
        docstore=ds,
        child_splitter=cs,
    )
    assert len(collection._search_indexes) == 1

    # With auto_create_index=False, does not create
    collection._search_indexes = []
    vs2 = MongoDBAtlasVectorSearch(collection, embeddings, auto_create_index=False)
    ds2 = MongoDBDocStore(collection)
    _ = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vs2,
        docstore=ds2,
        child_splitter=cs,
        auto_create_index=False,
    )
    assert len(collection._search_indexes) == 0
