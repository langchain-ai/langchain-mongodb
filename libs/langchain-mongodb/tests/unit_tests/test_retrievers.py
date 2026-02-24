import datetime

import pytest
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_core.structured_query import Comparator, Comparison
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mongodb.docstores import MongoDBDocStore
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
    MongoDBAtlasParentDocumentRetriever,
    MongoDBAtlasSelfQueryRetriever,
)
from langchain_mongodb.retrievers.self_querying import MongoDBStructuredQueryTranslator
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from ..utils import ConsistentFakeEmbeddings, FakeChatModel, MockCollection

_FIELD_INFO = [
    AttributeInfo(name="genre", description="The genre of the movie", type="string"),
]
_DOC_CONTENTS = "Descriptions of movies"


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


def test_self_query_retriever_default_translator(collection, embeddings):
    vs = MongoDBAtlasVectorSearch(collection, embeddings, auto_create_index=False)
    retriever = MongoDBAtlasSelfQueryRetriever.from_llm(
        llm=FakeChatModel(),
        vectorstore=vs,
        document_contents=_DOC_CONTENTS,
        metadata_field_info=_FIELD_INFO,
    )
    assert isinstance(
        retriever.structured_query_translator, MongoDBStructuredQueryTranslator
    )


@pytest.fixture()
def translator() -> MongoDBStructuredQueryTranslator:
    return MongoDBStructuredQueryTranslator()


def _comparison(attribute: str, comparator: Comparator, value) -> Comparison:
    return Comparison(attribute=attribute, comparator=comparator, value=value)


def test_visit_comparison_iso8601_date(translator):
    """ISO8601Date dicts from the query parser are converted to datetime objects."""
    iso_date = {"date": "2025-01-01", "type": "date"}
    result = translator.visit_comparison(
        _comparison("timestamp", Comparator.GTE, iso_date)
    )
    assert result == {"timestamp": {"$gte": datetime.datetime(2025, 1, 1)}}


def test_visit_comparison_iso8601_datetime(translator):
    """ISO8601DateTime dicts (with and without a Z suffix) are converted correctly."""
    iso_dt = {"datetime": "2025-06-15T12:30:00", "type": "datetime"}
    result = translator.visit_comparison(
        _comparison("created_at", Comparator.LT, iso_dt)
    )
    assert result == {"created_at": {"$lt": datetime.datetime(2025, 6, 15, 12, 30, 0)}}

    iso_dt_z = {"datetime": "2025-06-15T12:30:00Z", "type": "datetime"}
    result_z = translator.visit_comparison(
        _comparison("created_at", Comparator.LT, iso_dt_z)
    )
    expected_z = datetime.datetime(2025, 6, 15, 12, 30, 0, tzinfo=datetime.timezone.utc)
    assert result_z == {"created_at": {"$lt": expected_z}}


def test_visit_comparison_date_range(translator):
    """AND of two date comparisons produces proper datetime values on both sides."""
    from langchain_core.structured_query import Operation, Operator

    gte_cmp = _comparison(
        "timestamp", Comparator.GTE, {"date": "2025-01-01", "type": "date"}
    )
    lte_cmp = _comparison(
        "timestamp", Comparator.LTE, {"date": "2025-12-31", "type": "date"}
    )
    op = Operation(operator=Operator.AND, arguments=[gte_cmp, lte_cmp])
    result = translator.visit_operation(op)
    assert result == {
        "$and": [
            {"timestamp": {"$gte": datetime.datetime(2025, 1, 1)}},
            {"timestamp": {"$lte": datetime.datetime(2025, 12, 31)}},
        ]
    }


def test_visit_comparison_non_date_unchanged(translator):
    """Plain scalar values (int, float, str) are passed through unmodified."""
    assert translator.visit_comparison(_comparison("rating", Comparator.GT, 8.5)) == {
        "rating": {"$gt": 8.5}
    }

    assert translator.visit_comparison(
        _comparison("genre", Comparator.EQ, "comedy")
    ) == {"genre": {"$eq": "comedy"}}
