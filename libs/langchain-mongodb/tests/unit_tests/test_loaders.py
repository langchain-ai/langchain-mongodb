from typing import Any, Dict, List, Optional

import pytest

from langchain_mongodb.loaders import MongoDBLoader

from ..utils import MockCollection

RAW_DOC = {"_id": "1", "title": "hello", "body": "world"}


class MockCollectionCaptureProjection(MockCollection):
    """MockCollection whose find() accepts a projection and records it."""

    def __init__(self) -> None:
        super().__init__()
        self.last_projection: Optional[Dict[str, Any]] = None

    def find(  # type: ignore[override]
        self, find_query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        self.last_projection = projection
        return [dict(document) for document in self._data]


@pytest.fixture()
def collection() -> MockCollectionCaptureProjection:
    collection = MockCollectionCaptureProjection()
    collection._data = [dict(RAW_DOC)]
    return collection


def test_load_without_field_names_uses_entire_document(
    collection: MockCollectionCaptureProjection,
) -> None:
    loader = MongoDBLoader(collection)
    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].page_content == str(RAW_DOC)
    assert documents[0].metadata == {"database": "test", "collection": "test"}
    assert collection.last_projection is None


def test_load_with_only_metadata_names_fetches_entire_document(
    collection: MockCollectionCaptureProjection,
) -> None:
    loader = MongoDBLoader(
        collection,
        metadata_names=["_id"],
        include_db_collection_in_metadata=False,
    )
    documents = loader.load()

    assert documents[0].page_content == str(RAW_DOC)
    assert documents[0].metadata == {"_id": "1"}
    assert collection.last_projection is None


def test_load_with_field_names_projects_fields(
    collection: MockCollectionCaptureProjection,
) -> None:
    loader = MongoDBLoader(
        collection,
        field_names=["title"],
        metadata_names=["_id"],
        include_db_collection_in_metadata=False,
    )
    documents = loader.load()

    assert documents[0].page_content == "hello"
    assert documents[0].metadata == {"_id": "1"}
    assert collection.last_projection == {"title": 1, "_id": 1}
