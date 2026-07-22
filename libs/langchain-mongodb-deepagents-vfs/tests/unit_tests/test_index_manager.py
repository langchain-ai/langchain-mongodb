"""Unit tests for IndexManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_mongodb_deepagents_vfs.index_manager import IndexManager


@pytest.mark.unit
class TestIndexManagerNonAtlas:
    def test_ensure_indexes_on_mongomock_does_not_raise(self, mongo_collection):
        mgr = IndexManager(mongo_collection, embedding_dimensions=1024)
        # Should complete without raising even though mongomock != Atlas
        mgr.ensure_indexes()

    def test_wait_until_queryable_returns_true_on_non_atlas(self, mongo_collection):
        mgr = IndexManager(mongo_collection)
        result = mgr.wait_until_queryable(timeout=1)
        assert result is True

    def test_native_index_created(self, mongo_collection):
        mgr = IndexManager(mongo_collection)
        mgr.ensure_indexes()
        indexes = {idx["name"] for idx in mongo_collection.list_indexes()}
        assert "source_path_chunk_index_unique" in indexes

    def test_ensure_indexes_idempotent(self, mongo_collection):
        mgr = IndexManager(mongo_collection)
        mgr.ensure_indexes()
        mgr.ensure_indexes()  # second call must not raise


@pytest.mark.unit
class TestIndexManagerAtlasSimulated:
    """Simulate an Atlas-like collection with patched list_search_indexes."""

    def _atlas_collection(self):
        col = MagicMock()
        col.list_search_indexes.return_value = []
        col.create_index.return_value = None
        col.create_search_index.return_value = None
        col.list_indexes.return_value = iter([])
        return col

    def test_vector_and_fulltext_index_created_on_atlas(self):
        col = self._atlas_collection()
        mgr = IndexManager(col, embedding_dimensions=1024)
        mgr.ensure_indexes()
        assert col.create_search_index.call_count == 2

    def test_indexes_not_recreated_if_already_exist(self):
        col = self._atlas_collection()
        col.list_search_indexes.return_value = [
            {
                "name": "vector_search_embedding",
                "queryable": True,
                "status": "READY",
                "latestDefinition": {"mappings": {"fields": {}}},
            },
            {
                "name": "fulltext_search_content_filename",
                "queryable": True,
                "status": "READY",
                "latestDefinition": {
                    "mappings": {
                        "fields": {
                            "filename": {"type": "string", "analyzer": "lucene.keyword"}
                        }
                    }
                },
            },
        ]
        mgr = IndexManager(col)
        mgr.ensure_indexes()
        col.create_search_index.assert_not_called()

    def test_wait_until_queryable_returns_true_when_ready(self):
        col = self._atlas_collection()
        col.list_search_indexes.return_value = [
            {"name": "vector_search_embedding", "queryable": True, "status": "READY"},
            {
                "name": "fulltext_search_content_filename",
                "queryable": True,
                "status": "READY",
            },
        ]
        mgr = IndexManager(col)
        mgr._is_atlas = True  # force Atlas mode
        result = mgr.wait_until_queryable(timeout=5)
        assert result is True

    def test_wait_until_queryable_times_out(self):
        col = self._atlas_collection()
        col.list_search_indexes.return_value = [
            {"name": "vector_search_embedding", "queryable": False},
        ]
        mgr = IndexManager(col)
        mgr._is_atlas = True
        result = mgr.wait_until_queryable(timeout=1)
        assert result is False
