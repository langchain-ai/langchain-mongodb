import os
from typing import Any, Dict, List, Optional, Union, Generator, Iterable
from langchain_core.embeddings import Embeddings
from pymongo.collection import Collection
from pymongo.results import InsertManyResult, DeleteResult
from pymongo.operations import SearchIndexModel
from bson import ObjectId
from time import sleep
from time import monotonic
from langchain_mongodb import MongoDBAtlasVectorSearch

TIMEOUT = 120
INTERVAL = 0.5
CONNECTION_STRING = os.environ.get("MONGODB_URI", "")


DB_NAME = "langchain_test_db"

class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [1.0] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

class PatchedMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = super().bulk_embed_and_insert_texts(texts, metadatas, ids)
        n_docs = self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if (
                len(self.similarity_search("sandwich", k=n_docs, oversampling_factor=1))
                == n_docs
            ):
                return ids_inserted
            else:
                sleep(INTERVAL)
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")

    def _similarity_search_with_score(self, query_vector, **kwargs):
        # Remove the _ids for testing purposes.
        docs = super()._similarity_search_with_score(query_vector, **kwargs)
        for doc, _ in docs:
            del doc.metadata["_id"]
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        ret = super().delete(ids, **kwargs)
        n_docs = self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if (
                len(
                    self.similarity_search(
                        "sandwich", k=max(n_docs, 1), oversampling_factor=1
                    )
                )
                == n_docs
            ):
                return ret
            else:
                sleep(INTERVAL)
        raise TimeoutError(f"Failed to delete and index texts in {TIMEOUT}s.")

class MockDatabase:
    name = "test"
    def __init__(self, client=None):
        super().__init__()
        self.client = client or MockClient()
    def list_collection_names(self, authorizedCollections: bool = True) -> list[str]:
        return ["test"]
    def __getitem__(self, key: str) -> Any:
        return MockCollection(self)

class MockClient:
    is_closed = False
    def __getitem__(self, key: str) -> Any:
        return MockDatabase(self)
    def close(self):
        self.is_closed = True
    def append_metadata(self, metadata):
        pass

class MockCollection(Collection):
    """Mocked Mongo Collection"""
    _aggregate_result: List[Any]
    _insert_result: Optional[InsertManyResult]
    _data: List[Any]
    _simulate_cache_aggregation_query: bool
    def __init__(self, database: MockDatabase | None = None) -> None:
        super().__init__()
        self._data = []
        self._name = "test"
        self.is_closed = False
        self._aggregate_result = []
        self._insert_result = None
        self._search_indexes = []
        self._simulate_cache_aggregation_query = False
        self._database = database or MockDatabase()  # type:ignore[assignment]
    @property
    def database(self):
        return self._database
    def close(self):
        self.is_closed = True
    def list_search_indexes(self, name=None, session=None, comment=None, **kwargs):
        return [
            dict(name=idx.document["name"], status="READY")
            for idx in self._search_indexes
        ]
    def create_search_index(self, model, session=None, comment=None, **kwargs):
        if not isinstance(model, SearchIndexModel):
            model = SearchIndexModel(model, name=f"test{len(self._search_indexes)}")
        self._search_indexes.append(model)
    def delete_many(self, *args, **kwargs) -> DeleteResult:  # type: ignore
        old_len = len(self._data)
        self._data = []
        return DeleteResult({"n": old_len}, acknowledged=True)
    def insert_many(self, to_insert: List[Any], *args, **kwargs) -> InsertManyResult:  # type: ignore
        mongodb_inserts = [
            {"_id": ObjectId(), "score": 1, **insert} for insert in to_insert
        ]
        self._data.extend(mongodb_inserts)
        return self._insert_result or InsertManyResult(
            [k["_id"] for k in mongodb_inserts], acknowledged=True
        )
    def insert_one(self, to_insert: Any, *args, **kwargs) -> Any:  # type: ignore
        return self.insert_many([to_insert])
    def find_one(self, find_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # type: ignore
        find = self.find(find_query) or [None]  # type: ignore
        return find[0]
    def find(self, find_query: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:  # type: ignore
        def _is_match(item: Dict[str, Any]) -> bool:
            for key, match_val in find_query.items():
                if item.get(key) != match_val:
                    return False
            return True
        return [document for document in self._data if _is_match(document)]
    def update_one(  # type: ignore
        self,
        find_query: Dict[str, Any],
        options: Dict[str, Any],
        *args: Any,
        upsert=True,
        **kwargs: Any,
    ) -> None:  # type: ignore
        result = self.find_one(find_query)
        set_options = options.get("$set", {})
        if result:
            result.update(set_options)
        elif upsert:
            self._data.append({**find_query, **set_options})
    def __repr__(self) -> str:
        return "MockCollection"
