from __future__ import annotations

import os
from copy import deepcopy
from time import monotonic, sleep
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

from bson import ObjectId
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import model_validator
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.results import BulkWriteResult, DeleteResult, InsertManyResult

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import (
    make_serializable,
    oid_to_str,
    str_to_oid,
)

TIMEOUT = 120
INTERVAL = 0.5
CONNECTION_STRING = os.environ.get("MONGODB_URI", "")
DEFAULT_INSERT_BATCH_SIZE = 100

DB_NAME = "langchain_test_db"


def create_database() -> MongoDBDatabase:
    client = MongoClient(CONNECTION_STRING)
    coll = client[DB_NAME]["test"]
    coll.delete_many({})
    coll.insert_one({})
    return MongoDBDatabase(client, DB_NAME)


def create_llm() -> BaseChatModel:
    if os.environ.get("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", timeout=60, cache=False)
    return ChatOllama(model="llama3:8b", cache=False)


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
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")


class PatchedAsyncMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    async def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = await self.bulk_embed_and_insert_texts_async(
            texts, metadatas, ids
        )
        n_docs = await self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            search_results = await self.asimilarity_search(
                "sandwich", k=n_docs, oversampling_factor=1
            )
            if len(search_results) == n_docs:
                return ids_inserted
            else:
                sleep(INTERVAL)
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")

    async def bulk_embed_and_insert_texts_async(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Bulk insert single batch of texts, embeddings, and optionally ids.

        See add_texts for additional details.
        """
        if not texts:
            return []
        # Compute embedding vectors
        embeddings = self._embedding.embed_documents(list(texts))
        if not ids:
            ids = [str(ObjectId()) for _ in range(len(list(texts)))]
        docs = [
            {
                "_id": str_to_oid(i),
                self._text_key: t,
                self._embedding_key: embedding,
                **m,
            }
            for i, t, m, embedding in zip(ids, texts, metadatas, embeddings)
        ]
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        # insert the documents in MongoDB Atlas
        result = await self._collection.bulk_write(operations)
        assert result.upserted_ids is not None
        return [oid_to_str(_id) for _id in result.upserted_ids.values()]

    async def _similarity_search_with_score(self, query_vector, **kwargs):
        # Remove the _ids for testing purposes.
        docs = await self._similarity_search_with_score_async(query_vector, **kwargs)
        for doc, _ in docs:
            del doc.metadata["_id"]
        return docs

    async def _similarity_search_with_score_async(
        self,
        query_vector: List[float],
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Core search routine. See external methods for details."""

        # Atlas Vector Search, potentially with filter
        pipeline = [
            vector_search_stage(
                query_vector,
                self._embedding_key,
                self._index_name,
                k,
                pre_filter,
                oversampling_factor,
                **kwargs,
            ),
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        # Remove embeddings unless requested.
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})
        # Post-processing
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        # Execution
        cursor = await self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        # Format
        async for res in cursor:
            if self._text_key not in res:
                continue
            text = res.pop(self._text_key)
            score = res.pop("score")
            make_serializable(res)
            docs.append(
                (Document(page_content=text, metadata=res, id=res["_id"]), score)
            )
        return docs

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Async return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await self.similarity_search(query, k=k, **kwargs)

    async def delete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        ret = await super().delete(ids, **kwargs)
        n_docs = await self.collection.count_documents({})
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if (
                len(
                    await self.asimilarity_search(
                        "sandwich", k=max(n_docs, 1), oversampling_factor=1
                    )
                )
                == n_docs
            ):
                return ret
            else:
                sleep(INTERVAL)
        raise TimeoutError(f"Failed to embed, insert, and index texts in {TIMEOUT}s.")

    async def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids in add_texts.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of IDs of the added texts.
        """
        n_docs = len(documents)
        if ids:
            assert len(ids) == n_docs, "Number of ids must equal number of documents."
        else:
            ids = [doc.id or str(ObjectId()) for doc in documents]
        result_ids = []
        start = 0
        for end in range(batch_size, n_docs + batch_size, batch_size):
            texts, metadatas = zip(
                *[(doc.page_content, doc.metadata) for doc in documents[start:end]]
            )
            result_ids.extend(
                await self.bulk_embed_and_insert_texts(
                    texts=texts, metadatas=metadatas, ids=ids[start:end]
                )
            )
            start = end
        return result_ids

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_scores: bool = False,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Document]:  # noqa: E501
        """Return MongoDB documents most similar to the given query.

        Atlas Vector Search eliminates the need to run a separate
        search system alongside your database.

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: List of MQL match expressions comparing an indexed field
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                to filter/process results after $vectorSearch.
            oversampling_factor: Multiple of k used when generating number of candidates
                at each step in the HNSW Vector Search,
            include_scores: If True, the query score of each result
                will be included in metadata.
            include_embeddings: If True, the embedding vector of each result
                will be included in metadata.
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """
        docs_and_scores = await self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            oversampling_factor=oversampling_factor,
            include_embeddings=include_embeddings,
            **kwargs,
        )

        if include_scores:
            async for doc, score in docs_and_scores:
                doc.metadata["score"] = score
        return [doc for doc, _ in docs_and_scores]


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


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = "fake response"
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"key": "fake"}


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def check_queries_required(cls, values: dict) -> dict:
        if values.get("sequential_response") and not values.get("queries"):
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return values

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


class MockClient:
    is_closed = False

    def __getitem__(self, key: str) -> Any:
        return MockDatabase(self)

    def close(self):
        self.is_closed = True


class MockDatabase:
    name = "test"

    def __init__(self, client=None):
        self.client = client or MockClient()

    def list_collection_names(self) -> list[str]:
        return ["test"]

    def __getitem__(self, key: str) -> Any:
        return MockCollection(self)


class MockCollection(Collection):
    """Mocked Mongo Collection"""

    _aggregate_result: List[Any]
    _insert_result: Optional[InsertManyResult]
    _data: List[Any]
    _simulate_cache_aggregation_query: bool

    def __init__(self, database: MockDatabase | None = None) -> None:
        self._data = []
        self.is_closed = False
        self._aggregate_result = []
        self._insert_result = None
        self._simulate_cache_aggregation_query = False
        self._database = database or MockDatabase()  # type:ignore[assignment]

    @property
    def database(self):
        return self._database

    def close(self):
        self.is_closed = True

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

    def _execute_cache_aggregation_query(self, *args, **kwargs) -> List[Dict[str, Any]]:  # type: ignore
        """Helper function only to be used for MongoDBAtlasSemanticCache Testing

        Returns:
            List[Dict[str, Any]]: Aggregation query result
        """
        pipeline: List[Dict[str, Any]] = args[0]
        params = pipeline[0]["$vectorSearch"]
        embedding = params["queryVector"]
        # Assumes MongoDBAtlasSemanticCache.LLM == "llm_string"
        llm_string = params["filter"][MongoDBAtlasSemanticCache.LLM]["$eq"]

        acc = []
        for document in self._data:
            if (
                document.get("embedding") == embedding
                and document.get(MongoDBAtlasSemanticCache.LLM) == llm_string
            ):
                acc.append(document)
        return acc

    def bulk_write(self, requests, **kwargs):
        upserted = []
        for ind, request in enumerate(requests):
            doc = request._doc
            doc["score"] = "foo"
            self._data.append(doc)
            upserted.append(dict(index=ind, _id=doc["_id"]))
        return BulkWriteResult(dict(upserted=upserted), True)

    def aggregate(self, *args, **kwargs) -> List[Any]:  # type: ignore
        if self._simulate_cache_aggregation_query:
            return deepcopy(self._execute_cache_aggregation_query(*args, **kwargs))
        return deepcopy(self._aggregate_result)

    def count_documents(self, *args, **kwargs) -> int:  # type: ignore
        return len(self._data)

    def __repr__(self) -> str:
        return "MockCollection"
