from json import dumps, loads
from typing import Any, List, Optional

import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.pipelines import rerank_stage

from ..utils import DB_NAME, ConsistentFakeEmbeddings, MockCollection

INDEX_NAME = "langchain-test-index"
NAMESPACE = f"{DB_NAME}.langchain_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


def get_collection() -> MockCollection:
    return MockCollection()


@pytest.fixture()
def collection() -> MockCollection:
    return get_collection()


@pytest.fixture(scope="module")
def embedding_openai() -> Embeddings:
    return ConsistentFakeEmbeddings()


def test_initialization(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test initialization of vector store class"""
    assert MongoDBAtlasVectorSearch(collection, embedding_openai)


def test_init_from_texts(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test from_texts operation on an empty list"""
    assert MongoDBAtlasVectorSearch.from_texts(
        [], embedding_openai, collection=collection
    )


class TestMongoDBAtlasVectorSearch:
    @classmethod
    def setup_class(cls) -> None:
        # ensure the test collection is empty
        collection = get_collection()
        assert collection.count_documents({}) == 0  # type: ignore[index]

    @classmethod
    def teardown_class(cls) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        collection = get_collection()
        # delete all the documents in the collection
        collection.delete_many({})  # type: ignore[index]

    def _validate_search(
        self,
        vectorstore: MongoDBAtlasVectorSearch,
        collection: MockCollection,
        search_term: str = "sandwich",
        page_content: str = "What is a sandwich?",
        metadata: Optional[Any] = 1,
    ) -> None:
        collection._aggregate_result = list(
            filter(
                lambda x: search_term.lower() in x[vectorstore._text_key].lower(),
                collection._data,
            )
        )
        output = vectorstore.similarity_search("", k=1)
        assert output[0].page_content == page_content
        assert output[0].metadata.get("c") == metadata
        # Validate the ObjectId provided is json serializable
        assert loads(dumps(output[0].page_content)) == output[0].page_content
        assert loads(dumps(output[0].metadata)) == output[0].metadata
        assert isinstance(output[0].metadata["_id"], str)

    def test_from_documents(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        # TODO: test how DIMS is handled here.
        self._validate_search(
            vectorstore, collection, metadata=documents[2].metadata["c"]
        )
        vectorstore.close()
        assert collection.database.client.is_closed

    def test_from_texts(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        self._validate_search(vectorstore, collection, metadata=None)

    def test_from_texts_with_metadatas(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        self._validate_search(vectorstore, collection, metadata=metadatas[2]["c"])

    def test_from_texts_with_metadatas_and_pre_filter(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "The fence is purple.",
        ]
        metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding_openai,
            metadatas=metadatas,
            collection=collection,
            index_name=INDEX_NAME,
        )
        collection._aggregate_result = list(
            filter(
                lambda x: "sandwich" in x[vectorstore._text_key].lower()
                and x.get("c") < 0,
                collection._data,
            )
        )
        output = vectorstore.similarity_search(
            "Sandwich", k=1, pre_filter={"range": {"lte": 0, "path": "c"}}
        )
        assert output == []

    def test_mmr(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        texts = ["foo", "foo", "fou", "foy"]
        vectorstore = MongoDBAtlasVectorSearch.from_texts(
            texts,
            embedding=embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        query = "foo"
        self._validate_search(
            vectorstore,
            collection,
            search_term=query[0:2],
            page_content=query,
            metadata=None,
        )
        output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
        assert len(output) == len(texts)
        assert output[0].page_content == "foo"
        assert output[1].page_content != "foo"

    def test_auto_create_index(
        self, embedding_openai: Embeddings, collection: MockCollection
    ) -> None:
        # Explicit auto_create_index
        assert len(collection._search_indexes) == 0
        _ = MongoDBAtlasVectorSearch(
            embedding=embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
            auto_create_index=True,
        )
        assert len(collection._search_indexes) == 1

        # Explicit dimensions
        collection._search_indexes = []
        _ = MongoDBAtlasVectorSearch(
            embedding=embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
            dimensions=10,
        )
        assert len(collection._search_indexes) == 1

        # Does not auto-create
        collection._search_indexes = []
        _ = MongoDBAtlasVectorSearch(
            embedding=embedding_openai,
            collection=collection,
            index_name=INDEX_NAME,
        )
        assert len(collection._search_indexes) == 0


class MockCollectionCapturePipeline(MockCollection):
    """MockCollection that records the last pipeline passed to aggregate()."""

    def __init__(self) -> None:
        super().__init__()
        self.last_pipeline: List[Any] = []

    def aggregate(self, pipeline, *args, **kwargs) -> List[Any]:  # type: ignore[override]
        self.last_pipeline = list(pipeline)
        return super().aggregate(pipeline, *args, **kwargs)


@pytest.fixture()
def capturing_collection() -> MockCollectionCapturePipeline:
    return MockCollectionCapturePipeline()


def test_rerank_stage_minimal() -> None:
    stages = rerank_stage(query="cats", path="text", num_docs_to_rerank=10)
    assert len(stages) == 2
    rerank = stages[0]["$rerank"]
    assert rerank["query"] == {"text": "cats"}
    assert rerank["path"] == "text"
    assert rerank["numDocsToRerank"] == 10
    assert "model" not in rerank
    assert stages[1] == {
        "$set": {"score": {"$meta": "score"}, "rerankScore": {"$meta": "score"}}
    }


def test_rerank_stage_with_model_and_list_path() -> None:
    stages = rerank_stage(
        query="cats", path=["title", "body"], num_docs_to_rerank=25, model="rerank-2.5"
    )
    rerank = stages[0]["$rerank"]
    assert rerank["path"] == ["title", "body"]
    assert rerank["model"] == "rerank-2.5"


def test_vectorstore_pipeline_includes_rerank(
    capturing_collection: MockCollectionCapturePipeline, embedding_openai: Embeddings
) -> None:
    vs = MongoDBAtlasVectorSearch(
        capturing_collection, embedding_openai, index_name=INDEX_NAME
    )
    vs.similarity_search("cats", k=3, rerank_path="text", num_docs_to_rerank=10)
    pipeline = capturing_collection.last_pipeline
    stage_keys = [list(s.keys())[0] for s in pipeline]
    assert "$vectorSearch" in stage_keys
    assert "$rerank" in stage_keys
    rerank = pipeline[stage_keys.index("$rerank")]["$rerank"]
    assert rerank["query"] == {"text": "cats"}
    assert rerank["path"] == "text"
    assert rerank["numDocsToRerank"] == 10


def test_vectorstore_pipeline_limit_added_when_n_to_rerank_gt_k(
    capturing_collection: MockCollectionCapturePipeline, embedding_openai: Embeddings
) -> None:
    vs = MongoDBAtlasVectorSearch(
        capturing_collection, embedding_openai, index_name=INDEX_NAME
    )
    vs.similarity_search("cats", k=3, rerank_path="text", num_docs_to_rerank=20)
    pipeline = capturing_collection.last_pipeline
    stage_keys = [list(s.keys())[0] for s in pipeline]
    assert "$limit" in stage_keys
    assert pipeline[stage_keys.index("$limit")]["$limit"] == 3


def test_vectorstore_rerank_requires_query_text(
    capturing_collection: MockCollectionCapturePipeline, embedding_openai: Embeddings
) -> None:
    vs = MongoDBAtlasVectorSearch(
        capturing_collection, embedding_openai, index_name=INDEX_NAME
    )
    with pytest.raises(ValueError, match="rerank_query"):
        vs._similarity_search_with_score([0.1] * 10, k=3, rerank_path="text")
