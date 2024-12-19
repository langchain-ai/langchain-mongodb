"""Test MongoDB Atlas Vector Search functionality."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest  # type: ignore[import-not-found]
from bson import ObjectId
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.utils import oid_to_str

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

DB_NAME = "langchain_test_db"
INDEX_NAME = "langchain-test-index-vectorstores"
COLLECTION_NAME = "langchain_test_vectorstores"
DIMENSIONS = 5


@pytest.fixture(scope="module")
def collection(client: MongoClient) -> Collection:
    return client[DB_NAME][COLLECTION_NAME]


@pytest.fixture(scope="module")
def texts() -> List[str]:
    return [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]


@pytest.fixture(scope="module")
def trivial_embeddings() -> Embeddings:
    return ConsistentFakeEmbeddings(DIMENSIONS)


def test_delete(
    trivial_embeddings: Embeddings, collection: Any, texts: List[str]
) -> None:
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=trivial_embeddings,
        index_name="MATCHES_NOTHING",
    )
    clxn: Collection = vectorstore.collection
    clxn.delete_many({})
    assert clxn.count_documents({}) == 0
    ids = vectorstore.add_texts(texts)
    assert clxn.count_documents({}) == len(texts)

    deleted = vectorstore.delete(ids[-2:])
    assert deleted
    assert clxn.count_documents({}) == len(texts) - 2

    new_ids = vectorstore.add_texts(["Pigs eat stuff", "Pigs eat sandwiches"])
    assert set(new_ids).intersection(set(ids)) == set()  # new ids will be unique.
    assert isinstance(new_ids, list)
    assert all(isinstance(i, str) for i in new_ids)
    assert len(new_ids) == 2
    assert clxn.count_documents({}) == 4


def test_add_texts(
    trivial_embeddings: Embeddings,
    collection: Collection,
    texts: List[str],
) -> None:
    """Tests API of add_texts, focussing on id treatment

    Warning: This is slow because of the number of cases
    """
    metadatas: List[Dict[str, Any]] = [
        {"a": 1},
        {"b": 1},
        {"c": 1},
        {"d": 1, "e": 2},
    ]

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=collection,
        embedding=trivial_embeddings,
        index_name=INDEX_NAME,
    )
    vectorstore.delete()

    # Case 1. Add texts without ids
    provided_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
    all_docs = list(vectorstore._collection.find({}))
    assert all("_id" in doc for doc in all_docs)
    docids = set(doc["_id"] for doc in all_docs)
    assert all(isinstance(_id, ObjectId) for _id in docids)  #
    assert set(provided_ids) == set(oid_to_str(oid) for oid in docids)

    # Case 2: Test Document.metadata looks right. i.e. contains _id
    search_res = vectorstore.similarity_search_with_score("sandwich", k=1)
    doc, score = search_res[0]
    assert "_id" in doc.metadata

    # Case 3: Add new ids that are 24-char hex strings
    hex_ids = [oid_to_str(ObjectId()) for _ in range(2)]
    hex_texts = ["Text for hex_id"] * len(hex_ids)
    out_ids = vectorstore.add_texts(texts=hex_texts, ids=hex_ids)
    assert set(out_ids) == set(hex_ids)
    assert collection.count_documents({}) == len(texts) + len(hex_texts)
    assert all(
        isinstance(doc["_id"], ObjectId) for doc in vectorstore._collection.find({})
    )

    # Case 4: Add new ids that cannot be cast to ObjectId
    #   - We can still index and search on them
    str_ids = ["Sandwiches are beautiful,", "..sandwiches are fine."]
    str_texts = str_ids  # No reason for them to differ
    out_ids = vectorstore.add_texts(texts=str_texts, ids=str_ids)
    assert set(out_ids) == set(str_ids)
    assert collection.count_documents({}) == 8
    res = vectorstore.similarity_search("sandwich", k=8)
    assert any(str_ids[0] in doc.metadata["_id"] for doc in res)

    # Case 5: Test adding in multiple batches
    batch_size = 2
    batch_ids = [oid_to_str(ObjectId()) for _ in range(2 * batch_size)]
    batch_texts = [f"Text for batch text {i}" for i in range(2 * batch_size)]
    out_ids = vectorstore.add_texts(
        texts=batch_texts, ids=batch_ids, batch_size=batch_size
    )
    assert set(out_ids) == set(batch_ids)
    assert collection.count_documents({}) == 12

    # Case 6: _ids in metadata
    collection.delete_many({})
    # 6a. Unique _id in metadata, but ids=None
    # Will be added as if ids kwarg provided
    i = 0
    n = len(texts)
    assert len(metadatas) == n
    _ids = [str(i) for i in range(n)]
    for md in metadatas:
        md["_id"] = _ids[i]
        i += 1
    returned_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
    assert returned_ids == ["0", "1", "2", "3"]
    assert set(d["_id"] for d in vectorstore._collection.find({})) == set(_ids)

    # 6b. Unique "id", not "_id", but ids=None
    # New ids will be assigned
    i = 1
    for md in metadatas:
        md.pop("_id")
        md["id"] = f"{1}"
        i += 1
    returned_ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
    assert len(set(returned_ids).intersection(set(_ids))) == 0


def test_add_documents(
    collection: Collection,
    trivial_embeddings: Embeddings,
) -> None:
    """Tests add_documents.

    Note: Does not need indexes so no need to use patient patched vectorstore."""
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=trivial_embeddings,
        index_name="MATCHES_NOTHING",
    )
    vectorstore.collection.delete_many({})
    # Case 1: No ids
    n_docs = 10
    batch_size = 3
    docs = [
        Document(page_content=f"document {i}", metadata={"i": i}) for i in range(n_docs)
    ]
    result_ids = vectorstore.add_documents(docs, batch_size=batch_size)
    assert len(result_ids) == n_docs
    assert collection.count_documents({}) == n_docs

    # Case 2: ids
    collection.delete_many({})
    n_docs = 10
    batch_size = 3
    docs = [
        Document(page_content=f"document {i}", metadata={"i": i}) for i in range(n_docs)
    ]
    ids = [str(i) for i in range(n_docs)]
    result_ids = vectorstore.add_documents(docs, ids, batch_size=batch_size)
    assert len(result_ids) == n_docs
    assert set(ids) == set(collection.distinct("_id"))

    # Case 3: Single batch
    collection.delete_many({})
    n_docs = 3
    batch_size = 10
    docs = [
        Document(page_content=f"document {i}", metadata={"i": i}) for i in range(n_docs)
    ]
    ids = [str(i) for i in range(n_docs)]
    result_ids = vectorstore.add_documents(docs, ids, batch_size=batch_size)
    assert len(result_ids) == n_docs
    assert set(ids) == set(collection.distinct("_id"))
