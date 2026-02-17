import os
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo_search_utils import drop_vector_search_index

from langchain_mongodb.docstores import MongoDBDocStore
from langchain_mongodb.embeddings import AutoEmbeddings
from langchain_mongodb.index import create_vector_search_index
from langchain_mongodb.retrievers import (
    MongoDBAtlasParentDocumentRetriever,
)
from langchain_mongodb.utils import DRIVER_METADATA

from ..utils import (
    AUTOEMBED_MODEL,
    CONNECTION_STRING,
    DB_NAME,
    PatchedMongoDBAtlasVectorSearch,
)

COLLECTION_NAME = "langchain_test_parent_document_combined"
COLLECTION_NAME_AUTO = "langchain_test_parent_document_combined_auto"
COLLECTION_NAME_AUTOCREATE = "langchain_test_parent_document_autocreate"
COLLECTION_NAME_FROM_CS_AUTOCREATE = "langchain_test_parent_document_from_cs_autocreate"
VECTOR_INDEX_NAME = "langchain-test-parent-document-vector-index"
VECTOR_INDEX_NAME_AUTO = "langchain-test-parent-document-vector-index-auto"
SEARCH_INDEX_NAME = "text_index"
EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "page_content"
SIMILARITY = "cosine"
TIMEOUT = 60.0


@pytest.fixture
def embedding_param(request, embedding):
    if request.param == "auto":
        if not os.environ.get("COMMUNITY_WITH_SEARCH", ""):
            raise pytest.skip("Only run if COMMUNITY_WITH_SEARCH is set")
        return AutoEmbeddings(model=AUTOEMBED_MODEL)
    return embedding


@pytest.mark.parametrize("embedding_param", ["auto", "manual"], indirect=True)
def test_1clxn_retriever(
    technical_report_pages: List[Document],
    embedding_param: Embeddings,
    dimensions: int,
) -> None:
    # Setup
    client: MongoClient = MongoClient(
        CONNECTION_STRING,
        driver=DRIVER_METADATA,
    )
    db = client[DB_NAME]

    # Use different collections for auto vs manual embeddings
    if isinstance(embedding_param, AutoEmbeddings):
        collection_name = COLLECTION_NAME_AUTO
        dimensions = -1
        auto_embedding_model = AUTOEMBED_MODEL
        embedding_key = None
        relevance_score_fn = None
        path = TEXT_FIELD  # Use the same field name as text_key
        similarity = None
        index_name = VECTOR_INDEX_NAME_AUTO
    else:
        collection_name = COLLECTION_NAME
        auto_embedding_model = None
        embedding_key = EMBEDDING_FIELD
        relevance_score_fn = SIMILARITY
        path = EMBEDDING_FIELD
        similarity = SIMILARITY
        index_name = VECTOR_INDEX_NAME

    combined_clxn = db[collection_name]
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    # Clean up
    combined_clxn.delete_many({})
    # Create Search Index if it doesn't exist
    sixs = list(combined_clxn.list_search_indexes())

    # Check if the specific index exists
    if not any([index_name == ix["name"] for ix in sixs]):
        create_vector_search_index(
            collection=combined_clxn,
            index_name=index_name,
            dimensions=dimensions,
            auto_embedding_model=auto_embedding_model,
            path=path,
            similarity=similarity,
            wait_until_complete=TIMEOUT,
        )
    # Create Vector and Doc Stores
    if isinstance(embedding_param, AutoEmbeddings):
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=combined_clxn,
            embedding=embedding_param,
            index_name=index_name,
            text_key=TEXT_FIELD,
            embedding_key=None,
            relevance_score_fn=None,
            dimensions=-1,
        )
    else:
        vectorstore = PatchedMongoDBAtlasVectorSearch(
            collection=combined_clxn,
            embedding=embedding_param,
            index_name=index_name,
            text_key=TEXT_FIELD,
            embedding_key=embedding_key,
            relevance_score_fn=relevance_score_fn,
        )
    docstore = MongoDBDocStore(collection=combined_clxn, text_key=TEXT_FIELD)
    #  Combine into a ParentDocumentRetriever
    retriever = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        search_kwargs={"top_k": 10},
    )
    # Add documents (splitting, creating embedding, adding to vectorstore and docstore)
    retriever.add_documents(technical_report_pages)
    # invoke the retriever with a query
    question = "What percentage of the Uniform Bar Examination can GPT4 pass?"
    responses = retriever.invoke(question)

    assert all("GPT-4" in doc.page_content for doc in responses)
    # Check that the expected pages are included (but allow for additional pages)
    pages = set(doc.metadata["page"] for doc in responses)
    expected_pages = {4, 5, 29}
    assert expected_pages.issubset(pages), (
        f"Expected pages {expected_pages} to be in {pages}"
    )
    client.close()


def test_parent_document_retriever_auto_create_index(
    client: MongoClient,
    embedding: Embeddings,
    dimensions: int,
) -> None:
    db = client[DB_NAME]
    combined_clxn = db[COLLECTION_NAME_AUTOCREATE]
    combined_clxn.delete_many({})

    if any(
        ix["name"] == SEARCH_INDEX_NAME for ix in combined_clxn.list_search_indexes()
    ):
        drop_vector_search_index(
            combined_clxn, SEARCH_INDEX_NAME, wait_until_complete=TIMEOUT
        )

    index_names_before = [ix["name"] for ix in combined_clxn.list_search_indexes()]
    assert SEARCH_INDEX_NAME not in index_names_before

    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=combined_clxn,
        embedding=embedding,
        index_name=VECTOR_INDEX_NAME,
        text_key=TEXT_FIELD,
        embedding_key=EMBEDDING_FIELD,
        relevance_score_fn=SIMILARITY,
        auto_create_index=False,
    )
    docstore = MongoDBDocStore(collection=combined_clxn, text_key=TEXT_FIELD)
    _ = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        auto_create_index=True,
        search_index_name=SEARCH_INDEX_NAME,
    )
    index_names_after = [ix["name"] for ix in combined_clxn.list_search_indexes()]
    assert SEARCH_INDEX_NAME in index_names_after


def test_parent_document_retriever_from_connection_string_auto_create_index(
    embedding: Embeddings,
) -> None:
    client = MongoClient(CONNECTION_STRING, driver=DRIVER_METADATA)
    db = client[DB_NAME]
    combined_clxn = db[COLLECTION_NAME_FROM_CS_AUTOCREATE]
    combined_clxn.delete_many({})

    if any(
        ix["name"] == SEARCH_INDEX_NAME for ix in combined_clxn.list_search_indexes()
    ):
        drop_vector_search_index(
            combined_clxn, SEARCH_INDEX_NAME, wait_until_complete=TIMEOUT
        )

    index_names_before = [ix["name"] for ix in combined_clxn.list_search_indexes()]
    assert SEARCH_INDEX_NAME not in index_names_before

    retriever = MongoDBAtlasParentDocumentRetriever.from_connection_string(
        CONNECTION_STRING,
        embedding,
        RecursiveCharacterTextSplitter(chunk_size=400),
        DB_NAME,
        collection_name=COLLECTION_NAME_FROM_CS_AUTOCREATE,
        auto_create_index=True,
        search_index_name=SEARCH_INDEX_NAME,
    )
    try:
        index_names_after = [
            ix["name"] for ix in retriever.vectorstore.collection.list_search_indexes()
        ]
        assert SEARCH_INDEX_NAME in index_names_after
    finally:
        retriever.close()
    client.close()
