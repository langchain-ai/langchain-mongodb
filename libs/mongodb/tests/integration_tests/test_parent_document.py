import os
from importlib.metadata import version
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

from langchain_mongodb.docstores import MongoDBDocStore
from langchain_mongodb.index import create_vector_search_index
from langchain_mongodb.retrievers import (
    MongoDBAtlasParentDocumentRetriever,
)

from ..utils import ConsistentFakeEmbeddings, PatchedMongoDBAtlasVectorSearch

CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_parent_document_combined"
VECTOR_INDEX_NAME = "langchain-test-parent-document-vector-index"
EMBEDDING_FIELD = "embedding"
TEXT_FIELD = "page_content"
DIMENSIONS = 1536
SIMILARITY = "cosine"
TIMEOUT = 60.0


@pytest.fixture
def embedding_model() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    try:
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
            model="text-embedding-3-small",
        )
    except Exception:
        return ConsistentFakeEmbeddings(DIMENSIONS)


def test_1clxn_retriever(
    technical_report_pages: List[Document], embedding_model: Embeddings
) -> None:
    # Setup
    client: MongoClient = MongoClient(
        CONNECTION_STRING,
        driver=DriverInfo(name="langchain", version=version("langchain-mongodb")),
    )
    db = client[DB_NAME]
    combined_clxn = db[COLLECTION_NAME]
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
    # Clean up
    combined_clxn.delete_many({})
    # Create Search Index if it doesn't exist
    sixs = list(combined_clxn.list_search_indexes())
    if len(sixs) == 0:
        create_vector_search_index(
            collection=combined_clxn,
            index_name=VECTOR_INDEX_NAME,
            dimensions=DIMENSIONS,
            path=EMBEDDING_FIELD,
            similarity=SIMILARITY,
            wait_until_complete=TIMEOUT,
        )
    # Create Vector and Doc Stores
    vectorstore = PatchedMongoDBAtlasVectorSearch(
        collection=combined_clxn,
        embedding=embedding_model,
        index_name=VECTOR_INDEX_NAME,
        text_key=TEXT_FIELD,
        embedding_key=EMBEDDING_FIELD,
        relevance_score_fn=SIMILARITY,
    )
    docstore = MongoDBDocStore(collection=combined_clxn, text_key=TEXT_FIELD)
    #  Combine into a ParentDocumentRetriever
    retriever = MongoDBAtlasParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    )
    # Add documents (splitting, creating embedding, adding to vectorstore and docstore)
    retriever.add_documents(technical_report_pages)
    # invoke the retriever with a query
    question = "What percentage of the Uniform Bar Examination can GPT4 pass?"
    responses = retriever.invoke(question)

    assert len(responses) == 3
    assert all("GPT-4" in doc.page_content for doc in responses)
    assert {4, 5, 29} == set(doc.metadata["page"] for doc in responses)
