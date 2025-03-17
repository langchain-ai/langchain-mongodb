import os
from typing import List

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

from tests.utils import CONNECTION_STRING


@pytest.fixture
def technical_report_pages() -> List[Document]:
    """Returns a Document for each of the 100 pages of a GPT-4 Technical Report"""
    loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
    pages = loader.load()
    return pages


@pytest.fixture
def client() -> MongoClient:
    return MongoClient(CONNECTION_STRING)


@pytest.fixture
def embedding() -> Embeddings:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
            model="text-embedding-3-small",
        )

    return OllamaEmbeddings(model="all-minilm:l6-v2")


@pytest.fixture
def dimensions() -> int:
    if os.environ.get("OPENAI_API_KEY"):
        return 1536
    return 384
