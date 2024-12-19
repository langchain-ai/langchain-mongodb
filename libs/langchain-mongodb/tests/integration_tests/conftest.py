import os
from typing import List

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pymongo import MongoClient


@pytest.fixture(scope="session")
def technical_report_pages() -> List[Document]:
    """Returns a Document for each of the 100 pages of a GPT-4 Technical Report"""
    loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
    pages = loader.load()
    return pages


@pytest.fixture(scope="session")
def connection_string() -> str:
    return os.environ["MONGODB_URI"]


@pytest.fixture(scope="session")
def client(connection_string: str) -> MongoClient:
    return MongoClient(connection_string)
