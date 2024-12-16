from typing import List

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def technical_report_pages() -> List[Document]:
    """Returns a Document for each of the 100 pages of a GPT-4 Technical Report"""
    loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
    pages = loader.load()
    return pages
