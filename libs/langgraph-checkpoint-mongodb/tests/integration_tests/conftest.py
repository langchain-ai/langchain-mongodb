import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


@pytest.fixture(scope="session")
def embedding() -> Embeddings:
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIEmbeddings(model="text-embedding-3-small")
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore # noqa
            model="text-embedding-3-small",
        )
    return OllamaEmbeddings(model="all-minilm:l6-v2")


@pytest.fixture(scope="session")
def dimensions() -> int:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return 1536
    return 384
