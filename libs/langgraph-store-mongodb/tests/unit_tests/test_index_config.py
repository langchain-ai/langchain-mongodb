import pytest
from langchain_mongodb.embeddings import AutoEmbeddings

from langgraph.store.mongodb import create_vector_index_config


def test_create_vector_index_config_with_autoembeddings() -> None:
    config = create_vector_index_config(
        embed=AutoEmbeddings("voyage-4"),
        fields=["question"],
    )

    assert config["dims"] == -1
    assert config["relevance_score_fn"] is None


def test_create_vector_index_config_with_manual_embeddings() -> None:
    config = create_vector_index_config(1536, "openai:text-embedding-3-small")

    assert config["dims"] == 1536
    assert config["relevance_score_fn"] == "cosine"


def test_create_vector_index_config_requires_manual_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="dims is required"):
        create_vector_index_config(embed="openai:text-embedding-3-small")


def test_create_vector_index_config_rejects_autoembedding_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        create_vector_index_config(
            dims=1024,
            embed=AutoEmbeddings("voyage-4"),
        )


def test_create_vector_index_config_rejects_autoembedding_similarity() -> None:
    with pytest.raises(ValueError, match="similarity"):
        create_vector_index_config(
            embed=AutoEmbeddings("voyage-4"),
            relevance_score_fn="cosine",
        )
