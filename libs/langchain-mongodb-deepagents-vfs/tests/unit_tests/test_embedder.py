"""Unit tests for the Embedder module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_mongodb_deepagents_vfs.dtypes import Chunk
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode


def make_chunks(n: int, dims: int = 1024) -> list[Chunk]:
    return [
        Chunk(source_path="test.txt", chunk_index=i, content=f"chunk {i}")
        for i in range(n)
    ]


@pytest.mark.unit
class TestEmbedderBatchSize:
    def test_single_batch(self):
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 1024]
        embedder = Embedder(model=mock_model, dimensions=1024, batch_size=100)
        chunks = make_chunks(1)
        result = embedder.embed_batch(chunks)
        assert len(result) == 1
        assert len(result[0]) == 1024
        mock_model.embed_documents.assert_called_once()

    def test_multiple_batches(self):
        mock_model = MagicMock()
        mock_model.embed_documents.side_effect = lambda texts: [
            [0.0] * 1024 for _ in texts
        ]
        embedder = Embedder(model=mock_model, dimensions=1024, batch_size=3)
        chunks = make_chunks(7)
        result = embedder.embed_batch(chunks)
        assert len(result) == 7
        # Should have been called ceil(7/3) = 3 times
        assert mock_model.embed_documents.call_count == 3

    def test_empty_chunks(self):
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = []
        embedder = Embedder(model=mock_model, dimensions=1024)
        result = embedder.embed_batch([])
        assert result == []
        mock_model.embed_documents.assert_not_called()


@pytest.mark.unit
class TestEmbedderDimensionValidation:
    def test_dimension_mismatch_raises(self):
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.0] * 512]  # wrong dims
        embedder = Embedder(model=mock_model, dimensions=1024)
        chunks = make_chunks(1)
        with pytest.raises(AdapterError) as exc_info:
            embedder.embed_batch(chunks)
        assert exc_info.value.code == ErrorCode.E4002_EMBEDDING_DIMENSION_MISMATCH


@pytest.mark.unit
class TestEmbedderRetry:
    def test_rate_limit_retries(self):
        mock_model = MagicMock()
        # Fail twice with rate-limit error, succeed on third
        call_count = {"n": 0}

        def side_effect(texts):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("429 rate limit exceeded")
            return [[0.0] * 1024 for _ in texts]

        mock_model.embed_documents.side_effect = side_effect
        embedder = Embedder(model=mock_model, dimensions=1024)

        with patch("time.sleep"):  # don't actually sleep in tests
            result = embedder.embed_batch(make_chunks(1))
        assert len(result) == 1
        assert call_count["n"] == 3

    def test_non_rate_limit_error_raises_immediately(self):
        mock_model = MagicMock()
        mock_model.embed_documents.side_effect = Exception("Internal server error")
        embedder = Embedder(model=mock_model, dimensions=1024)
        with pytest.raises(AdapterError) as exc_info:
            embedder.embed_batch(make_chunks(1))
        assert exc_info.value.code == ErrorCode.E4001_EMBEDDING_API_FAILED
        assert mock_model.embed_documents.call_count == 1  # No retry

    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - {'error': {'type': 'insufficient_quota'}}",
            "Error code: 429 - you have no credits remaining",
            "429 credit_balance_exhausted",
        ],
    )
    def test_quota_exhausted_is_not_retried(self, message: str):
        """Quota/credit errors arrive as HTTP 429 but retrying cannot fix them."""
        mock_model = MagicMock()
        mock_model.embed_documents.side_effect = Exception(message)
        embedder = Embedder(model=mock_model, dimensions=1024)

        with patch("time.sleep") as sleep:
            with pytest.raises(AdapterError) as exc_info:
                embedder.embed_batch(make_chunks(1))

        assert exc_info.value.code == ErrorCode.E4001_EMBEDDING_API_FAILED
        assert mock_model.embed_documents.call_count == 1
        sleep.assert_not_called()


@pytest.mark.unit
class TestEmbedderProviderSelection:
    """EMBEDDING_PROVIDER=openai serves both OpenAI and Azure OpenAI.

    Mirrors the convention in the sibling packages of this monorepo, where
    AZURE_OPENAI_ENDPOINT selects Azure and OPENAI_API_KEY takes precedence.
    """

    @staticmethod
    def _model_type(monkeypatch, **env: str) -> str:
        for var in ("OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return type(Embedder()._model).__name__

    def test_openai_key_selects_openai(self, monkeypatch):
        assert (
            self._model_type(monkeypatch, OPENAI_API_KEY="sk-test")
            == "OpenAIEmbeddings"
        )

    def test_azure_endpoint_selects_azure(self, monkeypatch):
        assert (
            self._model_type(
                monkeypatch,
                AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
                AZURE_OPENAI_API_KEY="azure-key",
            )
            == "AzureOpenAIEmbeddings"
        )

    def test_openai_key_takes_precedence_over_azure(self, monkeypatch):
        assert (
            self._model_type(
                monkeypatch,
                OPENAI_API_KEY="sk-test",
                AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
                AZURE_OPENAI_API_KEY="azure-key",
            )
            == "OpenAIEmbeddings"
        )

    def test_explicit_model_bypasses_env_lookup(self, monkeypatch):
        """An injected Embeddings instance ignores provider env vars entirely."""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        mock_model = MagicMock()
        assert Embedder(model=mock_model)._model is mock_model
