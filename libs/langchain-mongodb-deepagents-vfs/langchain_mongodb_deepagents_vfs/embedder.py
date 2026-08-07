"""Embedding module — single point of contact with the embedding provider.

Provider is selected at construction time via the ``EMBEDDING_PROVIDER`` env var
(default: ``bedrock``).  Supported values:

  bedrock  — BedrockEmbeddings via langchain-aws (uses boto3 credential chain)
             Default model: amazon.titan-embed-text-v2:0 @ 1024 dims
             Requires ``langchain-aws`` and Bedrock model access in the account
             whose credentials the boto3 chain resolves.
  openai   — OpenAIEmbeddings (requires OPENAI_API_KEY)
             Default model: text-embedding-3-small @ 1024 dims
             Requires ``langchain-openai``.

Use :func:`resolve_provider` rather than reading ``EMBEDDING_PROVIDER``
directly, so callers cannot disagree with :class:`Embedder` about the default.

Override the model name with ``EMBEDDING_MODEL``.  Any LangChain-compatible
``Embeddings`` instance can also be passed directly to bypass env-var lookup.

Swapping providers is a constructor change (or env-var change) only — the
rest of the stack depends solely on this class, never on a specific provider.
"""

from __future__ import annotations

import logging
import os
import time

from langchain_core.embeddings import Embeddings

from langchain_mongodb_deepagents_vfs.dtypes import Chunk
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "bedrock"
_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
_BEDROCK_DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_DIMENSIONS = 1024
_DEFAULT_BATCH_SIZE = 100

# Substrings identifying provider errors that retrying can never resolve: the
# account is out of credit or over its billing quota.  These arrive as HTTP 429
# just like transient rate limits, so they must be matched *before* the
# retry-on-429 check or every failure costs the full backoff sequence.
_TERMINAL_QUOTA_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "no credits remaining",
    "exceeded your current quota",
    "billing",
)


def resolve_provider() -> str:
    """Return the configured embedding provider name, lowercased.

    Single source of truth for the ``EMBEDDING_PROVIDER`` default so callers
    (including tests) cannot disagree with :class:`Embedder` about which
    provider is actually in use.
    """
    return os.getenv("EMBEDDING_PROVIDER", DEFAULT_PROVIDER).lower()


class Embedder:
    """Wraps a LangChain-compatible Embeddings model.

    Args:
        model: A LangChain ``Embeddings`` instance.  If None, the provider is
            resolved from the ``EMBEDDING_PROVIDER`` env var (default ``bedrock``).
        model_name: Model identifier used when *model* is None.  Falls back to
            the ``EMBEDDING_MODEL`` env var, then the provider default.
        dimensions: Expected embedding vector size; validated on first call.
        batch_size: Number of chunk texts sent per API call.
    """

    def __init__(
        self,
        model: Embeddings | None = None,
        model_name: str | None = None,
        dimensions: int = _DEFAULT_DIMENSIONS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._dimensions = dimensions
        self._batch_size = batch_size
        if model is not None:
            self._model = model
        else:
            provider = resolve_provider()
            resolved_name = (
                model_name
                or os.getenv("EMBEDDING_MODEL")
                or (
                    _OPENAI_DEFAULT_MODEL
                    if provider == "openai"
                    else _BEDROCK_DEFAULT_MODEL
                )
            )
            self._model = self._build_model(provider, resolved_name, dimensions)

    @staticmethod
    def _build_model(provider: str, model_name: str, dimensions: int) -> Embeddings:
        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings

                logger.info(
                    "Embedder: provider=openai model=%s dimensions=%d",
                    model_name,
                    dimensions,
                )
                embeddings: Embeddings = OpenAIEmbeddings(
                    model=model_name, dimensions=dimensions
                )
                return embeddings
            except ImportError as exc:
                raise AdapterError(
                    ErrorCode.E4001_EMBEDDING_API_FAILED,
                    "langchain-openai is not installed; pip install langchain-openai",
                ) from exc

        if provider == "bedrock":
            try:
                from langchain_aws import BedrockEmbeddings

                region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                logger.info(
                    "Embedder: provider=bedrock model=%s region=%s dimensions=%d",
                    model_name,
                    region,
                    dimensions,
                )
                bedrock_embeddings: Embeddings = BedrockEmbeddings(
                    model_id=model_name, region_name=region
                )
                return bedrock_embeddings
            except ImportError as exc:
                raise AdapterError(
                    ErrorCode.E4001_EMBEDDING_API_FAILED,
                    "langchain-aws is not installed; pip install langchain-aws",
                ) from exc

        raise AdapterError(
            ErrorCode.E1001_MISSING_CONFIG,
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Supported values: openai, bedrock",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_batch(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed *chunks* in batches, returning one vector per chunk.

        Args:
            chunks: Chunk objects whose ``content`` field is embedded.

        Returns:
            List of float vectors, same order as *chunks*.

        Raises:
            AdapterError(E4001): API failure.
            AdapterError(E4002): Dimension mismatch.
            AdapterError(E4003): Rate limit.
        """
        texts = [c.content for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_vectors = self._embed_with_retry(batch)
            vectors.extend(batch_vectors)

        if vectors and len(vectors[0]) != self._dimensions:
            raise AdapterError(
                ErrorCode.E4002_EMBEDDING_DIMENSION_MISMATCH,
                f"Expected {self._dimensions} dims, got {len(vectors[0])}",
            )
        return vectors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the embedding model with exponential-backoff retry."""
        last_exc: Exception | None = None
        delays = [1, 2, 4, 8, 16]
        for delay in delays:
            try:
                return self._model.embed_documents(texts)
            except Exception as exc:
                exc_str = str(exc).lower()
                if any(marker in exc_str for marker in _TERMINAL_QUOTA_MARKERS):
                    logger.error(
                        "Embedding quota exhausted — not retrying. provider=%s "
                        "model=%s: %s",
                        resolve_provider(),
                        type(self._model).__name__,
                        exc,
                    )
                    raise AdapterError(
                        ErrorCode.E4001_EMBEDDING_API_FAILED,
                        f"Embedding provider quota/credit exhausted (not retryable): {exc}",
                    ) from exc
                if "rate" in exc_str or "429" in exc_str:
                    logger.warning("Embedding rate-limited, retrying in %ds…", delay)
                    time.sleep(delay)
                    last_exc = exc
                else:
                    logger.error(
                        "Embedding call failed [%s: %s] — model=%s texts=%d",
                        type(exc).__name__,
                        exc,
                        type(self._model).__name__,
                        len(texts),
                        exc_info=True,
                    )
                    raise AdapterError(
                        ErrorCode.E4001_EMBEDDING_API_FAILED, str(exc)
                    ) from exc
        raise AdapterError(ErrorCode.E4003_EMBEDDING_RATE_LIMITED, str(last_exc))
