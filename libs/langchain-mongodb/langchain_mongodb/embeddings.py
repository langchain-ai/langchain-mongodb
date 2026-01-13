from __future__ import annotations

from langchain_core.embeddings import Embeddings
from pymongo.errors import ConfigurationError

SUPPORTED_VOYAGE_MODELS = [
    "voyage-4",
    "voyage-code-3",
    "voyage-4-large",
    "voyage-4-lite",
]


class AutoEmbedding(Embeddings):
    def __init__(self, model_name: str):
        if model_name not in SUPPORTED_VOYAGE_MODELS:
            # TODO: double check that this should be a CondifurationError and not something else?
            raise ConfigurationError(
                f"The following embedding model is not supported: {model_name}. Supported models are: {', '.join(SUPPORTED_VOYAGE_MODELS)}."
            )

        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # """Embed search docs."""
        raise NotImplementedError(
            "One must choose a model. We recommend VoyageAIEmbeddings"
        )

    def embed_query(self, text: str) -> list[float]:
        # """Embed query text."""
        raise NotImplementedError(
            "One must choose a model. We recommend VoyageAIEmbeddings"
        )
