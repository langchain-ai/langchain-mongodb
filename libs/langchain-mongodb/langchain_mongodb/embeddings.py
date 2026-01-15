from __future__ import annotations

from langchain_core.embeddings import Embeddings


class AutoEmbeddings(Embeddings):
    def __init__(self, model: str):
        """MongoDB AutoEmbeddings

        AutoEmbedding enables MongoDB to automatically generate and manage embedding vectors.
        Since the embedding happens on the server, this class doesn't implement embed_documents
        or embed_query and simply requires a model name.
        For supported models, see https://www.mongodb.com/docs/atlas/atlas-vector-search/crud-embeddings/create-embeddings-automatic/?interface=driver&language=python&deployment-type=self#supported-embedding-models
        """

        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # """Embed search docs."""
        raise NotImplementedError(
            "With AutoEmbeddings, all embeddings and keys are handled in the vector search index."
        )

    def embed_query(self, text: str) -> list[float]:
        # """Embed query text."""
        raise NotImplementedError(
            "With AutoEmbeddings, all embeddings and keys are handled in the vector search index."
        )
