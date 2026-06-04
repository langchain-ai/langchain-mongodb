from typing import List, Optional, Union

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_mongodb.graphrag.graph import MongoDBGraphStore


class MongoDBGraphRAGRetriever(BaseRetriever):
    """RunnableSerializable API of MongoDB GraphRAG."""

    graph_store: MongoDBGraphStore
    """Underlying Knowledge Graph storing entities and their relationships."""
    rerank_path: Optional[Union[str, List[str]]] = None
    """Field or list of fields on entity documents to rerank on. Enables $rerank when set.
    The entity ``_id`` (name) is a natural choice; users may also pass a list such as
    ``["_id", "type"]``. Requires MongoDB 8.3+ and Native Reranking enabled in Atlas."""
    rerank_model: Optional[str] = None
    """Voyage AI reranking model (e.g. 'rerank-2.5'). Uses latest model if omitted."""
    num_docs_to_rerank: Optional[int] = None
    """Candidates passed to the reranker. Defaults to 1000 (all graph results). Max 1000."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve list of Entities found via traversal of KnowledgeGraph.

        Each Document's page_content is a string representation of the Entity dict.

        Description and details are provided in the underlying Entity Graph:
        :class:`~langchain_mongodb.graphrag.graph.MongoDBGraphStore`

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use if desired
        Returns:
            List of relevant documents, reranked by relevance if rerank_path is set.
        """
        entities = self.graph_store.similarity_search(
            query,
            rerank_path=self.rerank_path,
            rerank_model=self.rerank_model,
            num_docs_to_rerank=self.num_docs_to_rerank,
        )
        return [Document(str(e)) for e in entities]
