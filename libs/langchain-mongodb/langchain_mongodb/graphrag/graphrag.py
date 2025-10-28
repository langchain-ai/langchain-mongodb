from typing import List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_mongodb.graphrag.graph import MongoDBGraphStore


class MongoDBGraphRAGRetriever(BaseRetriever):
    """RunnableSerializable API of MongoDB GraphRAG."""

    graph_store: MongoDBGraphStore
    """Underlying Knowledge Graph storing entities and their relationships."""

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
            List of relevant documents.
        """
        return [Document(str(e)) for e in self.graph_store.similarity_search(query)]
