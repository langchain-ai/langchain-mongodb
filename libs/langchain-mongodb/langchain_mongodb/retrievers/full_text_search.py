from typing import Annotated, Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pymongo.collection import Collection

from langchain_mongodb.pipelines import text_search_stage
from langchain_mongodb.utils import make_serializable


class MongoDBAtlasFullTextSearchRetriever(BaseRetriever):
    """Retriever performs full-text searches using Lucene's standard (BM25) analyzer."""

    collection: Collection
    """MongoDB Collection on an Atlas cluster"""
    search_index_name: str
    """Atlas Search Index name"""
    search_field: str
    """Collection field that contains the text to be searched. It must be indexed"""
    k: Optional[int] = None
    """Number of documents to return. Default is no limit"""
    filter: Optional[Dict[str, Any]] = None
    """(Optional) List of MQL match expression comparing an indexed field"""
    include_scores: bool = True
    """If True, include scores that provide measure of relative relevance"""
    top_k: Annotated[
        Optional[int], Field(deprecated='top_k is deprecated, use "k" instead')
    ] = None
    """Number of documents to return. Default is no limit"""

    def close(self) -> None:
        """Close the resources used by the MongoDBAtlasFullTextSearchRetriever."""
        self.collection.database.client.close()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents that are highest scoring / most similar  to query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """
        default_k = self.k if self.k is not None else self.top_k
        pipeline = text_search_stage(  # type: ignore
            query=query,
            search_field=self.search_field,
            index_name=self.search_index_name,
            limit=kwargs.get("k", default_k),
            filter=self.filter,
            include_scores=self.include_scores,
        )

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = res.pop(self.search_field)
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
