import warnings
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pymongo.collection import Collection

from langchain_mongodb.index import create_fulltext_search_index
from langchain_mongodb.pipelines import text_search_stage
from langchain_mongodb.utils import _append_client_metadata, make_serializable


class MongoDBAtlasFullTextSearchRetriever(BaseRetriever):
    """Retriever performs full-text searches using Lucene's standard (BM25) analyzer."""

    collection: Collection
    """MongoDB Collection on an Atlas cluster"""
    search_index_name: str
    """Atlas Search Index name"""
    search_field: Union[str, List[str]]
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
    _added_metadata: bool = False
    """Number of documents to return. Default is no limit"""

    def __init__(
        self,
        *,
        collection: Collection,
        search_index_name: str,
        search_field: Union[str, List[str]],
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
        top_k: Optional[int] = None,
        auto_create_index: bool = True,
        auto_index_timeout: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            collection=collection,
            search_index_name=search_index_name,
            search_field=search_field,
            k=k,
            filter=filter,
            include_scores=include_scores,
            top_k=top_k,
            **kwargs,
        )
        if auto_create_index and not any(
            ix["name"] == self.search_index_name
            for ix in self.collection.list_search_indexes()
        ):
            field = (
                self.search_field[0]
                if isinstance(self.search_field, list)
                else self.search_field
            )
            create_fulltext_search_index(
                collection=self.collection,
                index_name=self.search_index_name,
                field=field,
                wait_until_complete=auto_index_timeout,
            )

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
        is_top_k_set = False
        with warnings.catch_warnings():
            # Ignore warning raised by checking the value of top_k.
            warnings.simplefilter("ignore", DeprecationWarning)
            if self.top_k is not None:
                is_top_k_set = True
        default_k = self.k if not is_top_k_set else self.top_k
        pipeline = text_search_stage(  # type: ignore
            query=query,
            search_field=self.search_field,
            index_name=self.search_index_name,
            limit=kwargs.get("k", default_k),
            filter=self.filter,
            include_scores=self.include_scores,
        )

        if not self._added_metadata:
            _append_client_metadata(self.collection.database.client)
            self._added_metadata = True

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = (
                res.pop(self.search_field)
                if isinstance(self.search_field, str)
                else res.pop(self.search_field[0])
            )
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
