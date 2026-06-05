import warnings
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pymongo.collection import Collection

from langchain_mongodb.index import create_fulltext_search_index
from langchain_mongodb.pipelines import rerank_stage, text_search_stage
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
    rerank_path: Optional[Union[str, List[str]]] = None
    """Field or list of fields to rerank on. Enables $rerank when set."""
    rerank_model: Optional[str] = None
    """Voyage AI reranking model (e.g. 'rerank-2.5'). Uses latest model if omitted."""
    num_docs_to_rerank: Optional[int] = None
    """Candidates passed to the reranker. Defaults to k. Max 1000."""
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
        rerank_path: Optional[Union[str, List[str]]] = None,
        rerank_model: Optional[str] = None,
        num_docs_to_rerank: Optional[int] = None,
        top_k: Optional[int] = None,
        auto_create_index: bool = True,
        auto_index_timeout: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            collection=collection,
            search_index_name=search_index_name,
            search_field=search_field,
            k=k,
            filter=filter,
            include_scores=include_scores,
            rerank_path=rerank_path,
            rerank_model=rerank_model,
            num_docs_to_rerank=num_docs_to_rerank,
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
        k = kwargs.get("k", default_k)
        # num_docs_to_rerank must be a concrete int for $rerank; fall back to 1000
        # (the stage maximum) when no limit is configured on the retriever.
        n_to_rerank: int = self.num_docs_to_rerank or k or 1000
        # Expand the text search limit so the reranker has enough candidates.
        text_limit = n_to_rerank if self.rerank_path else k
        pipeline = text_search_stage(  # type: ignore
            query=query,
            search_field=self.search_field,
            index_name=self.search_index_name,
            limit=text_limit,
            filter=self.filter,
            include_scores=self.include_scores,
        )

        # Native Reranking via $rerank (requires MongoDB 8.3+ and Atlas project setting).
        if self.rerank_path is not None:
            pipeline.extend(
                rerank_stage(query, self.rerank_path, n_to_rerank, self.rerank_model)
            )
            if k is not None and n_to_rerank > k:
                pipeline.append({"$limit": k})

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
