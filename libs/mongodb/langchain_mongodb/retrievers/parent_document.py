from __future__ import annotations

from importlib.metadata import version
from typing import Any, List

import pymongo
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.docstores import MongoDBDocStore
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import make_serializable


class MongoDBAtlasParentDocumentRetriever(ParentDocumentRetriever):
    """MongoDB Atlas's ParentDocumentRetriever

    Uses ONE Collection for both Vector and Doc store.

    For details, see parent classes
        :class:`~langchain.retrievers.parent_document_retriever.ParentDocumentRetriever`
        and :class:`~langchain.retrievers.MultiVectorRetriever` for further details.

    Examples:
        >>> from langchain_mongodb.retrievers.parent_document import (
        >>>     ParentDocumentRetriever
        >>> )
        >>> from langchain_text_splitters import RecursiveCharacterTextSplitter
        >>> from langchain_openai import OpenAIEmbeddings
        >>>
        >>> retriever = ParentDocumentRetriever.from_connection_string(
        >>>     "mongodb+srv://<user>:<clustername>.mongodb.net",
        >>>     OpenAIEmbeddings(model="text-embedding-3-large"),
        >>>     RecursiveCharacterTextSplitter(chunk_size=400),
        >>>     "example_database"
        >>> )
        retriever.add_documents([Document(..., technical_report_pages)
        >>> resp = retriever.invoke("Langchain MongDB Partnership Ecosystem")
        >>> print(resp)
        [Document(...), ...]

    """

    vectorstore: MongoDBAtlasVectorSearch
    """Vectorstore API to add, embed, and search through child documents"""

    docstore: MongoDBDocStore
    """Provides an API around the Collection to add the parent documents"""

    id_key: str = "doc_id"
    """Key stored in metadata pointing to parent document"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_vector = self.vectorstore._embedding.embed_query(query)

        pipeline = [
            vector_search_stage(
                query_vector,
                self.vectorstore._embedding_key,
                self.vectorstore._index_name,
                **self.search_kwargs,  # See MongoDBAtlasVectorSearch
            ),
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {  # Find corresponding parent doc
                "$lookup": {
                    "from": self.vectorstore.collection.name,
                    "localField": self.id_key,
                    "foreignField": "_id",
                    "as": "parent_context",
                    "pipeline": [
                        # Discard sub-documents
                        {"$match": {f"metadata.{self.id_key}": {"$exists": False}}},
                    ],
                }
            },  # Remove duplicate parent docs and reformat
            {"$unwind": {"path": "$parent_context"}},
            {
                "$group": {
                    "_id": "$parent_context._id",
                    "uniqueDocument": {"$first": "$parent_context"},
                }
            },
            {"$replaceRoot": {"newRoot": "$uniqueDocument"}},
        ]
        # Execute
        cursor = self.vectorstore._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []
        # Format into Documents
        for res in cursor:
            text = res.pop(self.vectorstore._text_key)
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        embedding_model: Embeddings,
        child_splitter: TextSplitter,
        database_name: str,
        collection_name: str = "document_with_chunks",
        id_key: str = "doc_id",
        **kwargs: Any,
    ) -> MongoDBAtlasParentDocumentRetriever:
        """Construct Retriever using one Collection for VectorStore and one for DocStore

        See parent classes
        :class:`~langchain.retrievers.parent_document_retriever.ParentDocumentRetriever`
        and :class:`~langchain.retrievers.MultiVectorRetriever` for further details.

        Args:
            connection_string: A valid MongoDB Atlas connection URI.
            embedding_model: The text embedding model to use for the vector store.
            child_splitter: Splits documents into chunks.
                If parent_splitter is given, the documents will have already been split.
            database_name: Name of database to connect to. Created if it does not exist.
            collection_name: Name of collection to use.
                It includes parent documents, sub-documents and their  embeddings.
            id_key: Key used to identify parent documents.
            **kwargs: Additional keyword arguments. See parent classes for more.

        Returns: A new MongoDBAtlasParentDocumentRetriever
        """
        client: MongoClient = MongoClient(
            connection_string,
            driver=DriverInfo(name="langchain", version=version("langchain-mongodb")),
        )
        collection = client[database_name][collection_name]
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection, embedding=embedding_model, **kwargs
        )

        docstore = MongoDBDocStore(collection=collection)
        docstore.collection.create_index([(id_key, pymongo.ASCENDING)])

        return cls(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            id_key=id_key,
            **kwargs,
        )
