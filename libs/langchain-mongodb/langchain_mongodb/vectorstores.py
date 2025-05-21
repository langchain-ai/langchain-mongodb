from __future__ import annotations

import logging
from importlib.metadata import version
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from bson import ObjectId
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import CollectionInvalid

from langchain_mongodb.index import (
    create_vector_search_index,
    update_vector_search_index,
)
from langchain_mongodb.pipelines import vector_search_stage
from langchain_mongodb.utils import (
    make_serializable,
    maximal_marginal_relevance,
    oid_to_str,
    str_to_oid,
)

VST = TypeVar("VST", bound=VectorStore)

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 100


class MongoDBAtlasVectorSearch(VectorStore):
    """MongoDB Atlas vector store integration.

    MongoDBAtlasVectorSearch performs data operations on
    text, embeddings and arbitrary data. In addition to CRUD operations,
    the VectorStore provides Vector Search
    based on similarity of embedding vectors following the
    Hierarchical Navigable Small Worlds (HNSW) algorithm.

    This supports a number of models to ascertain scores,
    "similarity" (default), "MMR", and "similarity_score_threshold".
    These are described in the search_type argument to as_retriever,
    which provides the Runnable.invoke(query) API, allowing
    MongoDBAtlasVectorSearch to be used within a chain.

    Setup:
        * Set up a MongoDB Atlas cluster. The free tier M0 will allow you to start.
          Search Indexes are only available on Atlas, the fully managed cloud service,
          not the self-managed MongoDB.
          Follow [this guide](https://www.mongodb.com/basics/mongodb-atlas-tutorial)

        * Create a Collection and a Vector Search Index.  The procedure is described
          [here](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure).
          You can optionally supply a `dimensions` argument to programmatically create a Vector
          Search Index.

        * Install ``langchain-mongodb``


        .. code-block:: bash

            pip install -qU langchain-mongodb pymongo


        .. code-block:: python

            import getpass
            MONGODB_ATLAS_CONNECTION_STRING = getpass.getpass("MongoDB Atlas Connection String:")

    Key init args — indexing params:
        embedding: Embeddings
            Embedding function to use.

    Key init args — client params:
        collection: Collection
            MongoDB collection to use.
        index_name: str
            Name of the Atlas Search index.

    Instantiate:
        .. code-block:: python

            from pymongo import MongoClient
            from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
            from pymongo import MongoClient
            from langchain_openai import OpenAIEmbeddings

            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=os=MONGODB_ATLAS_CONNECTION_STRING,
                namespace="db_name.collection_name",
                embedding=OpenAIEmbeddings(),
                index_name="vector_index",
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'_id': '2', 'baz': 'baz'}]


    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,post_filter=[{"bar": "baz"]})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'_id': '2', 'baz': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916096] foo [{'_id': '1', 'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916096] foo [{'_id': '1', 'baz': 'bar'}]

    Use as Retriever:

        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'_id': '2', 'embedding': [-0.01850726455450058, -0.0014740974875167012, -0.009762819856405258, ...], 'baz': 'baz'}, page_content='thud')]

    """  # noqa: E501

    def __init__(
        self,
        collection: Collection[Dict[str, Any]],
        embedding: Embeddings,
        index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        dimensions: int = -1,
        auto_create_index: bool = True,
        auto_index_timeout: int = 15,
        **kwargs: Any,
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to
            embedding: Text embedding model to use
            text_key: MongoDB field that will contain the text for each document
            index_name: Existing Atlas Vector Search Index
            embedding_key: Field that will contain the embedding for each document
            vector_index_name: Name of the Atlas Vector Search index
            relevance_score_fn: The similarity score used for the index
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'
            dimensions: Number of dimensions in embedding.  If the value is set and
                the index does not exist, an index will be created.
            auto_index_timeout: Timeout in seconds to wait for an auto-created index
               to be ready.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn

        if not auto_create_index or dimensions == -1:
            return
        coll = self._collection
        if not any(
            [ix["name"] == self._index_name for ix in coll.list_search_indexes()]
        ):
            create_vector_search_index(
                collection=coll,
                index_name=self._index_name,
                dimensions=dimensions,
                path=self._embedding_key,
                similarity=self._relevance_score_fn,
                wait_until_complete=auto_index_timeout,
            )

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def collection(self) -> Collection:
        return self._collection

    @collection.setter
    def collection(self, value: Collection) -> None:
        self._collection = value

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """All Atlas Vector Search Scores are normalized in [0,1] so no change needed."""
        return lambda score: score

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct a `MongoDB Atlas Vector Search` vector store
        from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (database and collection).
            embedding: The text embedding model to use for the vector store.

        Returns:
            A new MongoDBAtlasVectorSearch instance.

        """
        client: MongoClient = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        )
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection, embedding, **kwargs)

    def close(self) -> None:
        """Close the resources used by the MongoDBAtlasVectorSearch."""
        self._collection.database.client.close()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts, create embeddings, and add to the Collection and index.

        Important notes on ids:
            - If _id or id is a key in the metadatas dicts, one must
                pop them and provide as separate list.
            - They must be unique.
            - If they are not provided, the VectorStore will create unique ones,
                stored as bson.ObjectIds internally, and strings in Langchain.
                These will appear in Document.metadata with key, '_id'.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of ids added to the vectorstore.
        """

        # Check to see if metadata includes ids
        if metadatas is not None and (
            metadatas[0].get("_id") or metadatas[0].get("id")
        ):
            logger.warning(
                "_id or id key found in metadata. "
                "Please pop from each dict and input as separate list."
                "Retrieving methods will include the same id as '_id' in metadata."
            )

        texts_batch = texts
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)
        metadatas_batch = _metadatas

        result_ids = []
        if batch_size:
            texts_batch = []
            metadatas_batch = []
            size = 0
            i = 0
            for j, (text, metadata) in enumerate(zip(texts, _metadatas)):
                size += len(text) + len(metadata)
                texts_batch.append(text)
                metadatas_batch.append(metadata)
                if (j + 1) % batch_size == 0 or size >= 47_000_000:
                    if ids:
                        batch_res = self.bulk_embed_and_insert_texts(
                            texts_batch, metadatas_batch, ids[i : j + 1]
                        )
                    else:
                        batch_res = self.bulk_embed_and_insert_texts(
                            texts_batch, metadatas_batch
                        )
                    result_ids.extend(batch_res)
                    texts_batch = []
                    metadatas_batch = []
                    size = 0
                    i = j + 1
        if texts_batch:
            if ids:
                batch_res = self.bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch, ids[i : j + 1]
                )
            else:
                batch_res = self.bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch
                )
            result_ids.extend(batch_res)
        return result_ids

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.

        .. versionadded:: 0.6.0
        """
        docs = []
        oids = [str_to_oid(i) for i in ids]
        for doc in self._collection.aggregate([{"$match": {"_id": {"$in": oids}}}]):
            _id = doc.pop("_id")
            text = doc.pop("text")
            del doc["embedding"]
            docs.append(Document(page_content=text, id=_id, metadata=doc))
        return docs

    def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Bulk insert single batch of texts, embeddings, and optionally ids.

        See add_texts for additional details.
        """
        if not texts:
            return []
        # Compute embedding vectors
        embeddings = self._embedding.embed_documents(list(texts))
        if not ids:
            ids = [str(ObjectId()) for _ in range(len(list(texts)))]
        docs = [
            {
                "_id": str_to_oid(i),
                self._text_key: t,
                self._embedding_key: embedding,
                **m,
            }
            for i, t, m, embedding in zip(ids, texts, metadatas, embeddings)
        ]
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        # insert the documents in MongoDB Atlas
        result = self._collection.bulk_write(operations)
        assert result.upserted_ids is not None
        return [oid_to_str(_id) for _id in result.upserted_ids.values()]

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids in add_texts.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of IDs of the added texts.
        """
        n_docs = len(documents)
        if ids:
            assert len(ids) == n_docs, "Number of ids must equal number of documents."
        else:
            ids = [doc.id or str(ObjectId()) for doc in documents]
        result_ids = []
        start = 0
        for end in range(batch_size, n_docs + batch_size, batch_size):
            texts, metadatas = zip(
                *[(doc.page_content, doc.metadata) for doc in documents[start:end]]
            )
            result_ids.extend(
                self.bulk_embed_and_insert_texts(
                    texts=texts, metadatas=metadatas, ids=ids[start:end]
                )
            )
            start = end
        return result_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:  # noqa: E501
        """Return MongoDB documents most similar to the given query and their scores.

        Atlas Vector Search eliminates the need to run a separate
        search system alongside your database.

         Args:
            query: Input text of semantic query
            k: Number of documents to return. Also known as top_k.
            pre_filter: List of MQL match expressions comparing an indexed field
            post_filter_pipeline: (Optional) Arbitrary pipeline of MongoDB
                aggregation stages applied after the search is complete.
            oversampling_factor: This times k is the number of candidates chosen
                at each step in the in HNSW Vector Search
            include_embeddings: If True, the embedding vector of each result
                will be included in metadata.
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """
        embedding = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            oversampling_factor=oversampling_factor,
            include_embeddings=include_embeddings,
            **kwargs,
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_scores: bool = False,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Document]:  # noqa: E501
        """Return MongoDB documents most similar to the given query.

        Atlas Vector Search eliminates the need to run a separate
        search system alongside your database.

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: List of MQL match expressions comparing an indexed field
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                to filter/process results after $vectorSearch.
            oversampling_factor: Multiple of k used when generating number of candidates
                at each step in the HNSW Vector Search,
            include_scores: If True, the query score of each result
                will be included in metadata.
            include_embeddings: If True, the embedding vector of each result
                will be included in metadata.
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            oversampling_factor=oversampling_factor,
            include_embeddings=include_embeddings,
            **kwargs,
        )

        if include_scores:
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            fetch_k: (Optional) number of documents to fetch before passing to MMR
                algorithm. Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            pre_filter: List of MQL match expressions comparing an indexed field
            post_filter_pipeline: (Optional) pipeline of MongoDB aggregation stages
                following the $vectorSearch stage.
        Returns:
            List of documents selected by maximal marginal relevance.
        """
        return self.max_marginal_relevance_search_by_vector(
            embedding=self._embedding.embed_query(query),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        collection: Optional[Collection] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct a `MongoDB Atlas Vector Search` vector store from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided MongoDB Atlas Vector Search index
                (Lucene)

        This is intended to be a quick way to get started.

        See `MongoDBAtlasVectorSearch` for kwargs and further description.


        Example:
            .. code-block:: python
                from pymongo import MongoClient

                from langchain_mongodb import MongoDBAtlasVectorSearch
                from langchain_openai import OpenAIEmbeddings

                mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
                collection = mongo_client["<db_name>"]["<collection_name>"]
                embeddings = OpenAIEmbeddings()
                vectorstore = MongoDBAtlasVectorSearch.from_texts(
                    texts,
                    embeddings,
                    metadatas=metadatas,
                    collection=collection
                )
        """
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from VectorStore by ids.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments passed to Collection.delete_many()

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        filter = {}
        if ids:
            oids = [str_to_oid(i) for i in ids]
            filter = {"_id": {"$in": oids}}
        return self._collection.delete_many(filter=filter, **kwargs).acknowledged

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await run_in_executor(None, self.delete, ids=ids, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        **kwargs: Any,
    ) -> List[Document]:  # type: ignore
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            pre_filter: (Optional) dictionary of arguments to filter document fields on.
            post_filter_pipeline: (Optional) pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
            oversampling_factor: Multiple of k used when generating number
                of candidates in HNSW Vector Search,
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs = self._similarity_search_with_score(
            embedding,
            k=fetch_k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            include_embeddings=True,
            oversampling_factor=oversampling_factor,
            **kwargs,
        )
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,  # type: ignore[arg-type]
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            oversampling_factor=oversampling_factor,
            **kwargs,
        )

    def _similarity_search_with_score(
        self,
        query_vector: List[float],
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Core search routine. See external methods for details."""

        # Atlas Vector Search, potentially with filter
        pipeline = [
            vector_search_stage(
                query_vector,
                self._embedding_key,
                self._index_name,
                k,
                pre_filter,
                oversampling_factor,
                **kwargs,
            ),
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        # Remove embeddings unless requested.
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})
        # Post-processing
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        # Execution
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        # Format
        for res in cursor:
            if self._text_key not in res:
                continue
            text = res.pop(self._text_key)
            score = res.pop("score")
            make_serializable(res)
            docs.append(
                (Document(page_content=text, metadata=res, id=res["_id"]), score)
            )
        return docs

    def create_vector_search_index(
        self,
        dimensions: int,
        filters: Optional[List[str]] = None,
        update: bool = False,
        wait_until_complete: Optional[float] = None,
    ) -> None:
        """Creates a MongoDB Atlas vectorSearch index for the VectorStore

        Note**: This method may fail as it requires a MongoDB Atlas with these
        `pre-requisites <https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#prerequisites>`.
        Currently, vector and full-text search index operations need to be
        performed manually on the Atlas UI for shared M0 clusters.

        Args:
            dimensions (int): Number of dimensions in embedding
            filters (Optional[List[Dict[str, str]]], optional): additional filters
            for index definition.
                Defaults to None.
            update (Optional[bool]): Updates existing vectorSearch index.
                 Defaults to False.
            wait_until_complete (Optional[float]): If given, a TimeoutError is raised
                if search index is not ready after this number of seconds.
                If not given, the default, operation will not wait.
        """
        try:
            self._collection.database.create_collection(self._collection.name)
        except CollectionInvalid:
            pass

        index_operation = (
            update_vector_search_index if update else create_vector_search_index
        )

        index_operation(
            collection=self._collection,
            index_name=self._index_name,
            dimensions=dimensions,
            path=self._embedding_key,
            similarity=self._relevance_score_fn,
            filters=filters or [],
            wait_until_complete=wait_until_complete,
        )  # type: ignore [operator]
