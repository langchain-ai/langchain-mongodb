from typing import Any, Dict, Sequence, Tuple, Union

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from langchain_mongodb import MongoDBAtlasVectorSearch


class MongoDBStructuredQueryTranslator(Visitor):
    """Translator between MongoDB Query API and LangChain's StructuredQuery.

    With Vector Search Indexes, one can index boolean, date, number, objectId, string,
    and UUID fields to pre-filter your data.
    Filtering your data is useful to narrow the scope of your semantic search
    and ensure that not all vectors are considered for comparison.
    It reduces the number of documents against which to run similarity comparisons,
    which can decrease query latency and increase the accuracy of search results.
    """

    """Available logical comparators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.IN,
        Comparator.NIN,
    ]

    """Available logical operators."""
    allowed_operators = [Operator.AND, Operator.OR]

    ## Convert an operator or a comparator to Mongo Query Format
    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        map_dict = {
            Operator.AND: "$and",
            Operator.OR: "$or",
            Comparator.EQ: "$eq",
            Comparator.NE: "$ne",
            Comparator.GTE: "$gte",
            Comparator.LTE: "$lte",
            Comparator.LT: "$lt",
            Comparator.GT: "$gt",
            Comparator.IN: "$in",
            Comparator.NIN: "$nin",
        }
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        if comparison.comparator in [Comparator.IN, Comparator.NIN] and not isinstance(
            comparison.value, list
        ):
            comparison.value = [comparison.value]
        comparator = self._format_func(comparison.comparator)
        attribute = comparison.attribute
        return {attribute: {comparator: comparison.value}}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"pre_filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs


class MongoDBAtlasSelfQueryRetriever(SelfQueryRetriever):
    """Retriever that uses an LLM to deduce filters for Vector Search algorithm.

    This can greatly increase power of vector search on collections with structured metadata.

    Before calling the search algorithm of the vector store, this retriever
    first prompts an LLM to find logical statements (e.g. and, in) in a semantic query.
    From the response, it forms a structured query,
    which it passes to a VectorStore as filters,

    The fields to look for conditions are specified by ``metadata_field_info``
    a simple list of attribute information for each fieldname, type, description.
    See `How to do "self-querying" retrieval <https://python.langchain.com/docs/how_to/self_query/>`_
    for more information.

    One must index the fields that you want to filter your data by
    as the filter type in a vectorSearch type index definition.

    Example usage:

    .. code-block:: python

        from langchain_mongodb.retrievers import MongoDBAtlasSelfQueryRetriever
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from langchain_ollama.embeddings import OllamaEmbeddings

        # Start with the standard MongoDB Atlas vector store
        vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string="mongodb://127.0.0.1:40947/?directConnection=true",
            namespace=f"{DB_NAME}.{COLLECTION_NAME}",
            embedding=OllamaEmbeddings(model="all-minilm:l6-v2")
        )
        # Define metadata describing the data
        metadata_field_info = [
            AttributeInfo(
                name="genre",
                description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'animated']",
                type="string",
            ),
            AttributeInfo(
                name="year",
                description="The year the movie was released",
                type="integer",
            ),
            AttributeInfo(
                name="rating", description="A 1-10 rating for the movie", type="float"
            ),
        ]

        # Create  search index with filters
        vectorstore.create_vector_search_index(
            dimensions=dimensions,
            filters=[f.name for f in metadata_field_info],
            wait_until_complete=TIMEOUT
        )

        # Add documents, including embeddings
        vectorstore.add_documents(fictitious_movies)

        # Create the retriever from the VectorStore, an LLM and info about the documents
        retriever = MongoDBAtlasSelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            metadata_field_info=metadata_field_info,
            document_contents="Descriptions of movies",
            enable_limit=True
        )

        # This example results in the following composite filter sent to $vectorSearch:
        # {'filter': {'$and': [{'year': {'$lt': 1960}}, {'rating': {'$gt': 8}}]}}
        print(retriever.invoke("Movies made before 1960 that are rated higher than 8"))

    See Also:
        * `Run Vector Search Queries <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#run-vector-search-queries>`_
        * `How to Index Fields for Vector Search <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type>`_
        * :class:`~langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch`
    """

    vectorstore: MongoDBAtlasVectorSearch
    """The underlying vector store from which documents will be retrieved."""
    structured_query_translator: MongoDBStructuredQueryTranslator
    """Translator for turning LangChain internal query language into Atlas search params."""
    query_constructor: Runnable[dict, StructuredQuery] = Field(alias="llm_chain")
    """The query constructor chain for generating the vector store queries."""
    search_type: str = "similarity"
    """The search type to perform on the vector store."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to MongoDBAtlasVectorSearch (e.g. ``{'k':10})``."""
    verbose: bool = False
    """logs the structured query generated by the LLM"""
    use_original_query: bool = False
    """Use original query instead of the LLM's revised query that removes statements with filters."""

    @classmethod
    def from_llm(  # type:ignore[override]
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        document_contents: str,
        metadata_field_info: Sequence[Union[AttributeInfo, dict]],
        enable_limit: bool = False,
        use_original_query: bool = False,
        **kwargs: Any,
    ) -> SelfQueryRetriever:
        """Create a self-querying retriever from an LLM, vector store, and document metadata.

        This method does NOT create the vector search index. See example usage.

        Args:
            llm: A Reasoning model that will produce the structured query.
            vectorstore: MongoDBAtlasVectorSearch.
            document_contents: Description of the data in the collection.
            metadata_field_info: Fields must be present in vector search index.
            enable_limit: Whether to instruct the LLM to look for statements involving limits.
            use_original_query: By default, sentences defining filters and limits are removed from query.
                Set to True if you wish to include these in the text to embed.
            **kwargs: Additional arguments to pass to retriever constructor (e.g. search_kwargs)

        Returns: A retriever invoked by a text query.
        """

        return super().from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_contents,
            metadata_field_info=metadata_field_info,
            structured_query_translator=MongoDBStructuredQueryTranslator(),
            enable_limit=enable_limit,
            use_original_query=use_original_query,
            **kwargs,
        )

    def close(self) -> None:
        """Close the resources used by the retriever."""
        self.vectorstore.close()
