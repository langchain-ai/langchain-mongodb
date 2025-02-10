"""Retriever that generates and executes structured queries over its own data source."""

import logging
from typing import Dict, Sequence, Tuple, Union

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class MongoDBStructuredQueryTranslator(Visitor):
    """Translator between  MongoDB filters and LangChain's StructuredQuery

    With Vector Search Indexes, one can index boolean, date, number, objectId, string,
    and UUID fields to pre-filter your data.
    Filtering your data is useful to narrow the scope of your semantic search
    and ensure that not all vectors are considered for comparison.
    It reduces the number of documents against which to run similarity comparisons,
    which can decrease query latency and increase the accuracy of search results.

    """

    """Subset of allowed logical comparators."""
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

    """Subset of allowed logical operators."""
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
    def __init__(
        self,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        metadata_field_info: Sequence[Union[AttributeInfo, dict]],
        document_contents="Descriptions of movies",
    ):
        pass
