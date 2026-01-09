"""Aggregation pipeline components used in Atlas Full-Text, Vector, and Hybrid Search

See the following for more:
    - `Full-Text Search <https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search>`_
    - `MongoDB Operators <https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref>`_
    - `Vector Search <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/>`_
    - `Filter Example <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_
"""

from typing import Any, Dict, List, Optional, Union

from pymongo_search_utils import (
    combine_pipelines,  # noqa: F401
    final_hybrid_stage,  # noqa: F401
    reciprocal_rank_stage,  # noqa: F401
    vector_search_stage,  # noqa: F401
)


def text_search_stage(
    query: str,
    search_field: Union[str, List[str]],
    index_name: str,
    limit: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None,
    include_scores: Optional[bool] = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:  # noqa: E501
    """Full-Text search using Lucene's standard (BM25) analyzer

    Args:
        query: Input text to search for
        search_field: Field in Collection that will be searched
        index_name: Atlas Search Index name
        limit: Maximum number of documents to return. Default of no limit
        filter: Any MQL match expression comparing an indexed field
        include_scores: Scores provide measure of relative relevance

    Returns:
        Dictionary defining the $search stage
    """
    pipeline = [
        {
            "$search": {
                "index": index_name,
                "text": {"query": query, "path": search_field},
            }
        }
    ]
    if filter:
        pipeline.append({"$match": filter})  # type: ignore
    if include_scores:
        pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})
    if limit:
        pipeline.append({"$limit": limit})  # type: ignore

    return pipeline  # type: ignore

def autoembedded_vector_search_stage(
    query: str,
    search_field: str,
    index_name: str,
    model_name: str,
    top_k: int = 4,
    filter: dict[str, Any] | None = None,
    oversampling_factor: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:  # noqa: E501
    """Vector Search Stage without Scores.

    Scoring is applied later depending on strategy.
    vector search includes a vectorSearchScore that is typically used.
    hybrid uses Reciprocal Rank Fusion.

    Args:
        query: the non embedded query
        search_field: Field in Collection containing text
        index_name: Name of Atlas Vector Search Index tied to Collection
        top_k: Number of documents to return
        oversampling_factor: this times limit is the number of candidates
        filter: MQL match expression comparing an indexed field.
            Some operators are not supported.
            See `vectorSearch filter docs <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_


    Returns:
        Dictionary defining the $vectorSearch
    """
    # TODO: wait i think this is basically the same as the normal? the path just means something different?
    stage = {
        "index": index_name,
        "path": search_field,
        "query": {"text": query},
        "numCandidates": top_k * oversampling_factor,
        "limit": top_k,
        "model": model_name,
    }
    if filter:
        stage["filter"] = filter
    return {"$vectorSearch": stage}