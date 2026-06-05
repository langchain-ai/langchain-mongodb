"""Aggregation pipeline components used in Atlas Full-Text, Vector, and Hybrid Search

See the following for more:
    - `Full-Text Search <https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search>`_
    - `MongoDB Operators <https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref>`_
    - `Vector Search <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/>`_
    - `Filter Example <https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter>`_
"""

from typing import Any, Dict, List, Optional, Union

from pymongo_search_utils import (
    autoembedding_vector_search_stage,  # noqa: F401
    combine_pipelines,  # noqa: F401
    final_hybrid_stage,  # noqa: F401
    reciprocal_rank_stage,  # noqa: F401
    vector_search_stage,  # noqa: F401
)


def rerank_stage(
    query: str,
    path: Union[str, List[str]],
    num_docs_to_rerank: int,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """$rerank aggregation stage for Native Reranking in Atlas.

    Requires MongoDB 8.3+ and Native Reranking enabled via Atlas Project Settings.
    Best used after a $search, $vectorSearch, $rankFusion, or $scoreFusion stage.

    Will migrate to pymongo_search_utils once available there.

    Args:
        query: Text query used for reranking
        path: Field or list of fields to rerank on
        num_docs_to_rerank: Number of documents to pass to the reranker (max 1000)
        model: Voyage AI reranking model (e.g. "rerank-2.5", "rerank-2", "rerank-2.5-lite").
            Omit to use the latest available model.

    Returns:
        List of pipeline stages: $rerank followed by $set to update the score field
    """
    spec: Dict[str, Any] = {
        "query": {"text": query},
        "path": path,
        "numDocsToRerank": num_docs_to_rerank,
    }
    if model is not None:
        spec["model"] = model
    return [
        {"$rerank": spec},
        {"$set": {"score": {"$meta": "score"}, "rerankScore": {"$meta": "score"}}},
    ]


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
