"""Search Index Commands"""

import logging
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional

from langchain_mongodb.embeddings import AutoEmbedding
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel
from pymongo_search_utils import (
    create_fulltext_search_index,  # noqa: F401
    create_vector_search_index,  # noqa: F401
    drop_vector_search_index,  # noqa: F401
    update_vector_search_index,  # noqa: F401
)
from pymongo_search_utils.index import is_index_ready, wait_for_predicate

logger = logging.getLogger(__file__)


def _vector_search_index_definition(
    dimensions: int,
    path: str,
    similarity: str,
    filters: Optional[List[str]] = None,
    vector_index_options: dict | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    vector_index_options = vector_index_options or {}
    fields = [
        {
            "numDimensions": dimensions,
            "path": path,
            "similarity": similarity,
            "type": "vector",
            **vector_index_options,
        },
    ]
    if filters:
        for field in filters:
            fields.append({"type": "filter", "path": field})
    definition = {"fields": fields}
    definition.update(kwargs)
    return definition

def vector_search_autoembedded_index_definition(
    path: str,
    embedding: AutoEmbedding,
    filters: list[str] | None = None,
    vector_index_options: dict | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a vector search index with autoembeddings e definition.

    Args:
        path (str): The name of the text field to be indexed.
        embedding (AutoEmbedding): The autoembedding model to use.
        similarity (str): The type of similarity metric to use.
        One of "euclidean", "cosine", or "dotProduct".
        filters (Optional[List[str]]): If provided, a list of fields to filter on
        in addition to the vector search.
        kwargs (Any): Keyword arguments supplying any additional options to the vector search index.

    Returns:
        A dictionary representing the vector search index definition.
    """
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    fields = [
        {
            "type": "autoEmbed",
            "path": path,
            "model": embedding.model_name,
            "modality": "text",
            **(vector_index_options or {}),
        },
    ]
    if filters:
        for field in filters:
            fields.append({"type": "filter", "path": field})
    definition = {"fields": fields}
    definition.update(kwargs)
    return definition

def create_autoembedded_vector_search_index(
    collection: Collection[Any],
    index_name: str,
    path: str,
    embedding: AutoEmbedding,
    filters: Optional[List[str]] = None,
    update: bool = False,
    wait_until_complete: Optional[float] = None,
    vector_index_options: dict | None = None,
    **kwargs: Any,
) -> None:
    try:
        collection.database.create_collection(collection.name)
    except CollectionInvalid:
        pass

    definition = vector_search_autoembedded_index_definition(
        path=path,
        embedding=embedding,
        filters=filters,
        vector_index_options=vector_index_options,
        **kwargs,
    )
    print(definition)

    if update:
        collection.update_search_index(
            name=index_name,
            definition=definition,
        )
        if wait_until_complete:
            wait_for_predicate(
                predicate=lambda: is_index_ready(collection, index_name),
                err=f"Index {index_name} update did not complete in {wait_until_complete}!",
                timeout=wait_until_complete,
            )
    else:
        if collection.name not in collection.database.list_collection_names():
            collection.database.create_collection(collection.name)

        result = collection.create_search_index(
            SearchIndexModel(
                definition=definition,
                name=index_name,
                type="vectorSearch",
            )
        )

        if wait_until_complete:
            wait_for_predicate(
                predicate=lambda: is_index_ready(collection, index_name),
                err=f"{index_name=} did not complete in {wait_until_complete}!",
                timeout=wait_until_complete,
            )

def _is_index_ready(collection: Collection, index_name: str) -> bool:
    """Check for the index name in the list of available search indexes to see if the
    specified index is of status READY

    Args:
        collection (Collection): MongoDB Collection to for the search indexes
        index_name (str): Vector Search Index name

    Returns:
        bool : True if the index is present and READY false otherwise
    """
    for index in collection.list_search_indexes(index_name):
        if index["status"] == "READY":
            return True
    return False


def _wait_for_predicate(
    predicate: Callable, err: str, timeout: float = 120, interval: float = 0.5
) -> None:
    """Generic to block until the predicate returns true

    Args:
        predicate (Callable[, bool]): A function that returns a boolean value
        err (str): Error message to raise if nothing occurs
        timeout (float, optional): Wait time for predicate. Defaults to TIMEOUT.
        interval (float, optional): Interval to check predicate. Defaults to DELAY.

    Raises:
        TimeoutError: _description_
    """
    start = monotonic()
    while not predicate():
        if monotonic() - start > timeout:
            raise TimeoutError(err)
        sleep(interval)
