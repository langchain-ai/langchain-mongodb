"""Search Index Commands"""

import logging
from time import monotonic, sleep
from typing import Any, Callable, Dict, List, Optional

from pymongo.collection import Collection
from pymongo_search_utils import (
    create_fulltext_search_index,  # noqa: F401
    create_vector_search_index,  # noqa: F401
    drop_vector_search_index,  # noqa: F401
    update_vector_search_index,  # noqa: F401
)

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
