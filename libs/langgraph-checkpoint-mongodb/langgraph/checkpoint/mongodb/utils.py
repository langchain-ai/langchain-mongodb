from typing import Any, Callable, Dict, Union

from langgraph.checkpoint.base import CheckpointMetadata


def prepare_metadata(
    prepare: Callable, metadata: Union[CheckpointMetadata, Any]
) -> Union[bytes, Dict[str, Any]]:
    """Recursively serialize or deserialize all values in metadata dictionary.

    The CheckpointMetadata class itself cannot be stored directly in MongoDB,
    but as a dictionary it can. For efficient filtering in MongoDB,
    we keep dict keys as strings.

    The `prepare` function is one of the methods, `dumps` or `loads`,
    of :class:`~langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`.
    On `dumps`, one goes from :class:`~langgraph.checkpoint.base.CheckpointMetadata` -> Dict[str, bytes]
    On `loads`, one goes from `Dict[str, Any]` -> `CheckpointMetadata`.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = prepare_metadata(prepare, value)
        return output
    else:
        return prepare(metadata)
