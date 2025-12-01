"""
:private:
Utilities for langchain-checkpoint-mongod.
"""

from importlib.metadata import version
from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.base import SerializerProtocol
from pymongo import AsyncMongoClient
from pymongo.driver_info import DriverInfo

DRIVER_METADATA = DriverInfo(
    name="Langgraph", version=version("langgraph-checkpoint-mongodb")
)


def _append_client_metadata(client: AsyncMongoClient) -> None:
    # append_metadata was added in PyMongo 4.14.0, but is a valid database name on earlier versions
    if callable(client.append_metadata):
        client.append_metadata(DRIVER_METADATA)


def loads_metadata(
    serde: SerializerProtocol, metadata: dict[str, Any]
) -> CheckpointMetadata:
    """Deserialize metadata document

    The CheckpointMetadata class itself cannot be stored directly in MongoDB,
    but as a dictionary it can. For efficient filtering in MongoDB,
    we keep dict keys as strings.

    metadata is stored in MongoDB collection with string keys and
    serde serialized keys.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = loads_metadata(serde, value)
        return output
    else:
        return serde.loads_typed(metadata)


def dumps_metadata(
    serde: SerializerProtocol,
    metadata: Union[CheckpointMetadata, Any],
) -> Union[bytes, dict[str, Any]]:
    """Serialize all values in metadata dictionary.

    Keep dict keys as strings for efficient filtering in MongoDB
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = dumps_metadata(serde, value)
        return output
    else:
        return serde.dumps_typed(metadata)
