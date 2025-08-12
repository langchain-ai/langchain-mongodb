"""
:private:
Utilities for langgraph-store-mongodb.
"""

from importlib.metadata import version

from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

DRIVER_METADATA = DriverInfo(
    name="Langgraph", version=version("langgraph-store-mongodb")
)


def _append_client_metadata(client: MongoClient) -> None:
    # append_metadata was added in PyMongo 4.14.0, but is a valid database name on earlier versions
    if callable(client.append_metadata):
        client.append_metadata(DRIVER_METADATA)
