"""
:private:
Utilities for langgraph-store-mongodb.
"""

from importlib.metadata import version

from pymongo import MongoClient
from pymongo.driver_info import DriverInfo
from pymongo_search_utils import append_client_metadata

DRIVER_METADATA = DriverInfo(
    name="Langgraph", version=version("langgraph-store-mongodb")
)


def _append_client_metadata(client: MongoClient) -> None:
    append_client_metadata(client=client, driver_info=DRIVER_METADATA)
