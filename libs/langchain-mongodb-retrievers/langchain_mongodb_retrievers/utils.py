"""Various Utility Functions"""

from __future__ import annotations

import logging
from importlib.metadata import version

from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

logger = logging.getLogger(__name__)

DRIVER_METADATA = DriverInfo(
    name="langchain-mongodb-retrievers", version=version("langchain-mongodb-retrievers")
)


def _append_client_metadata(client: MongoClient) -> None:
    # append_metadata was added in PyMongo 4.14.0, but is a valid database name on earlier versions
    if callable(client.append_metadata):
        client.append_metadata(DRIVER_METADATA)
