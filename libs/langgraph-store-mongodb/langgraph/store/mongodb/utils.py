"""
:private:
Utilities for langgraph-store-mongodb.
"""

from importlib.metadata import version

from pymongo.driver_info import DriverInfo

DRIVER_METADATA = DriverInfo(
    name="Langgraph", version=version("langgraph-store-mongodb")
)
