"""Wrapper around a MongoDB database."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from importlib.metadata import version
from typing import Any, Dict, Iterable, List, Optional, Union

from bson import ObjectId
from bson.binary import Binary
from bson.decimal128 import Decimal128
from bson.json_util import dumps
from pymongo import MongoClient
from pymongo.cursor import Cursor
from pymongo.driver_info import DriverInfo
from pymongo.errors import PyMongoError

NUM_DOCUMENTS_TO_SAMPLE = 4
MAX_STRING_LENGTH_OF_SAMPLE_DOCUMENT_VALUE = 20

_BSON_LOOKUP = {
    str: "String",
    int: "Number",
    float: "Number",
    bool: "Boolean",
    ObjectId: "ObjectId",
    date: "Date",
    datetime: "Timestamp",
    None: "Null",
    Decimal128: "Decimal128",
    Binary: "Binary",
}


class MongoDBDatabase:
    """Wrapper around a MongoDB database."""

    def __init__(
        self,
        client: MongoClient,
        database: str,
        schema: Optional[str] = None,
        ignore_collections: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
        sample_docs_in_collection_info: int = 3,
        indexes_in_collection_info: bool = False,
    ):
        """Create a MongoDBDatabase from client and database name."""
        self._client = client
        self._db = client[database]
        self._schema = schema
        if include_collections and ignore_collections:
            raise ValueError(
                "Cannot specify both include_collections and ignore_collections"
            )
        self._include_colls = set(include_collections or [])
        self._ignore_colls = set(ignore_collections or [])
        self._all_colls = set(self._db.list_collection_names())

        self._sample_docs_in_coll_info = sample_docs_in_collection_info
        self._indexes_in_coll_info = indexes_in_collection_info

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        database: Optional[str] = None,
        **kwargs: Any,
    ) -> MongoDBDatabase:
        """Construct a MongoDBDatabase from URI."""
        client: MongoClient[dict[str, Any]] = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        )
        database = database or client.get_default_database().name
        return cls(client, database, **kwargs)

    def close(self) -> None:
        """Close the resources used by the MongoDBDatabase."""
        self._client.close()

    def get_usable_collection_names(self) -> Iterable[str]:
        """Get names of collections available."""
        if self._include_colls:
            return sorted(self._include_colls)

        return sorted(self._all_colls - self._ignore_colls)

    @property
    def collection_info(self) -> str:
        """Information about all collections in the database."""
        return self.get_collection_info()

    def get_collection_info(self, collection_names: Optional[List[str]] = None) -> str:
        """Get information about specified collections.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_collection_info`, the specified number of sample rows will be
        appended to each collection description. This can increase performance as
        demonstrated in the paper.
        """
        all_coll_names = self.get_usable_collection_names()
        if collection_names is not None:
            missing_collections = set(collection_names).difference(all_coll_names)
            if missing_collections:
                raise ValueError(
                    f"collection_names {missing_collections} not found in database"
                )
            all_coll_names = collection_names

        colls = []
        for coll in all_coll_names:
            # add schema
            schema = self._get_collection_schema(coll)
            coll_info = f"Database name: {self._db.name}\n"
            coll_info += f"Collection name: {coll}\n"
            coll_info += f"Schema from a sample of documents from the collection:\n{schema.rstrip()}"
            has_extra_info = (
                self._indexes_in_coll_info or self._sample_docs_in_coll_info
            )
            if has_extra_info:
                coll_info += "\n\n/*"
            if self._indexes_in_coll_info:
                coll_info += f"\n{self._get_collection_indexes(coll)}\n"
            if self._sample_docs_in_coll_info:
                coll_info += f"\n{self._get_sample_docs(coll)}\n"
            if has_extra_info:
                coll_info += "*/"
            colls.append(coll_info)
        colls.sort()
        final_str = "\n\n".join(colls)
        return final_str

    def _get_collection_schema(self, collection: str) -> str:
        coll = self._db[collection]
        doc = coll.find_one({}) or dict()
        return "\n".join(self._parse_doc(doc, ""))

    def _parse_doc(self, doc: dict[str, Any], prefix: str) -> list[str]:
        sub_schema = []
        for key, value in doc.items():
            if prefix:
                full_key = f"{prefix}.{key}"
            else:
                full_key = key
            if isinstance(value, dict):
                sub_schema.extend(self._parse_doc(value, full_key))
            elif isinstance(value, list):
                if not len(value):
                    sub_schema.append(f"{full_key}: Array")
                elif isinstance(value[0], dict):
                    sub_schema.extend(self._parse_doc(value[0], f"{full_key}[]"))
                else:
                    if type(value[0]) in _BSON_LOOKUP:
                        type_name = _BSON_LOOKUP[type(value[0])]
                        sub_schema.append(f"{full_key}: Array<{type_name}>")
                    else:
                        sub_schema.append(f"{full_key}: Array")
            elif type(value) in _BSON_LOOKUP:
                type_name = _BSON_LOOKUP[type(value)]
                sub_schema.append(f"{full_key}: {type_name}")
        if not sub_schema:
            sub_schema.append(f"{prefix}: Document")
        return sub_schema

    def _get_collection_indexes(self, collection: str) -> str:
        coll = self._db[collection]
        indexes = list(coll.list_indexes())
        if not indexes:
            return ""
        return f"Collection Indexes:\n{json.dumps(indexes, indent=2)}"

    def _get_sample_docs(self, collection: str) -> str:
        col = self._db[collection]
        docs = list(col.find({}, limit=self._sample_docs_in_coll_info))
        for doc in docs:
            self._elide_doc(doc)
        return (
            f"{self._sample_docs_in_coll_info} documents from {collection} collection:\n"
            f"{dumps(docs, indent=2)}"
        )

    def _elide_doc(self, doc: dict[str, Any]) -> None:
        for key, value in doc.items():
            if isinstance(value, dict):
                self._elide_doc(value)
            elif isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict):
                        self._elide_doc(item)
                    elif (
                        isinstance(item, str)
                        and len(item) > MAX_STRING_LENGTH_OF_SAMPLE_DOCUMENT_VALUE
                    ):
                        item = item[: MAX_STRING_LENGTH_OF_SAMPLE_DOCUMENT_VALUE + 1]
                    items.append(item)
                doc[key] = items
            elif (
                isinstance(value, str)
                and len(value) > MAX_STRING_LENGTH_OF_SAMPLE_DOCUMENT_VALUE
            ):
                doc[key] = value[: MAX_STRING_LENGTH_OF_SAMPLE_DOCUMENT_VALUE + 1]

    def _parse_command(self, command: str) -> Any:
        # Convert a JavaScript command to a python object.
        command = re.sub(r"\s+", " ", command.strip())
        # Handle missing closing parens.
        if command.endswith("]"):
            command += ")"
        agg_command = command[command.index("[") : -1]
        try:
            return json.loads(agg_command)
        except Exception as e:
            raise ValueError(f"Cannot execute command {command}") from e

    def run(self, command: str) -> Union[str, Cursor]:
        """Execute a MongoDB aggregation command and return a string representing the results.

        If the statement returns documents, a string of the results is returned.
        If the statement returns no documents, an empty string is returned.

        The command MUST be of the form: `db.collectionName.aggregate(...)`.
        """
        if not command.startswith("db."):
            raise ValueError(f"Cannot run command {command}")
        col_name = command.split(".")[1]
        if col_name not in self.get_usable_collection_names():
            raise ValueError(f"Collection {col_name} does not exist!")
        coll = self._db[col_name]
        if ".aggregate(" not in command:
            raise ValueError(f"Cannot execute command {command}")
        agg = self._parse_command(command)
        return dumps(list(coll.aggregate(agg)), indent=2)

    def get_collection_info_no_throw(
        self, collection_names: Optional[List[str]] = None
    ) -> str:
        """Get information about specified collections.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_collection_info`, the specified number of sample rows will be
        appended to each collection description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_collection_info(collection_names)
        except ValueError as e:
            """Format the error message"""
            raise e
            return f"Error: {e}"

    def run_no_throw(self, command: str) -> Union[str, Cursor]:
        """Execute a MongoDB command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command)
        except PyMongoError as e:
            """Format the error message"""
            return f"Error: {e}"

    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        collection_names = list(self.get_usable_collection_names())
        collection_info = self.get_collection_info_no_throw()
        return {
            "collection_info": collection_info,
            "collection_names": ", ".join(collection_names),
        }
