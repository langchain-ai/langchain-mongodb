# Based on https://github.com/langchain-ai/langchain/blob/edbe7d5f5e0dcc771c1f53a49bb784a3960ce448/libs/community/langchain_community/document_loaders/mongodb.py
from __future__ import annotations

import logging
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo

logger = logging.getLogger(__name__)


class MongoDBLoader(BaseLoader):
    """Document Loaders are classes to load Documents.

    Document Loaders are usually used to load a lot of Documents in a single run."""

    def __init__(
        self,
        collection: Collection,
        *,
        filter_criteria: Optional[Dict] = None,
        field_names: Optional[Sequence[str]] = None,
        metadata_names: Optional[Sequence[str]] = None,
        include_db_collection_in_metadata: bool = True,
    ) -> None:
        """
        Initializes the MongoDB loader with necessary database connection
        details and configurations.

        Args:
            collection (Collection): The pymongo collection to fetch documents from.
            filter_criteria (Optional[Dict]): MongoDB filter criteria for querying
            documents.
            field_names (Optional[Sequence[str]]): List of field names to retrieve
            from documents.
            metadata_names (Optional[Sequence[str]]): Additional metadata fields to
            extract from documents.
            include_db_collection_in_metadata (bool): Flag to include database and
            collection names in metadata.
        """
        self.collection = collection
        self.db = collection.database
        self.db_name = self.db.name
        self.collection_name = collection.name
        self.field_names = field_names or []
        self.filter_criteria = filter_criteria or {}
        self.metadata_names = metadata_names or []
        self.include_db_collection_in_metadata = include_db_collection_in_metadata

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        db_name: str,
        collection_name: str,
        *,
        filter_criteria: Optional[Dict] = None,
        field_names: Optional[Sequence[str]] = None,
        metadata_names: Optional[Sequence[str]] = None,
        include_db_collection_in_metadata: bool = True,
    ) -> MongoDBLoader:
        """
        Creates a MongoDB loader with necessary database connection
        details and configurations.

        Args:
            connection_string (str): MongoDB connection URI.
            db_name (str):Name of the database to connect to.
            collection_name (str): Name of the collection to fetch documents from.
            filter_criteria (Optional[Dict]): MongoDB filter criteria for querying
            documents.
            field_names (Optional[Sequence[str]]): List of field names to retrieve
            from documents.
            metadata_names (Optional[Sequence[str]]): Additional metadata fields to
            extract from documents.
            include_db_collection_in_metadata (bool): Flag to include database and
            collection names in metadata.
        """
        client: MongoClient[dict[str, Any]] = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain-mongodb")),
        )
        collection = client[db_name][collection_name]
        return MongoDBLoader(
            collection,
            filter_criteria=filter_criteria,
            field_names=field_names,
            metadata_names=metadata_names,
            include_db_collection_in_metadata=include_db_collection_in_metadata,
        )

    def close(self) -> None:
        """Close the resources used by the MongoDBLoader."""
        self.db.client.close()

    def load(self) -> List[Document]:
        """Load data into Document objects."""
        result = []
        total_docs = self.collection.count_documents(self.filter_criteria)

        projection = self._construct_projection()

        for doc in self.collection.find(self.filter_criteria, projection):
            metadata = self._extract_fields(doc, self.metadata_names, default="")

            # Optionally add database and collection names to metadata
            if self.include_db_collection_in_metadata:
                metadata.update(
                    {"database": self.db_name, "collection": self.collection_name}
                )

            # Extract text content from filtered fields or use the entire document
            if self.field_names is not None:
                fields = self._extract_fields(doc, self.field_names, default="")
                texts = [str(value) for value in fields.values()]
                text = " ".join(texts)
            else:
                text = str(doc)

            result.append(Document(page_content=text, metadata=metadata))

        if len(result) != total_docs:
            logger.warning(
                f"Only partial collection of documents returned. "
                f"Loaded {len(result)} docs, expected {total_docs}."
            )

        return result

    async def aload(self) -> List[Document]:
        """Asynchronously loads data into Document objects."""
        return await run_in_executor(None, self.load)

    def _construct_projection(self) -> Optional[Dict]:
        """Constructs the projection dictionary for MongoDB query based
        on the specified field names and metadata names."""
        field_names = list(self.field_names) or []
        metadata_names = list(self.metadata_names) or []
        all_fields = field_names + metadata_names
        return {field: 1 for field in all_fields} if all_fields else None

    def _extract_fields(
        self,
        document: Dict,
        fields: Sequence[str],
        default: str = "",
    ) -> Dict:
        """Extracts and returns values for specified fields from a document."""
        extracted = {}
        for field in fields or []:
            value = document
            for key in field.split("."):
                value = value.get(key, default)
                if value == default:
                    break
            new_field_name = field.replace(".", "_")
            extracted[new_field_name] = value
        return extracted
