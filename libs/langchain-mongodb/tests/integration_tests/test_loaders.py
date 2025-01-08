from typing import Dict, List

from langchain_core.documents import Document
from pymongo import MongoClient

from langchain_mongodb.loaders import MongoDBLoader

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_loader"


def raw_docs() -> List[Dict]:
    return [
        {"_id": "1", "address": {"building": "1", "room": "1"}},
        {"_id": "2", "address": {"building": "2", "room": "2"}},
        {"_id": "3", "address": {"building": "3", "room": "2"}},
    ]


def expected_documents() -> List[Document]:
    return [
        Document(
            page_content="2 2",
            metadata={"_id": "2", "database": DB_NAME, "collection": COLLECTION_NAME},
        ),
        Document(
            page_content="3 2",
            metadata={"_id": "3", "database": DB_NAME, "collection": COLLECTION_NAME},
        ),
    ]


async def test_load_with_filters(client: MongoClient) -> None:
    filter_criteria = {"address.room": {"$eq": "2"}}
    field_names = ["address.building", "address.room"]
    metadata_names = ["_id"]
    include_db_collection_in_metadata = True

    collection = client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.insert_many(raw_docs())

    loader = MongoDBLoader(
        collection,
        filter_criteria=filter_criteria,
        field_names=field_names,
        metadata_names=metadata_names,
        include_db_collection_in_metadata=include_db_collection_in_metadata,
    )
    documents = await loader.aload()

    assert documents == expected_documents()
