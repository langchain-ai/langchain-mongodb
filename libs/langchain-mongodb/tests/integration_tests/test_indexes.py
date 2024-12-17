import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from pymongo import MongoClient

from langchain_mongodb.indexes import MongoDBRecordManager

CONNECTION_STRING = os.environ["MONGODB_URI"]
DB_NAME = "langchain_test_db"
COLLECTION_NAME = "langchain_test_docstore"
NAMESPACE = f"{DB_NAME}.{COLLECTION_NAME}"


@pytest.fixture
def manager() -> MongoDBRecordManager:
    """Initialize the test MongoDB and yield the DocumentManager instance."""
    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]
    document_manager = MongoDBRecordManager(collection=collection)
    return document_manager


@pytest_asyncio.fixture
async def amanager() -> MongoDBRecordManager:
    """Initialize the test MongoDB and yield the DocumentManager instance."""
    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]
    document_manager = MongoDBRecordManager(collection=collection)
    return document_manager


def test_update(manager: MongoDBRecordManager) -> None:
    """Test updating records in the MongoDB."""
    read_keys = manager.list_keys()
    updated_keys = ["update_key1", "update_key2", "update_key3"]
    manager.update(updated_keys)
    all_keys = manager.list_keys()
    assert sorted(all_keys) == sorted(read_keys + updated_keys)


async def test_aupdate(amanager: MongoDBRecordManager) -> None:
    """Test updating records in the MongoDB."""
    read_keys = await amanager.alist_keys()
    aupdated_keys = ["aupdate_key1", "aupdate_key2", "aupdate_key3"]
    await amanager.aupdate(aupdated_keys)
    all_keys = await amanager.alist_keys()
    assert sorted(all_keys) == sorted(read_keys + aupdated_keys)


def test_update_timestamp(manager: MongoDBRecordManager) -> None:
    """Test updating records with timestamps in MongoDB."""
    with patch.object(
        manager, "get_time", return_value=datetime(2024, 2, 23).timestamp()
    ):
        manager.update(["key1"])
    records = list(
        manager._collection.find({"namespace": manager.namespace, "key": "key1"})
    )

    assert [
        {
            "key": record["key"],
            "namespace": record["namespace"],
            "updated_at": record["updated_at"],
            "group_id": record.get("group_id"),
        }
        for record in records
    ] == [
        {
            "group_id": None,
            "key": "key1",
            "namespace": NAMESPACE,
            "updated_at": datetime(2024, 2, 23).timestamp(),
        }
    ]


async def test_aupdate_timestamp(amanager: MongoDBRecordManager) -> None:
    """Test asynchronously updating records with timestamps in MongoDB."""
    with patch.object(
        amanager, "get_time", return_value=datetime(2024, 2, 23).timestamp()
    ):
        await amanager.aupdate(["key1"])

    records = [
        doc
        for doc in amanager._collection.find(
            {"namespace": amanager.namespace, "key": "key1"}
        )
    ]

    assert [
        {
            "key": record["key"],
            "namespace": record["namespace"],
            "updated_at": record["updated_at"],
            "group_id": record.get("group_id"),
        }
        for record in records
    ] == [
        {
            "group_id": None,
            "key": "key1",
            "namespace": NAMESPACE,
            "updated_at": datetime(2024, 2, 23).timestamp(),
        }
    ]


def test_exists(manager: MongoDBRecordManager) -> None:
    """Test checking if keys exist in MongoDB."""
    keys = ["key1", "key2", "key3"]
    manager.update(keys)
    exists = manager.exists(keys)
    assert len(exists) == len(keys)
    assert all(exists)

    exists = manager.exists(["key1", "key4"])
    assert len(exists) == 2
    assert exists == [True, False]


async def test_aexists(amanager: MongoDBRecordManager) -> None:
    """Test asynchronously checking if keys exist in MongoDB."""
    keys = ["key1", "key2", "key3"]
    await amanager.aupdate(keys)
    exists = await amanager.aexists(keys)
    assert len(exists) == len(keys)
    assert all(exists)

    exists = await amanager.aexists(["key1", "key4"])
    assert len(exists) == 2
    assert exists == [True, False]


def test_list_keys(manager: MongoDBRecordManager) -> None:
    """Test listing keys in MongoDB."""
    manager.delete_keys(manager.list_keys())
    with patch.object(
        manager, "get_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        manager.update(["key1"])
    with patch.object(
        manager, "get_time", return_value=datetime(2022, 1, 1).timestamp()
    ):
        manager.update(["key2"])
    with patch.object(
        manager, "get_time", return_value=datetime(2023, 1, 1).timestamp()
    ):
        manager.update(["key3"])
    with patch.object(
        manager, "get_time", return_value=datetime(2024, 1, 1).timestamp()
    ):
        manager.update(["key4"], group_ids=["group1"])
    assert sorted(manager.list_keys()) == sorted(["key1", "key2", "key3", "key4"])
    assert sorted(manager.list_keys(after=datetime(2022, 2, 1).timestamp())) == sorted(
        ["key3", "key4"]
    )
    assert sorted(manager.list_keys(group_ids=["group1", "group2"])) == sorted(["key4"])


async def test_alist_keys(amanager: MongoDBRecordManager) -> None:
    """Test asynchronously listing keys in MongoDB."""
    await amanager.adelete_keys(await amanager.alist_keys())
    with patch.object(
        amanager, "get_time", return_value=datetime(2021, 1, 1).timestamp()
    ):
        await amanager.aupdate(["key1"])
    with patch.object(
        amanager, "get_time", return_value=datetime(2022, 1, 1).timestamp()
    ):
        await amanager.aupdate(["key2"])
    with patch.object(
        amanager, "get_time", return_value=datetime(2023, 1, 1).timestamp()
    ):
        await amanager.aupdate(["key3"])
    with patch.object(
        amanager, "get_time", return_value=datetime(2024, 1, 1).timestamp()
    ):
        await amanager.aupdate(["key4"], group_ids=["group1"])
    assert sorted(await amanager.alist_keys()) == sorted(
        ["key1", "key2", "key3", "key4"]
    )
    assert sorted(
        await amanager.alist_keys(after=datetime(2022, 2, 1).timestamp())
    ) == sorted(["key3", "key4"])
    assert sorted(await amanager.alist_keys(group_ids=["group1", "group2"])) == sorted(
        ["key4"]
    )


def test_namespace_is_used(manager: MongoDBRecordManager) -> None:
    """Verify that namespace is taken into account for all operations in MongoDB."""
    manager.delete_keys(manager.list_keys())
    manager.update(["key1", "key2"], group_ids=["group1", "group2"])
    manager._collection.insert_many(
        [
            {"key": "key1", "namespace": "puppies", "group_id": None},
            {"key": "key3", "namespace": "puppies", "group_id": None},
        ]
    )
    assert sorted(manager.list_keys()) == sorted(["key1", "key2"])
    manager.delete_keys(["key1"])
    assert sorted(manager.list_keys()) == sorted(["key2"])
    manager.update(["key3"], group_ids=["group3"])
    doc = manager._collection.find_one({"key": "key3", "namespace": NAMESPACE})
    assert doc is not None
    assert doc["group_id"] == "group3"


async def test_anamespace_is_used(amanager: MongoDBRecordManager) -> None:
    """
    Verify that namespace is taken into account for all operations
    in MongoDB asynchronously.
    """
    await amanager.adelete_keys(await amanager.alist_keys())
    await amanager.aupdate(["key1", "key2"], group_ids=["group1", "group2"])
    amanager._collection.insert_many(
        [
            {"key": "key1", "namespace": "puppies", "group_id": None},
            {"key": "key3", "namespace": "puppies", "group_id": None},
        ]
    )
    assert sorted(await amanager.alist_keys()) == sorted(["key1", "key2"])
    await amanager.adelete_keys(["key1"])
    assert sorted(await amanager.alist_keys()) == sorted(["key2"])
    await amanager.aupdate(["key3"], group_ids=["group3"])
    doc = amanager._collection.find_one({"key": "key3", "namespace": NAMESPACE})
    assert doc is not None
    assert doc["group_id"] == "group3"


def test_delete_keys(manager: MongoDBRecordManager) -> None:
    """Test deleting keys from MongoDB."""
    manager.update(["key1", "key2", "key3"])
    manager.delete_keys(["key1", "key2"])
    remaining_keys = manager.list_keys()
    assert sorted(remaining_keys) == sorted(["key3"])


async def test_adelete_keys(amanager: MongoDBRecordManager) -> None:
    """Test asynchronously deleting keys from MongoDB."""
    await amanager.aupdate(["key1", "key2", "key3"])
    await amanager.adelete_keys(["key1", "key2"])
    remaining_keys = await amanager.alist_keys()
    assert sorted(remaining_keys) == sorted(["key3"])
