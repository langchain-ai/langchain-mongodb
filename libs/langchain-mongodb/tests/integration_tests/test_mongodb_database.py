# flake8: noqa: E501
"""Test MongoDB database wrapper."""

import json

import pytest
from pymongo import MongoClient

from langchain_mongodb.agent_toolkit import MongoDBDatabase

DB_NAME = "langchain_test_db_user"


@pytest.fixture
def db(client: MongoClient) -> MongoDBDatabase:
    client[DB_NAME].user.delete_many({})
    user = dict(name="Alice", bio="Engineer from Ohio")
    client[DB_NAME]["user"].insert_one(user)
    company = dict(name="Acme", location="Montana")
    client[DB_NAME]["company"].insert_one(company)
    return MongoDBDatabase(client, DB_NAME)


def test_collection_info(db: MongoDBDatabase) -> None:
    """Test that collection info is constructed properly."""
    output = db.collection_info
    expected_header = f"""
    Database name: {DB_NAME}
    Collection name: company
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    location: String""".strip()

    for line in expected_header.splitlines():
        assert line.strip() in output


def test_collection_info_w_sample_docs(db: MongoDBDatabase) -> None:
    """Test that collection info is constructed properly."""

    # Provision.
    values = [
        {"name": "Harrison", "bio": "bio"},
        {"name": "Chase", "bio": "bio"},
    ]
    db._client[DB_NAME]["user"].delete_many({})
    db._client[DB_NAME]["user"].insert_many(values)

    # Query and verify.
    db = MongoDBDatabase(db._client, DB_NAME, sample_docs_in_collection_info=2)
    output = db.collection_info

    expected_header = f"""
    Database name: {DB_NAME}
    Collection name: company
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    location: String
    """.strip()

    for line in expected_header.splitlines():
        assert line.strip() in output


def test_database_run(db: MongoDBDatabase) -> None:
    """Verify running MQL expressions returning results as strings."""

    # Provision.
    db._client[DB_NAME]["user"].delete_many({})
    user = dict(name="Harrison", bio="That is my Bio " * 24)
    db._client[DB_NAME]["user"].insert_one(user)

    # Query and verify.
    command = """db.user.aggregate([ { "$match": { "name": "Harrison" } } ])"""
    output = db.run(command)
    assert isinstance(output, str)
    docs = json.loads(output.strip())
    del docs[0]["_id"]
    del user["_id"]
    assert docs[0] == user
