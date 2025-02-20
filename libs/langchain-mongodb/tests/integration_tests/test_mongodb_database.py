# flake8: noqa: E501
"""Test MongoDB database wrapper."""

import json

import pytest
from pymongo import MongoClient

from langchain_mongodb.agent_toolkit import MongoDBDatabase


@pytest.fixture
def db(client: MongoClient) -> MongoDBDatabase:
    client.drop_database("test")
    user = dict(name="Alice", bio="Engineer from Ohio")
    client["test"]["user"].insert_one(user)
    company = dict(name="Acme", location="Montana")
    client["test"]["company"].insert_one(company)
    return MongoDBDatabase(client, "test")


def test_collection_info(db: MongoDBDatabase) -> None:
    """Test that collection info is constructed properly."""
    output = db.collection_info
    expected_output = """
    Database name: test
    Collection name: company
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    location: String

    /*
    3 documents from company collection:
    [
      {
        "_id": {
          "$oid": "..."
        },
        "name": "Acme",
        "location": "Montana"
      }
    ]
    */

    Database name: test
    Collection name: user
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    bio: String

    /*
    3 documents from user collection:
    [
      {
        "_id": {
          "$oid": "..."
        },
        "name": "Alice",
        "bio": "Engineer from Ohio"
      }
    ]
    */
    """.strip()
    for line1, line2 in zip(output.splitlines(), expected_output.splitlines()):
        if "$oid" in line1:
            continue
        assert line1.strip() == line2.strip()


def test_collection_info_w_sample_docs(db: MongoDBDatabase) -> None:
    """Test that collection info is constructed properly."""

    # Provision.
    values = [
        {"name": "Harrison", "bio": "bio"},
        {"name": "Chase", "bio": "bio"},
    ]
    db._client["test"]["user"].delete_many({})
    db._client["test"]["user"].insert_many(values)

    # Query and verify.
    db = MongoDBDatabase(db._client, "test", sample_docs_in_collection_info=2)
    output = db.collection_info

    expected_output = """
    Database name: test
    Collection name: company
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    location: String

    /*
    2 documents from company collection:
    [
      {
        "_id": {
          "$oid": "..."
        },
        "name": "Acme",
        "location": "Montana"
      }
    ]
    */

    Database name: test
    Collection name: user
    Schema from a sample of documents from the collection:
    _id: ObjectId
    name: String
    bio: String

    /*
    2 documents from user collection:
    [
      {
        "_id": {
          "$oid": "..."
        },
        "name": "Harrison",
        "bio": "bio"
      },
            {
        "_id": {
          "$oid": "..."
        },
        "name": "Chase",
        "bio": "bio"
      }
    ]
    */""".strip()

    for line1, line2 in zip(output.splitlines(), expected_output.splitlines()):
        if "$oid" in line1:
            continue
        assert line1.strip() == line2.strip()


def test_database_run(db: MongoDBDatabase) -> None:
    """Verify running MQL expressions returning results as strings."""

    # Provision.
    db._client["test"]["user"].delete_many({})
    user = dict(name="Harrison", bio="That is my Bio " * 24)
    db._client["test"]["user"].insert_one(user)

    # Query and verify.
    command = """db.user.aggregate([ { $match: { name: 'Harrison' } } ])"""
    output = db.run(command)
    docs = json.loads(output.strip())
    del docs[0]["_id"]
    del user["_id"]
    assert docs[0] == user


CMDS = [
    """db.Invoice.aggregate([ { $group: { _id: "$BillingCountry", totalSpent: { $sum: "$Total" } } }, { $sort: { totalSpent: -1 } }, { $limit: 5 } ])""",
    """db.Invoice.aggregate([{ $group: { _id: '$BillingCountry', totalSpent: { $sum: '$Total' } } }, { $sort: { totalSpent: -1 } }, { $limit: 5 }])""",
    """db.Invoice.aggregate([{$group: {_id: "$BillingCountry", totalSpent: {$sum: "$Total"}}}, {$sort: {totalSpent: -1}}, {$limit: 5}])""",
    """db.Invoice.aggregate([ { "$group": { "_id": "$BillingCountry", "totalSpent": { "$sum": "$Total" } } }, { "$sort": { "totalSpent": -1 } }, { "$limit": 5 } ])""",
    """db.InvoiceLine.aggregate([ { $group: { _id: "$InvoiceId", totalSpent: { $sum: { $multiply: [ "$UnitPrice", "$Quantity" ] }} }}, { $lookup: { from: "Customer", localField: "_id", foreignField: "CustomerId", as: "customerInfo" }}, { $unwind: "$customerInfo" }, { $group: { _id: "$customerInfo.Country", totalSpent: { $sum: "$totalSpent" } }}, { $sort: { totalSpent: -1 }}, { $limit: 5 } ])""",
    """db.Invoice.aggregate([ { $group: { _id: "$BillingCountry", totalSpent: { $sum: "$Total" } } }, { $sort: { totalSpent: -1 } }, { $limit: 5 } ])""",
]


def test_database_parse_command(db: MongoDBDatabase) -> None:
    for cmd in CMDS:
        db._parse_command(cmd)
