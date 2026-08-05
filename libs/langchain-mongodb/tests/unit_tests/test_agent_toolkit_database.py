from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from pymongo.errors import OperationFailure

from langchain_mongodb.agent_toolkit.database import MongoDBDatabase


@pytest.fixture
def db() -> MongoDBDatabase:
    database = MongoDBDatabase.__new__(MongoDBDatabase)
    database._include_colls = {"allowed", "related"}
    database._ignore_colls = set()
    database._all_colls = {"allowed", "related", "excluded"}
    database._db = MagicMock()
    _aggregate_mock(database).return_value = [{"count": 1}]
    return database


def _aggregate_mock(db: MongoDBDatabase) -> Any:
    return cast(Any, db._db["allowed"].aggregate)


def test_run_executes_valid_pipeline(db: MongoDBDatabase) -> None:
    pipeline = [
        {
            "$lookup": {
                "from": "related",
                "localField": "related_id",
                "foreignField": "_id",
                "as": "related",
            }
        }
    ]

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run("db.allowed.aggregate([])")

    assert '"count": 1' in result
    _aggregate_mock(db).assert_called_once_with(pipeline)


@pytest.mark.parametrize(
    "pipeline",
    [
        [{"$out": "related"}],
        [{"$merge": {"into": "related"}}],
        [{"$match": {"$where": "return true"}}],
        [{"$project": {"value": {"$function": {"body": "return 1"}}}}],
        [
            {
                "$group": {
                    "_id": None,
                    "value": {"$accumulator": {"init": "function() {}"}},
                }
            }
        ],
    ],
)
def test_run_rejects_blocked_operators(
    db: MongoDBDatabase, pipeline: list[dict[str, object]]
) -> None:
    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run_no_throw("db.allowed.aggregate([])")

    assert isinstance(result, str)
    assert result.startswith("Error: Aggregation operator $")
    _aggregate_mock(db).assert_not_called()


@pytest.mark.parametrize(
    "pipeline",
    [
        [{"$lookup": {"from": "excluded", "pipeline": [], "as": "result"}}],
        [{"$graphLookup": {"from": "excluded", "startWith": "$_id"}}],
        [{"$unionWith": "excluded"}],
        [{"$unionWith": {"coll": "excluded", "pipeline": []}}],
    ],
)
def test_run_enforces_collection_boundaries(
    db: MongoDBDatabase, pipeline: list[dict[str, object]]
) -> None:
    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run_no_throw("db.allowed.aggregate([])")

    assert result == "Error: Collection excluded is not available."
    _aggregate_mock(db).assert_not_called()


def test_run_validates_facet_pipeline(db: MongoDBDatabase) -> None:
    pipeline = [
        {
            "$facet": {
                "unsafe": [{"$lookup": {"from": "excluded", "pipeline": [], "as": "x"}}]
            }
        }
    ]

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run_no_throw("db.allowed.aggregate([])")

    assert result == "Error: Collection excluded is not available."
    _aggregate_mock(db).assert_not_called()


@pytest.mark.parametrize("stage_name", ["$rankFusion", "$scoreFusion"])
def test_run_validates_fusion_pipeline(db: MongoDBDatabase, stage_name: str) -> None:
    pipeline = [
        {
            stage_name: {
                "input": {
                    "pipelines": {
                        "unsafe": [
                            {
                                "$lookup": {
                                    "from": "excluded",
                                    "pipeline": [],
                                    "as": "x",
                                }
                            }
                        ]
                    }
                }
            }
        }
    ]

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run_no_throw("db.allowed.aggregate([])")

    assert result == "Error: Collection excluded is not available."
    _aggregate_mock(db).assert_not_called()


@pytest.mark.parametrize("stage_name", ["$lookup", "$unionWith"])
def test_run_allows_collectionless_documents_pipeline(
    db: MongoDBDatabase, stage_name: str
) -> None:
    pipeline = [{stage_name: {"pipeline": [{"$documents": [{"value": 1}]}]}}]

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run("db.allowed.aggregate([])")

    assert '"count": 1' in result
    _aggregate_mock(db).assert_called_once_with(pipeline)


def test_run_allows_blocked_operator_name_inside_literal(db: MongoDBDatabase) -> None:
    pipeline = [{"$project": {"value": {"$literal": {"$function": "data"}}}}]

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=pipeline,
    ):
        result = db.run("db.allowed.aggregate([])")

    assert '"count": 1' in result
    _aggregate_mock(db).assert_called_once_with(pipeline)


def test_run_no_throw_hides_driver_error_details(db: MongoDBDatabase) -> None:
    _aggregate_mock(db).side_effect = OperationFailure(
        "document containing private value was rejected"
    )

    with patch(
        "langchain_mongodb.agent_toolkit.database.parse_command",
        return_value=[],
    ):
        result = db.run_no_throw("db.allowed.aggregate([])")

    assert result == "Error: Error executing aggregation."
    assert "private value" not in result


def test_collection_info_no_throw_hides_driver_error_details(
    db: MongoDBDatabase,
) -> None:
    with patch.object(
        db,
        "get_collection_info",
        side_effect=OperationFailure("private document details"),
    ):
        result = db.get_collection_info_no_throw()

    assert result == "Error: collection information could not be retrieved."
    assert "private document details" not in result
