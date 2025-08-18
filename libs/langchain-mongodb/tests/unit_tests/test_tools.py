from __future__ import annotations

from datetime import datetime
from typing import Type

from bson import ObjectId
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_mongodb.agent_toolkit import MongoDBDatabase
from langchain_mongodb.agent_toolkit.tool import (
    InfoMongoDBDatabaseTool,
    ListMongoDBDatabaseTool,
    QueryMongoDBCheckerTool,
    QueryMongoDBDatabaseTool,
)

from ..utils import FakeLLM, MockClient


class TestQueryMongoDBDatabaseToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[QueryMongoDBDatabaseTool]:
        return QueryMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=MongoDBDatabase(MockClient(), "test"))  # type:ignore[arg-type]

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(query="db.foo.aggregate()")


class TestInfoMongoDBDatabaseToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[InfoMongoDBDatabaseTool]:
        return InfoMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=MongoDBDatabase(MockClient(), "test"))  # type:ignore[arg-type]

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(collection_names="test")


class TestListMongoDBDatabaseToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[ListMongoDBDatabaseTool]:
        return ListMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=MongoDBDatabase(MockClient(), "test"))  # type:ignore[arg-type]

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict()


class TestQueryMongoDBCheckerToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[QueryMongoDBCheckerTool]:
        return QueryMongoDBCheckerTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=MongoDBDatabase(MockClient(), "test"), llm=FakeLLM())  # type:ignore[arg-type]

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(query="db.foo.aggregate()")


def test_database_parse_command() -> None:
    db = MongoDBDatabase(MockClient(), "test")  # type:ignore[arg-type]

    command = """db.user.aggregate([ { "$match": { "_id": ObjectId("123412341234123412341234") } } ])"""
    result = db._parse_command(command)
    assert isinstance(result[0]["$match"]["_id"], ObjectId)

    command = """db.user.aggregate([ { "$match": { "date":  ISODate("2017-04-27T04:26:42.709Z") } } ])"""
    result = db._parse_command(command)
    assert isinstance(result[0]["$match"]["date"], datetime)

    command = """db.user.aggregate([ { "$match": { "date": new Date("2017-04-27T04:26:42.709Z") } } ])"""
    result = db._parse_command(command)
    assert isinstance(result[0]["$match"]["date"], datetime)
