from __future__ import annotations

from typing import Type

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
