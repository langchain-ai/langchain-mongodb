from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_mongodb.agent_toolkit.tool import (
    InfoMongoDBDatabaseTool,
    ListMongoDBDatabaseTool,
    QueryMongoDBCheckerTool,
    QueryMongoDBDatabaseTool,
)
from tests.utils import create_database, create_llm


class TestQueryMongoDBDatabaseToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[QueryMongoDBDatabaseTool]:
        return QueryMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=create_database())

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(query='db.test.aggregate([{"$match": {}}])')


class TestInfoMongoDBDatabaseToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[InfoMongoDBDatabaseTool]:
        return InfoMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=create_database())

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(collection_names="test")


class TestListMongoDBDatabaseToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ListMongoDBDatabaseTool]:
        return ListMongoDBDatabaseTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=create_database())

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict()


class TestQueryMongoDBCheckerToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[QueryMongoDBCheckerTool]:
        return QueryMongoDBCheckerTool

    @property
    def tool_constructor_params(self) -> dict:
        return dict(db=create_database(), llm=create_llm())

    @property
    def tool_invoke_params_example(self) -> dict:
        return dict(query='db.test.aggregate([{"$match": {}}])')
