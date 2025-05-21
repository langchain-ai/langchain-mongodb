"""Tools for interacting with a MongoDB database."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pymongo.cursor import Cursor

from .database import MongoDBDatabase
from .prompt import MONGODB_QUERY_CHECKER


class BaseMongoDBDatabaseTool(BaseModel):
    """Base tool for interacting with a MongoDB database."""

    db: MongoDBDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QueryMongoDBDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct MongoDB query.")


class QueryMongoDBDatabaseTool(BaseMongoDBDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for querying a MongoDB database."""

    name: str = "mongodb_query"
    description: str = """
    Execute a MongoDB query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QueryMongoDBDatabaseToolInput

    def _run(self, query: str, **kwargs: Any) -> str | Cursor:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class _InfoMongoDBDatabaseToolInput(BaseModel):
    collection_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the collection names for which to return the schema. "
            "Example input: 'collection1, collection2, collection3'"
        ),
    )


class InfoMongoDBDatabaseTool(BaseMongoDBDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting metadata about a MongoDB database."""

    name: str = "mongodb_schema"
    description: str = (
        "Get the schema and sample documents for the specified MongoDB collections."
    )
    args_schema: Type[BaseModel] = _InfoMongoDBDatabaseToolInput

    def _run(
        self,
        collection_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for collections in a comma-separated list."""
        return self.db.get_collection_info_no_throw(
            [t.strip() for t in collection_names.split(",")]
        )


class _ListMongoDBDatabaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListMongoDBDatabaseTool(BaseMongoDBDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting collection names."""

    name: str = "mongodb_list_collections"
    description: str = "Input is an empty string, output is a comma-separated list of collections in the database."
    args_schema: Type[BaseModel] = _ListMongoDBDatabaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a comma-separated list of collection names."""
        return ", ".join(self.db.get_usable_collection_names())


class _QueryMongoDBCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and MongoDB query to be checked.")


class QueryMongoDBCheckerTool(BaseMongoDBDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = MONGODB_QUERY_CHECKER
    llm: BaseLanguageModel
    prompt: PromptTemplate = Field(init=False)
    name: str = "mongodb_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with mongodb_query!
    """
    args_schema: Type[BaseModel] = _QueryMongoDBCheckerToolInput

    @model_validator(mode="before")
    @classmethod
    def initialize_prompt(cls, values: Dict[str, Any]) -> Any:
        if "prompt" not in values:
            values["prompt"] = PromptTemplate(
                template=MONGODB_QUERY_CHECKER, input_variables=["query"]
            )

        if values["prompt"].input_variables != ["query"]:
            raise ValueError(
                "Prompt for QueryCheckerTool must have input variables ['query']"
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        # TODO: check the query using pymongo first.
        chain = self.prompt | self.llm
        return chain.invoke(query)  # type:ignore[arg-type]
