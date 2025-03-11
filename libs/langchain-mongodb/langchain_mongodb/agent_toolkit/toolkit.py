"""Toolkit for interacting with an MongoDB database."""

from typing import List

from langchain_core.caches import BaseCache as BaseCache
from langchain_core.callbacks import Callbacks as Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from .database import MongoDBDatabase
from .tool import (
    InfoMongoDBDatabaseTool,
    ListMongoDBDatabaseTool,
    QueryMongoDBCheckerTool,
    QueryMongoDBDatabaseTool,
)


class MongoDBDatabaseToolkit(BaseToolkit):
    """MongoDBDatabaseToolkit for interacting with MongoDB databases.

    Setup:
        Install ``langchain-mongodb``.

        .. code-block:: bash

            pip install -U langchain-mongodb

    Key init args:
        db: MongoDBDatabase
            The MongoDB database.
        llm: BaseLanguageModel
            The language model (for use with QueryMongoDBCheckerTool)

    Instantiate:
        .. code-block:: python

            from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
            from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
            from langchain_openai import ChatOpenAI

            db = MongoDBDatabase.from_connection_string("mongodb://localhost:27017/chinook")
            llm = ChatOpenAI(temperature=0)

            toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)

    Tools:
        .. code-block:: python

            toolkit.get_tools()

    Use within an agent:
        .. code-block:: python

            from langchain import hub
            from langgraph.prebuilt import create_react_agent
            from mongodb_agent_toolkit.prompt import MONGODB_AGENT_SYSTEM_PROMPT

            # Pull prompt (or define your own)
            system_message = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=5)

            # Create agent
            agent_executor = create_react_agent(
                llm, toolkit.get_tools(), state_modifier=system_message
            )

            # Query agent
            example_query = "Which country's customers spent the most?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()
    """  # noqa: E501

    db: MongoDBDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_mongodb_database_tool = ListMongoDBDatabaseTool(db=self.db)
        info_mongodb_database_tool_description = (
            "Input to this tool is a comma-separated list of collections, output is the "
            "schema and sample rows for those collections. "
            "Be sure that the collectionss actually exist by calling "
            f"{list_mongodb_database_tool.name} first! "
            "Example Input: collection1, collection2, collection3"
        )
        info_mongodb_database_tool = InfoMongoDBDatabaseTool(
            db=self.db, description=info_mongodb_database_tool_description
        )
        query_mongodb_database_tool_description = (
            "Input to this tool is a detailed and correct MongoDB query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_mongodb_database_tool.name} "
            "to query the correct collections fields."
        )
        query_mongodb_database_tool = QueryMongoDBDatabaseTool(
            db=self.db, description=query_mongodb_database_tool_description
        )
        query_mongodb_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_mongodb_database_tool.name}!"
        )
        query_mongodb_checker_tool = QueryMongoDBCheckerTool(
            db=self.db, llm=self.llm, description=query_mongodb_checker_tool_description
        )
        return [
            query_mongodb_database_tool,
            info_mongodb_database_tool,
            list_mongodb_database_tool,
            query_mongodb_checker_tool,
        ]

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()


MongoDBDatabaseToolkit.model_rebuild()
