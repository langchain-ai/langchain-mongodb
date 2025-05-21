import os
import sqlite3

import pytest
import requests
from flaky import flaky  # type:ignore[import-untyped]
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient

from langchain_mongodb.agent_toolkit import (
    MONGODB_AGENT_SYSTEM_PROMPT,
    MongoDBDatabase,
    MongoDBDatabaseToolkit,
)
from tests.utils import CONNECTION_STRING

DB_NAME = "langchain_test_db_chinook"


@pytest.fixture
def db(client: MongoClient) -> MongoDBDatabase:
    # Load the raw data into sqlite.
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript(sql_script)

    # Convert the sqlite data to MongoDB data.
    con.row_factory = sqlite3.Row
    cursor = con.cursor()
    sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor.execute(sql_query)
    tables = [i[0] for i in cursor.fetchall()]
    cursor.close()
    for t in tables:
        coll = client[DB_NAME][t]
        coll.delete_many({})
        cursor = con.cursor()
        cursor.execute(f"select * from {t}")
        docs = [dict(i) for i in cursor.fetchall()]
        cursor.close()
        coll.insert_many(docs)
    return MongoDBDatabase(client, DB_NAME)


@flaky(max_runs=5, min_passes=4)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="test requires OpenAI for chat responses"
)
def test_toolkit_response(db):
    db_wrapper = MongoDBDatabase.from_connection_string(
        CONNECTION_STRING, database=DB_NAME
    )
    llm = ChatOpenAI(model="gpt-4o-mini", timeout=60)

    toolkit = MongoDBDatabaseToolkit(db=db_wrapper, llm=llm)

    system_message = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=5)

    test_query = "Which country's customers spent the most?"
    agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=system_message)
    agent.step_timeout = 60
    events = agent.stream(
        {"messages": [("user", test_query)]},
        stream_mode="values",
    )
    messages = []
    for event in events:
        messages.extend(event["messages"])
    assert "USA" in messages[-1].content, messages[-1].content
