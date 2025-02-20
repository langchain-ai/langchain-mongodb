# flake8: noqa


MONGODB_AGENT_SYSTEM_PROMPT = """You are an agent designed to interact with a MongoDB database.
Given an input question, create a syntactically correct MongoDB query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant field to return the most interesting examples in the database.
Never query for all the fields from a specific collection, only ask for the relevant fields given the question.

You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any update, insert, or delete operations.

To start you should ALWAYS look at the collections in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant collections."""

MONGODB_PREFIX = """You are an agent designed to interact with a MongoDB database.
Given an input question, create a syntactically correct query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant field to return the most interesting examples in the collection.
Never query for all the fields from a specific collection, only ask for the relevant fields given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any update, insert, or delete operations.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""

MONGODB_SUFFIX = """Begin!

Question: {input}
Thought: I should look at the collections in the database to see what I can query.  Then I should query the schema of the most relevant collections.
{agent_scratchpad}"""

MONGODB_FUNCTIONS_SUFFIX = """I should look at the collections in the database to see what I can query.  Then I should query the schema of the most relevant collections."""


MONGODB_QUERY_CHECKER = """
{query}

Double check the MongoDB query above for common mistakes, including:
- Missing content in the aggegregation pipeline
- Improperly quoting identifiers
- Improperly quoting operators

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final MongoDB query only.

MongoDB Query: """


COMPASS_PROMPT = """
Database name: %s
Collection name: %s
Schema from a sample of documents from the collection:
```
%s
```
Sample documents from the collection: %s

Generate an aggregation that does the following:
Reduce prose to the minimum, your output will be parsed by a machine. You generate
    MongoDB aggregation pipelines. Provide only the aggregation pipeline
    contents in an array in shell syntax, wrapped with XML delimiters as
    follows:
<aggregation>[]</aggregation>
Additional instructions:
- Only pass the contents of the aggregation, no surrounding syntax.
- The current date is %s
"""
