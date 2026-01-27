# langraph-store-mongodb

LangGraph long-term memory using MongoDB.

## Installation

```bash
pip install -U langgraph-store-mongodb
```

## Usage

For more detailed usage examples and documentation, please refer to the [MongoDB LangGraph documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/).

```python
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
# Connect to your MongoDB cluster
client = MongoClient("<connection-string>")
# Initialize the MongoDB checkpointer
checkpointer = MongoDBSaver(client)
# Instantiate the graph with the checkpointer
app = graph.compile(checkpointer=checkpointer)
```
