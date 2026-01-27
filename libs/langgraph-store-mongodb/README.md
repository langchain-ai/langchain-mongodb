# langraph-store-mongodb

LangGraph long-term memory using MongoDB.

## Installation

```bash
pip install -U langgraph-store-mongodb
```

## Usage

For more detailed usage examples and documentation, please refer to the [MongoDB LangGraph documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/).

```python
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from langchain_voyageai import VoyageAIEmbeddings

# Vector search index configuration with client-side embedding
index_config = create_vector_index_config(
    embed = VoyageAIEmbeddings(),
    dims = <dimensions>,
    fields = ["<field-name>"],
    filters = ["<filter-field-name>", ...] # Optional
)

# Store memories in MongoDB collection
with MongoDBStore.from_conn_string(
     conn_string=MONGODB_URI,
     db_name="<database-name>",
     collection_name="<collection-name>",
     index_config=index_config
 ) as store:
     store.put(
         namespace=("user", "memories"),
         key=f"memory_{hash(content)}",
         value={"content": content}
     )

# Retrieve memories from MongoDB collection
with MongoDBStore.from_conn_string(
    conn_string=MONGODB_URI,
    db_name="<database-name>",
    collection_name="<collection-name>",
    index_config=index_config
) as store:
     results = store.search(
         ("user", "memories"),
         query="<query-text>",
         limit=3
     )
     for result in results:
         print(result.value)

# To delete memories, use store.delete(namespace, key)
# To batch operations, use store.batch(ops)
```
