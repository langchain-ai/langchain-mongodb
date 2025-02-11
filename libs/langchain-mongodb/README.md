from libs.community.tests.unit_tests.chains.test_pebblo_retrieval import retriever

# langchain-mongodb

# Installation
```
pip install -U langchain-mongodb
```

# Usage
- [Integrate Atlas Vector Search with LangChain](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/#get-started-with-the-langchain-integration) for a walkthrough on using your first LangChain implementation with MongoDB Atlas.

## Using MongoDBAtlasVectorSearch
```python
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

# Pull MongoDB Atlas URI from environment variables
MONGODB_ATLAS_CONNECTION_STRING = os.environ["MONGODB_CONNECTION_STRING"]
DB_NAME = "langchain_db"
COLLECTION_NAME = "test"
VECTOR_SEARCH_INDEX_NAME = "index_name"

MODEL_NAME = "text-embedding-3-large"
OPENAI_API_KEY =  os.environ["OPENAI_API_KEY"]


vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_ATLAS_CONNECTION_STRING,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=OpenAIEmbeddings(model=MODEL_NAME),
    index_name=VECTOR_SEARCH_INDEX_NAME,
)

retrieved_docs = vectorstore.similarity_search(
    "How do I deploy MongoDBAtlasVectorSearch in our production environment?")
```
