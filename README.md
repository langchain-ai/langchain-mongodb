# 🦜️🔗 LangChain MongoDB

This is a Monorepo containing partner packages of MongoDB and LangChainAI.
It includes integrations between MongoDB, Atlas, LangChain, and LangGraph.

It contains the following packages.

- `langchain-mongodb` ([PyPI](https://pypi.org/project/langchain-mongodb/))
- `langgraph-checkpoint-mongodb` ([PyPI](https://pypi.org/project/langgraph-checkpoint-mongodb/))
- `langgraph-store-mongodb` ([PyPI](https://pypi.org/project/langgraph-store-mongodb/))

**Note**: This repository replaces all MongoDB integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Features

### LangChain

#### Components

- [MongoDBAtlasFullTextSearchRetriever](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#full-text-retriever)
- [MongoDBAtlasHybridSearchRetriever](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#hybrid-search-retriever)
- [MongoDBAtlasSemanticCache](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#semantic-cache)
- [MongoDBAtlasVectorSearch](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#vector-store)
- [MongoDBCache](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#mongodb-cache)
- [MongoDBChatMessageHistory](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/#chat-history)

#### API Reference

- [MongoDBAtlasParentDocumentRetriever](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/retrievers/langchain_mongodb.retrievers.parent_document.MongoDBAtlasParentDocumentRetriever.html#langchain_mongodb.retrievers.parent_document.MongoDBAtlasParentDocumentRetriever)
- [MongoDBAtlasSelfQueryRetriever](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/retrievers/langchain_mongodb.retrievers.self_querying.MongoDBAtlasSelfQueryRetriever.html).
- [MongoDBDatabaseToolkit](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/agent_toolkit/langchain_mongodb.agent_toolkit.toolkit.MongoDBDatabaseToolkit.html)
- [MongoDBDatabase](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/agent_toolkit/langchain_mongodb.agent_toolkit.database.MongoDBDatabase.html#langchain_mongodb.agent_toolkit.database.MongoDBDatabase)
- [MongoDBDocStore](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/docstores/langchain_mongodb.docstores.MongoDBDocStore.html#langchain_mongodb.docstores.MongoDBDocStore)
- [MongoDBGraphStore](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/graphrag/langchain_mongodb.graphrag.graph.MongoDBGraphStore.html)
- [MongoDBLoader](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/loaders/langchain_mongodb.loaders.MongoDBLoader.html#langchain_mongodb.loaders.MongoDBLoader)
- [MongoDBRecordManager](https://langchain-mongodb.readthedocs.io/en/latest/langchain_mongodb/indexes/langchain_mongodb.indexes.MongoDBRecordManager.html#langchain_mongodb.indexes.MongoDBRecordManager)

### LangGraph

- Checkpointing (BaseCheckpointSaver)
    - [MongoDBSaver](https://langchain-mongodb.readthedocs.io/en/latest/langgraph_checkpoint_mongodb/saver/langgraph.checkpoint.mongodb.saver.MongoDBSaver.html#mongodbsaver)

- Long-term memory (BaseStore)
   - [MongoDBStore](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/#mongodb-langgraph-store--long-term-memory-)

## Installation

You can install `langchain-mongodb`, `langgraph-checkpoint-mongodb` and `langgraph-store-mongodb` from PyPI.

```bash
pip install langchain-mongodb langgraph-checkpoint-mongodb langgraph-store-mongodb
```



## Usage

See [langchain-mongodb usage](libs/langchain-mongodb/README.md#usage), [langgraph-checkpoint-mongodb usage](libs/langgraph-checkpoint-mongodb/README.md#usage) and [langgraph-store-mongodb usage](libs/langgraph-store-mongodb/README.md#usage).

For more detailed usage examples and documentation, please refer to the [MongoDB LangChain documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/) and the [MongoDB LangGraph documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/).

API docs can be found on [ReadTheDocs](https://langchain-mongodb.readthedocs.io/en/latest/index.html).

## Contributing

See the [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
