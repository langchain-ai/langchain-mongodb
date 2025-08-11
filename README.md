# 🦜️🔗 LangChain MongoDB

This is a Monorepo containing partner packages of MongoDB and LangChainAI.
It includes integrations between MongoDB, Atlas, LangChain, and LangGraph.

It contains the following packages.

- `langchain-mongodb` ([PyPI](https://pypi.org/project/langchain-mongodb/))
- `langgraph-checkpoint-mongodb` ([PyPI](https://pypi.org/project/langgraph-checkpoint-mongodb/))

**Note**: This repository replaces all MongoDB integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Features

### LangChain

#### Components

- [MongoDBAtlasFullTextSearchRetriever](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#full-text-search-retriever)
- [MongoDBAtlasHybridSearchRetriever](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#hybrid-search-retriever)
- [MongoDBAtlasSemanticCache](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#mongodbatlassemanticcache)
- [MongoDBAtlasVectorSearch](https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas/)
- [MongoDBCache](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#mongodbcache)
- [MongoDBChatMessageHistory](https://python.langchain.com/docs/integrations/memory/mongodb_chat_message_history/)

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
    - [AsyncMongoDBSaver](https://langchain-mongodb.readthedocs.io/en/latest/langgraph_checkpoint_mongodb/aio/langgraph.checkpoint.mongodb.aio.AsyncMongoDBSaver.html#asyncmongodbsaver)

- Long-term memory (BaseStore)
   - [MongoDBStore](https://langchain-mongodb.readthedocs.io/en/latest/langgraph_store_mongodb/base/langgraph.store.mongodb.base.MongoDBStore.html#langgraph.store.mongodb.base.MongoDBStore)

## Installation

You can install the `langchain-mongodb` package from PyPI.

```bash
pip install langchain-mongodb
```

You can install the `langgraph-checkpoint-mongodb` package from PyPI as well:

```bash
pip install langgraph-checkpoint-mongodb
```

## Usage

See [langchain-mongodb usage](libs/langchain-mongodb/README.md#usage) and [langgraph-checkpoint-mongodb usage](libs/langgraph-checkpoint-mongodb/README.md#usage).

For more detailed usage examples and documentation, please refer to the [LangChain documentation](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/).

API docs can be found on [ReadTheDocs](https://langchain-mongodb.readthedocs.io/en/latest/index.html).

## Contributing

See the [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
