# 🦜️🔗 LangChain MongoDB

This is a Monorepo containing partner packages of MongoDB and LangChainAI.
It includes integrations between MongoDB, Atlas, LangChain, and LangGraph.

It contains the following packages.

- `langchain-mongodb` ([PyPI](https://pypi.org/project/langchain-mongodb/))
- `langgraph-checkpoint-mongodb` ([PyPI](https://pypi.org/project/langgraph-checkpoint-mongodb/))

**Note**: This repository replaces all MongoDB integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

## Features

### LangChain

- Vector store
    -[MongoDBAtlasVectorSearch](https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas/)
- Message histories
    - [MongoDBChatMessageHistory](https://python.langchain.com/docs/integrations/memory/mongodb_chat_message_history/)
- Model caches
    - [MongoDBCache](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#mongodbcache)
    - [MongoDBAtlasSemanticCache](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#mongodbatlassemanticcache)
- Retrievers
    - [MongoDBAtlasHybridSearchRetriever](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#hybrid-search-retriever)
    - [MongoDBAtlasFullTextSearchRetriever](https://python.langchain.com/docs/integrations/providers/mongodb_atlas/#full-text-search-retriever)
    - `MongoDBAtlasParentDocumentRetriever` - new feature, docs coming soon!
- Indexing
    - `MongoDBRecordManager` - new feature, docs coming soon!

### LangGraph

- Checkpointing
    - `MongoDBSaver` - new feature, docs coming soon!

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

## Contributing

See the [Contributing Guide](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
