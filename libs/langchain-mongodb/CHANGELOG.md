# Changelog

---

## Changes in version 0.6.1 (2025/04/16)

- Fixed a syntax error in a docstring.
- Fixed some incorrect typings.
- Added detail to README
- Added MongoDBDocStore to README.

## Changes in version 0.6 (2025/03/26)

- Added Natural language to MQL Database tool.
- Added `MongoDBAtlasSelfQueryRetriever`.
- Added logic for vector stores to optionally create vector search indexes.
- Added `close()` methods to classes to ensure proper cleanup of resources.
- Changed the default `batch_size` to 100 to align with resource constraints on
  AI model APIs.

## Changes in version 0.5 (2025/02/25)

- Added GraphRAG support via `MongoDBGraphStore`.

## Changes in version 0.4 (2025/01/09)

- Added support for `MongoDBRecordManager`.
- Added support for `MongoDBLoader`.
- Added support for `numpy 2.0`.
- Added zizmor GitHub Actions security scanning.
- Added local LLM support for testing.

## Changes in version 0.3 (2024/12/13)

- Added support for `MongoDBAtlasParentDocumentRetriever`.
- Migrated to https://github.com/langchain-ai/langchain-mongodb.

## Changes in version 0.2 (2024/09/13)

- Added support for `MongoDBAtlasFullTextSearchRetriever` and `MongoDBAtlasHybridSearchRetriever`.

## Changes in version 0.1 (2024/02/29)

- Initial release, added support for `MongoDBAtlasVectorSearch`.
