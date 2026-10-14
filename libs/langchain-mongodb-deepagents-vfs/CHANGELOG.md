# Changelog

## Unreleased

- `MongoFilesystemBackend.grep()` now reports an empty search scope. When
  `path` and/or `glob` select no indexed files, it returns
  `GrepResult(error="[E5005] ...", matches=None)` instead of
  `error=None, matches=[]`, so callers can tell "nothing was searched" apart
  from "searched, no match" (still `error=None, matches=[]`). Callers that read
  `result.matches` without checking `result.error` must handle `None`.
  See [#462](https://github.com/langchain-ai/langchain-mongodb/issues/462).

## 0.1.0  (2026/08/24)

- Initial release of `langchain-mongodb-deepagents-vfs`.
- Implements DeepAgents' `BackendProtocol`, routing `grep`, `glob`, and `ls`
  through MongoDB Atlas (vector search + full-text search + hybrid
  `$rankFusion`) while forwarding all other file operations (`read`, `write`,
  `edit`, `upload_files`, `download_files`) directly to S3.
