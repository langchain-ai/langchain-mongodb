# Changelog

## 0.1.0

- Initial release of `langchain-mongodb-deepagents-vfs`.
- Implements DeepAgents' `BackendProtocol`, routing `grep`, `glob`, and `ls`
  through MongoDB Atlas (vector search + full-text search + hybrid
  `$rankFusion`) while forwarding all other file operations (`read`, `write`,
  `edit`, `upload_files`, `download_files`) directly to S3.
