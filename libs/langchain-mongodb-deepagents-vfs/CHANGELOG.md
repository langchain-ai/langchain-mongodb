# Changelog

## Unreleased

- `grep`, `glob`, and `ls` now ignore a leading `/` on `path`. Deep Agents'
  file tools pass absolute paths such as `/docs`, but indexed keys are
  relative, so these calls previously matched nothing and returned empty
  results even when the files existed.

## 0.1.0  (2026/08/24)

- Initial release of `langchain-mongodb-deepagents-vfs`.
- Implements DeepAgents' `BackendProtocol`, routing `grep`, `glob`, and `ls`
  through MongoDB Atlas (vector search + full-text search + hybrid
  `$rankFusion`) while forwarding all other file operations (`read`, `write`,
  `edit`, `upload_files`, `download_files`) directly to S3.
