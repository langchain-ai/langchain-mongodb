# Changelog

## 0.1.1  (2026/10/30)

- `s3_prefix` now gets a trailing `/` if it is missing, so it matches whole
  path segments only: `s3_prefix="docs"` covers `docs/...` but no longer
  `docs2/...`. This applies to read, write, edit, sync, and watch.
- Raised the `pymongo` floor to `>=4.18.2`.
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
