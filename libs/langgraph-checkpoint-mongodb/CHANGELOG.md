# Changelog

---

## Changes in version 0.2.2 (2025/11/12)

- Bumps minimum version of langgraph-checkpoint to 3.0 to address the Remode Code Execution CVE in JsonPlusSerializer's "json" mode.
- Fixed teardown step in release.yml GitHub workflow.

## Changes in version 0.2.1 (2025/09/25)

- Fixes bug when graph interrupted leading to DuplicateKeyError when TTL is set.

## Changes in version 0.2.0 (2025/08/20)

- Implements async methods of MongoDBSaver.
- Deprecates ASyncMongoDBSaver, to be removed in 0.3.0.
- Add additional client metadata to ``collection`` objects consumed by ``langgraph-checkpoint-mongodb``.
- Fixes persistence of user metadata for ``langgraph`` 0.5+.

## Changes in version 0.1.4 (2025/06/13)

- Add TTL (time-to-live) indexes for automatic deletion of old checkpoints and writes
- Add delete_thread and adelete_thread methods for manual delete of checkpoints and writes.

## Changes in version 0.1.3 (2025/04/01)

- Add compatibility with `pymongo.AsyncMongoClient`.

## Changes in version 0.1.2 (2025/03/26)

- Add compatibility with `langgraph-checkpoint` 2.0.23.

## Changes in version 0.1.1 (2025/02/26)

- Remove dependency on `langgraph`.

## Changes in version 0.1 (2024/12/13)

- Initial release, added support for `MongoDBSaver`.
