# Changelog

---

## Changes in version 0.3.1 (2026/01/22)
- Fixes issue #287 to migrate checkpoint data created with v<0.2.2 with a migration script: [migrate_checkpoints_to_typed_metadata.py](./scripts/migrate_checkpoints_to_typed_metadata.py).
- Fixes issue #299 to ensure TTL indexes are always created on initialization of MongoDBSaver if enabled.
- Addresses a separate concern in issue #299 to ensure that all timestamps are stored in UTC format.

## Changes in version 0.3.0 (2025/11/19)
- Allow custom serde objects to be passed to MongoDBSaver for serialization/deserialization.
- Remove the deprecated AsyncMongoDBSaver class, which has been replaced by MongoDBSaver's async methods.
- Update dependencies to require LangChain and LangGraph versions 1.0.0 and above.

## Changes in version 0.2.2 (2025/11/13)

- Bumps minimum version of langgraph-checkpoint to 3.0 to address the Remode Code Execution CVE in JsonPlusSerializer's "json" mode, described [here](https://osv.dev/vulnerability/GHSA-wwqv-p2pp-99h5).
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
