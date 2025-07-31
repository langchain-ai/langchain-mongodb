# Changelog

---

## Changes in version 0.2.0 (TBD)

- Implements async methods of MongoDBSaver.
- Deprecates ASyncMongoDBSaver, to be removed in 0.3.0

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
