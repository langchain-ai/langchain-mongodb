# Changelog

---

## Changes in version 0.2.1 (Unreleased)

- Fixes duplicate-key collisions when two items share a namespace component and the same key (INTPYTHON-948). The unique index is now created on a denormalized `namespace_str` field (e.g. `"users/alice/preferences"`) instead of the `namespace` array, which was creating a multikey index in MongoDB. **Operational note:** this release is not safe for mixed-version deployments. Old clients that write documents without `namespace_str` will create documents that new clients cannot retrieve. Upgrade all writers and readers together; do not run old and new versions of the store concurrently against the same collection.

## Changes in version 0.2.0 (2026/01/15)

- Adds support for semantic search with Automated embedding in Vector search (for public preview on MongoDB Community).

## Changes in version 0.1.1 (2025/11/13)

- Bumps minimum version of langgraph-checkpoint to 3.0 to address the Remode Code Execution CVE in JsonPlusSerializer's "json" mode, described [here](https://osv.dev/vulnerability/GHSA-wwqv-p2pp-99h5).
- Only lists authorized collections when listing collections.

## Changes in version 0.1.0 (2025/08/20)

- Add additional client metadata to ``collection`` objects consumed by ``langgraph-store-mongodb``.
- Improve testing.

## Changes in version 0.0.1 (2025/05/09)

- Initial release, added support for `MongoDBStore`.
