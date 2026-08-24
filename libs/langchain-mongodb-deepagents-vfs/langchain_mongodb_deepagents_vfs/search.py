"""SearchRouter: MongoDB-powered ls, glob, grep.

All three operations return DeepAgents-compatible DTO shapes.
grep uses $rankFusion (hybrid search) when Atlas indexes are available,
degrading gracefully to regex + no-vector search on non-Atlas clusters.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Iterable
from typing import Any

from pymongo.collection import Collection
from pymongo_search_utils import text_search_stage, vector_search_stage
from wcmatch.glob import BRACE, GLOBSTAR, globmatch

from langchain_mongodb_deepagents_vfs.dtypes import (
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
)
from langchain_mongodb_deepagents_vfs.embedder import Embedder
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

logger = logging.getLogger(__name__)

_DEFAULT_GREP_LIMIT = 20
_DEFAULT_GLOB_LIMIT = 200
_DEFAULT_LS_LIMIT = 1000

# Server-side cap on the non-Atlas regex grep so a pathological pattern can't
# saturate MongoDB CPU. Paired with re.escape() (literal match) at the callsite.
_GREP_MAX_TIME_MS = 5000


def _file_infos(paths: Iterable[str]) -> list[FileInfo]:
    """Wrap source paths in the protocol's FileInfo shape.

    ``size``/``modified_at`` are omitted rather than faked: chunks are stored,
    not the objects themselves, so neither is known without an extra S3 call.
    """
    return [FileInfo(path=p, is_dir=False) for p in paths]


class SearchRouter:
    """Routes ls / glob / grep to the appropriate MongoDB query mechanism.

    Args:
        collection: The MongoDB collection storing chunk documents.
        embedder: Embedder instance for generating query vectors.
        atlas_available: Whether Atlas Search / Vector Search is available.
            Detected automatically if None.
        grep_limit: Max results returned by grep.
        glob_limit: Max results returned by glob.
    """

    def __init__(
        self,
        collection: Collection,  # type: ignore[type-arg]
        embedder: Embedder,
        atlas_available: bool | None = None,
        grep_limit: int = _DEFAULT_GREP_LIMIT,
        glob_limit: int = _DEFAULT_GLOB_LIMIT,
    ) -> None:
        self._col = collection
        self._embedder = embedder
        self._grep_limit = grep_limit
        self._glob_limit = glob_limit
        self._atlas = atlas_available  # resolved on first use if None

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List the immediate children of *path*.

        Uses a native MongoDB aggregation: filter by source_path prefix, then
        extract the next path segment. Returns directories (shared prefixes)
        and files (leaf keys).

        Args:
            path: Virtual directory path (e.g. "docs/").

        Returns:
            LsResult with ``FileInfo`` entries sorted alphabetically.
            Following the protocol convention, directory entries carry a
            trailing ``/`` in ``path`` and ``is_dir=True``.

        Raises:
            AdapterError(E5003): Query failure.
        """
        prefix = path.rstrip("/")
        if prefix:
            prefix = prefix + "/"
        escaped = re.escape(prefix)
        match_stage: dict[str, Any] = (
            {"source_path": {"$regex": f"^{escaped}"}} if prefix else {}
        )
        pipeline: list[dict[str, Any]] = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "$arrayElemAt": [
                            {
                                "$split": [
                                    {"$substr": ["$source_path", len(prefix), -1]},
                                    "/",
                                ]
                            },
                            0,
                        ]
                    },
                    "full_paths": {"$addToSet": "$source_path"},
                }
            },
            {"$sort": {"_id": 1}},
            {"$limit": _DEFAULT_LS_LIMIT},
        ]
        try:
            docs = list(self._col.aggregate(pipeline))
        except Exception as exc:
            raise AdapterError(ErrorCode.E5003_LS_FAILED, str(exc)) from exc

        entries: list[FileInfo] = []
        for doc in docs:
            segment: str = doc["_id"] or ""
            if not segment:
                continue
            # If any stored path has more segments after this one → it's a directory
            full_paths: list[str] = doc["full_paths"]
            is_dir = any(
                p[len(prefix) :].count("/") > 0
                or not p[len(prefix) :].endswith(segment)
                for p in full_paths
            )
            entry_path = prefix + segment + ("/" if is_dir else "")
            # size omitted, not zeroed — see _file_infos: we store chunks, not
            # the objects, so byte size isn't known without an extra S3 call.
            entries.append(FileInfo(path=entry_path, is_dir=is_dir))

        return LsResult(entries=entries)

    # ------------------------------------------------------------------
    # glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: str = "") -> GlobResult:
        """Find files matching *pattern* under *path*.

        Follows the standard glob semantics the ``BackendProtocol`` documents,
        using the same matcher (``wcmatch`` with BRACE + GLOBSTAR) as
        deepagents' own backends, so behaviour is identical to theirs:

        - ``*`` matches within one path segment and does **not** cross "/",
          so "*.pdf" matches "report.pdf" but not "docs/report.pdf".
        - ``**`` matches recursively — use "**/*.pdf" to search subdirectories.
        - ``?`` matches a single character, ``[abc]`` a character set, and
          ``{a,b}`` alternates.

        The pattern is matched against each key *relative to* ``path``, so
        ``glob("*.pdf", path="docs/")`` matches "docs/report.pdf".

        There is a single implementation for Atlas and non-Atlas clusters:
        Atlas Search's ``wildcard`` operator cannot express ``**``, ``[abc]``
        or ``{a,b}``, so matching all patterns identically means matching in
        Python. Only one document per distinct key is fetched (``$group``), so
        the cost is O(files), not O(chunks).

        Args:
            pattern: Glob pattern applied to the key relative to *path*
                (e.g. "*.pdf", "**/*.md").
            path: Restrict search to this path prefix.

        Returns:
            GlobResult with a ``FileInfo`` per matching source_path
            (deduplicated).

        Raises:
            AdapterError(E5002): Query failure.
        """
        try:
            return self._glob(pattern, path)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(ErrorCode.E5002_GLOB_FAILED, str(exc)) from exc

    def _glob(self, pattern: str, path: str) -> GlobResult:
        prefix = path.rstrip("/")
        if prefix:
            prefix = prefix + "/"
        match_stage: dict[str, Any] = (
            {"source_path": {"$regex": f"^{re.escape(prefix)}"}} if prefix else {}
        )
        pipeline: list[dict[str, Any]] = [
            {"$match": match_stage},
            {"$group": {"_id": "$source_path"}},
            {"$sort": {"_id": 1}},
        ]
        # Leading "/" is not meaningful for S3 keys, which are always relative.
        effective = pattern.lstrip("/")
        results: list[str] = []
        for doc in self._col.aggregate(pipeline):
            source_path = doc["_id"]
            if not source_path:
                continue
            if globmatch(source_path[len(prefix) :], effective, flags=BRACE | GLOBSTAR):
                results.append(source_path)
                if len(results) >= self._glob_limit:
                    break
        return GlobResult(matches=_file_infos(results))

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(self, pattern: str, path: str = "", glob: str = "") -> GrepResult:
        """Search chunk content for *pattern* using hybrid search.

        Uses $rankFusion combining Atlas Full-Text Search and Vector Search
        when available; falls back to regex on non-Atlas clusters.

        Args:
            pattern: Search query (natural language or keywords).
            path: Restrict to source_paths starting with this prefix.
            glob: Further restrict to filenames matching this glob.

        Returns:
            GrepResult with matches deduplicated by (source_path, line_start).

        Raises:
            AdapterError(E5001): Search failure.
        """
        try:
            if self._is_atlas_available():
                return self._grep_hybrid(pattern, path, glob)
            return self._grep_regex(pattern, path, glob)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(ErrorCode.E5001_GREP_FAILED, str(exc)) from exc

    def _grep_hybrid(self, pattern: str, path: str, glob: str) -> GrepResult:
        # Generate query embedding
        from langchain_mongodb_deepagents_vfs.dtypes import Chunk

        dummy_chunk = Chunk(source_path="", chunk_index=0, content=pattern)
        try:
            query_vector = self._embedder.embed_batch([dummy_chunk])[0]
        except AdapterError as exc:
            logger.warning(
                "Vector embedding for query failed, falling back to full-text only: %s",
                exc,
            )
            query_vector = None

        path_filter: list[dict[str, Any]] = []
        if path:
            path_filter.append({"source_path": {"$regex": f"^{re.escape(path)}"}})
        if glob:
            path_filter.append({"filename": {"$regex": fnmatch.translate(glob)}})

        pre_filter: dict[str, Any] = {"$and": path_filter} if path_filter else {}

        if query_vector is not None:
            # Both branches feed the same combination step, so both need a
            # comparable bound: $vectorSearch already caps itself via its own
            # limit/numCandidates fields, but $search has no such built-in cap
            # — without an explicit $limit here, the fulltext branch could
            # rank an unbounded number of candidates before the single $limit
            # trailing the whole $rankFusion ever gets a chance to trim it.
            top_k = self._grep_limit * 2
            vs_stage = vector_search_stage(
                query_vector,
                "embedding",
                "vector_search_embedding",
                top_k=top_k,
                # $vectorSearch filter only supports comparison operators ($eq, $in, etc.),
                # not $regex — apply path/glob filtering as a $match stage after the ANN search
                filter=None,
                oversampling_factor=5,  # numCandidates = top_k * 5 = grep_limit * 10
            )
            pipeline: list[dict[str, Any]] = [
                {
                    "$rankFusion": {
                        "input": {
                            # pipelines: named object, each value is an array of stages
                            "pipelines": {
                                "fulltext": [
                                    {
                                        "$search": {
                                            "index": "fulltext_search_content_filename",
                                            "text": {
                                                "query": pattern,
                                                "path": "content",
                                            },
                                        }
                                    },
                                    {"$limit": top_k},
                                    *([{"$match": pre_filter}] if pre_filter else []),
                                ],
                                "vector": [
                                    vs_stage,
                                    *([{"$match": pre_filter}] if pre_filter else []),
                                ],
                            }
                        },
                        "combination": {"weights": {"fulltext": 0.5, "vector": 0.5}},
                    }
                },
                {"$limit": self._grep_limit},
                {
                    "$project": {
                        "source_path": 1,
                        "line_start": 1,
                        "content": 1,
                        "_id": 0,
                    }
                },
            ]
        else:
            # Full-text only fallback (no vector)
            pipeline = [
                *text_search_stage(
                    pattern,
                    "content",
                    "fulltext_search_content_filename",
                    limit=self._grep_limit,
                    filter=pre_filter if pre_filter else None,
                ),
                {
                    "$project": {
                        "source_path": 1,
                        "line_start": 1,
                        "content": 1,
                        "_id": 0,
                    }
                },
            ]

        docs = list(self._col.aggregate(pipeline))
        return self._dedupe_to_grep_result(docs)

    def _grep_regex(self, pattern: str, path: str, glob: str) -> GrepResult:
        # Escape the user pattern to a literal: the non-Atlas fallback is a
        # substring search, not a regex engine exposed to callers. This defuses
        # catastrophic-backtracking inputs like "(a+)+$". max_time_ms bounds the
        # scan server-side as a second line of defence.
        query: dict[str, Any] = {
            "content": {"$regex": re.escape(pattern), "$options": "i"}
        }
        if path:
            query["source_path"] = {"$regex": f"^{re.escape(path)}"}
        if glob:
            query["filename"] = {"$regex": fnmatch.translate(glob)}
        cursor = (
            self._col.find(
                query,
                {"source_path": 1, "line_start": 1, "content": 1, "_id": 0},
            )
            .limit(self._grep_limit)
            .max_time_ms(_GREP_MAX_TIME_MS)
        )
        return self._dedupe_to_grep_result(list(cursor))

    @staticmethod
    def _dedupe_to_grep_result(docs: list[dict[str, Any]]) -> GrepResult:
        """Project ranked docs onto the protocol's GrepMatch shape.

        Relevance scores drive ordering inside the pipeline but are not part of
        ``GrepMatch``, so they are not emitted — rank is carried by list order.
        """
        seen: set[tuple[str, int]] = set()
        matches: list[GrepMatch] = []
        for doc in docs:
            key = (doc["source_path"], doc["line_start"])
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                GrepMatch(
                    path=doc["source_path"],
                    line=doc["line_start"],
                    text=doc["content"],
                )
            )
        return GrepResult(matches=matches)

    # ------------------------------------------------------------------
    # Atlas availability check
    # ------------------------------------------------------------------

    def _is_atlas_available(self) -> bool:
        if self._atlas is not None:
            return self._atlas
        try:
            self._col.list_search_indexes()
            self._atlas = True
        except (AttributeError, Exception):
            self._atlas = False
        return self._atlas
