"""Unit tests for SearchRouter (uses mongomock, no Atlas)."""

from __future__ import annotations

import pytest

from langchain_mongodb_deepagents_vfs.dtypes import GlobResult, GrepResult, LsResult
from langchain_mongodb_deepagents_vfs.search import SearchRouter


def _seed(collection, docs):
    """Insert minimal chunk docs into mongomock collection."""
    collection.insert_many(docs)


def _make_doc(source_path, chunk_index=0, filename=None, content="text", line_start=0):
    fname = filename or source_path.rsplit("/", 1)[-1]
    return {
        "source_path": source_path,
        "chunk_index": chunk_index,
        "filename": fname,
        "content": content,
        "line_start": line_start,
        "embedding": [0.0] * 1024,
    }


@pytest.mark.unit
class TestSearchRouterLs:
    def test_ls_empty_path(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/a.txt"),
                _make_doc("docs/b.txt"),
                _make_doc("images/c.png"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.ls("")
        # Should see top-level segments: docs, images
        names = {e["path"].rstrip("/").rsplit("/", 1)[-1] for e in result.entries}
        assert "docs" in names
        assert "images" in names

    def test_ls_specific_directory(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/api/ref.txt"),
                _make_doc("docs/guide.txt"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.ls("docs")
        names = {e["path"].rstrip("/").rsplit("/", 1)[-1] for e in result.entries}
        assert "guide.txt" in names or "api" in names

    def test_ls_returns_ls_result(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.ls("anything")
        assert isinstance(result, LsResult)


@pytest.mark.unit
class TestSearchRouterGlob:
    def test_glob_by_extension(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/report.pdf", filename="report.pdf"),
                _make_doc("docs/guide.txt", filename="guide.txt"),
                _make_doc("docs/summary.pdf", filename="summary.pdf"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        # "**/" is required to reach into docs/ — "*" does not cross "/".
        result = router.glob("**/*.pdf")
        assert isinstance(result, GlobResult)
        for m in result.matches:
            assert m["path"].endswith(".pdf")
        assert len(result.matches) == 2

    def test_glob_no_match(self, mongo_collection, mock_embedder):
        _seed(mongo_collection, [_make_doc("docs/guide.txt", filename="guide.txt")])
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.glob("**/*.pdf")
        assert result.matches == []

    def test_glob_star_does_not_cross_separator(self, mongo_collection, mock_embedder):
        """Standard glob semantics: "*" matches within one segment only."""
        _seed(
            mongo_collection,
            [
                _make_doc("root.pdf", filename="root.pdf"),
                _make_doc("docs/nested.pdf", filename="nested.pdf"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        assert [m["path"] for m in router.glob("*.pdf").matches] == ["root.pdf"]
        assert [m["path"] for m in router.glob("**/*.pdf").matches] == [
            "docs/nested.pdf",
            "root.pdf",
        ]

    def test_glob_brace_and_charset(self, mongo_collection, mock_embedder):
        """BRACE alternation and [abc] sets, which fnmatch/Atlas wildcard lack."""
        _seed(
            mongo_collection,
            [
                _make_doc("a.py", filename="a.py"),
                _make_doc("b.md", filename="b.md"),
                _make_doc("c.txt", filename="c.txt"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        assert {m["path"] for m in router.glob("*.{py,md}").matches} == {"a.py", "b.md"}
        assert [m["path"] for m in router.glob("[ab].py").matches] == ["a.py"]

    def test_glob_pattern_is_relative_to_path(self, mongo_collection, mock_embedder):
        """The pattern matches the key relative to *path*, not the full key."""
        _seed(
            mongo_collection,
            [
                _make_doc("docs/api/auth.md", filename="auth.md"),
                _make_doc("docs/install.md", filename="install.md"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        # Relative to "docs/", install.md is a direct child; auth.md is not.
        assert [m["path"] for m in router.glob("*.md", path="docs/").matches] == [
            "docs/install.md"
        ]
        assert [m["path"] for m in router.glob("api/*.md", path="docs/").matches] == [
            "docs/api/auth.md"
        ]

    def test_glob_with_path_prefix(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/a.pdf", filename="a.pdf"),
                _make_doc("images/b.pdf", filename="b.pdf"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.glob("*.pdf", path="docs/")
        assert all(m["path"].startswith("docs/") for m in result.matches)


@pytest.mark.unit
class TestSearchRouterGrep:
    def test_grep_regex_fallback(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/api.txt", content="authentication flow diagram"),
                _make_doc("docs/setup.txt", content="installation instructions"),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("authentication")
        assert isinstance(result, GrepResult)
        assert any("api.txt" in m["path"] for m in result.matches)

    def test_grep_regex_pattern_escaped_to_literal(
        self, mongo_collection, mock_embedder
    ):
        # "a+" as a regex matches "aaa"; escaped to a literal it must not, proving
        # user input can't reach the $regex engine as an unbounded pattern.
        _seed(mongo_collection, [_make_doc("docs/a.txt", content="aaa")])
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        assert router.grep("a+").matches == []
        assert any("a.txt" in m["path"] for m in router.grep("aaa").matches)

    def test_grep_deduplication(self, mongo_collection, mock_embedder):
        # Two chunks from the same file with same line_start → should be deduped
        _seed(
            mongo_collection,
            [
                _make_doc(
                    "docs/a.txt", chunk_index=0, content="auth token", line_start=5
                ),
                _make_doc(
                    "docs/a.txt", chunk_index=1, content="auth session", line_start=5
                ),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("auth")
        paths_lines = [(m["path"], m["line"]) for m in result.matches]
        assert len(paths_lines) == len(set(paths_lines))

    def test_grep_returns_line_start(self, mongo_collection, mock_embedder):
        _seed(
            mongo_collection,
            [
                _make_doc("docs/a.txt", content="vector search", line_start=42),
            ],
        )
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("vector")
        assert result.matches[0]["line"] == 42

    def test_grep_match_keys_match_protocol_exactly(
        self, mongo_collection, mock_embedder
    ):
        """No extra keys (e.g. 'score') may leak into the protocol's GrepMatch."""
        _seed(mongo_collection, [_make_doc("docs/a.txt", content="vector search")])
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("vector")
        assert result.matches
        for m in result.matches:
            assert set(m) == {"path", "line", "text"}
