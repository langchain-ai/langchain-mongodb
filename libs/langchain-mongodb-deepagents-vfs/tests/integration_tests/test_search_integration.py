"""Integration tests for SearchRouter on mongomock with pre-seeded data."""

from __future__ import annotations

import pytest

from langchain_mongodb_deepagents_vfs.search import SearchRouter


def seed_collection(col, entries):
    """Insert chunk documents into mongomock collection."""
    col.insert_many(entries)


def make_doc(source_path, chunk_index=0, filename=None, content="text", line_start=0):
    fname = filename or source_path.split("/")[-1]
    return {
        "source_path": source_path,
        "chunk_index": chunk_index,
        "filename": fname,
        "content": content,
        "line_start": line_start,
        "embedding": [0.0] * 1024,
        "etag": "abc123",
    }


@pytest.mark.integration
class TestSearchRouterIntegration:
    @pytest.fixture(autouse=True)
    def seed(self, mongo_collection):
        seed_collection(
            mongo_collection,
            [
                make_doc(
                    "docs/api/auth.txt",
                    chunk_index=0,
                    content="authentication token flow",
                    line_start=0,
                ),
                make_doc(
                    "docs/api/auth.txt",
                    chunk_index=1,
                    content="refresh token expiry",
                    line_start=10,
                ),
                make_doc(
                    "docs/guide/install.md",
                    chunk_index=0,
                    content="installation prerequisites",
                    line_start=0,
                ),
                make_doc(
                    "docs/report.pdf",
                    chunk_index=0,
                    content="quarterly financial report",
                    line_start=0,
                    filename="report.pdf",
                ),
                make_doc(
                    "images/logo.png",
                    chunk_index=0,
                    content="binary image data",
                    line_start=0,
                    filename="logo.png",
                ),
            ],
        )

    def test_ls_root(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.ls("")
        names = {e["path"].rstrip("/").rsplit("/", 1)[-1] for e in result.entries}
        assert "docs" in names
        assert "images" in names

    def test_ls_docs_directory(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.ls("docs")
        names = {e["path"].rstrip("/").rsplit("/", 1)[-1] for e in result.entries}
        # Should see api, guide, report.pdf
        assert len(names) >= 1

    def test_glob_pdf_files(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.glob("**/*.pdf")
        assert len(result.matches) == 1
        assert result.matches[0]["path"] == "docs/report.pdf"

    def test_glob_non_recursive_by_default(self, mongo_collection, mock_embedder):
        """Every seeded key is nested, so a bare "*.pdf" from root matches none."""
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        assert router.glob("*.pdf").matches == []
        assert (
            router.glob("*.pdf", path="docs/").matches[0]["path"] == "docs/report.pdf"
        )

    def test_glob_md_files(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.glob("**/*.md")
        assert any("install.md" in m["path"] for m in result.matches)

    def test_glob_with_path_restriction(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.glob("*.png", path="images/")
        assert all(m["path"].startswith("images/") for m in result.matches)

    def test_grep_returns_match(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("authentication")
        assert len(result.matches) > 0
        assert any("auth.txt" in m["path"] for m in result.matches)

    def test_grep_with_path_restriction(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("token", path="docs/api/")
        assert all("docs/api/" in m["path"] for m in result.matches)

    def test_grep_with_glob_restriction(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("report", glob="*.pdf")
        for match in result.matches:
            assert match["path"].endswith(".pdf")

    def test_grep_no_match_returns_empty(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("zzznonexistenttermzzz")
        assert result.matches == []
        assert result.error is None

    def test_grep_match_has_line_start(self, mongo_collection, mock_embedder):
        router = SearchRouter(mongo_collection, mock_embedder, atlas_available=False)
        result = router.grep("refresh token")
        token_match = next(
            (m for m in result.matches if "auth.txt" in m["path"] and m["line"] == 10),
            None,
        )
        assert token_match is not None
