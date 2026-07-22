"""Distribution and metadata tests — verifies the package is PyPI-ready."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent


@pytest.mark.unit
class TestPackageMetadata:
    def test_package_importable(self):
        import langchain_mongodb_deepagents_vfs

        assert langchain_mongodb_deepagents_vfs.__version__ == "0.1.0"

    def test_public_exports(self):
        from langchain_mongodb_deepagents_vfs import (
            AdapterError,
            ErrorCode,
            MongoFilesystemBackend,
        )

        assert MongoFilesystemBackend is not None
        assert AdapterError is not None
        assert ErrorCode is not None

    def test_version_string_format(self):
        import langchain_mongodb_deepagents_vfs

        parts = langchain_mongodb_deepagents_vfs.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_all_public_classes_have_docstrings(self):
        import inspect

        import langchain_mongodb_deepagents_vfs.backend as backend_mod
        import langchain_mongodb_deepagents_vfs.chunker as chunker_mod
        import langchain_mongodb_deepagents_vfs.embedder as embedder_mod
        import langchain_mongodb_deepagents_vfs.search as search_mod
        import langchain_mongodb_deepagents_vfs.sync as sync_mod

        for mod in [backend_mod, chunker_mod, embedder_mod, search_mod, sync_mod]:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if not name.startswith("_") and obj.__module__ == mod.__name__:
                    assert obj.__doc__, f"{mod.__name__}.{name} has no docstring"


@pytest.mark.unit
class TestBuildArtifacts:
    """Run build + twine check (requires build and twine installed)."""

    def test_build_wheel(self, tmp_path):
        """Build the package and verify the wheel is produced."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(tmp_path),
                str(ROOT),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"build failed:\n{result.stderr}"
        wheels = list(tmp_path.glob("*.whl"))
        assert len(wheels) == 1, f"expected 1 wheel, got {wheels}"

    def test_twine_check(self, tmp_path):
        """Verify wheel passes twine metadata validation."""
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(tmp_path),
                str(ROOT),
            ],
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            [sys.executable, "-m", "twine", "check"]
            + [str(p) for p in tmp_path.glob("*.whl")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"twine check failed:\n{result.stdout}\n{result.stderr}"
        )
