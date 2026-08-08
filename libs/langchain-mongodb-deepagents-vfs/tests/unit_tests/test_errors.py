"""Unit tests for the error layer."""

from __future__ import annotations

import pytest

from langchain_mongodb_deepagents_vfs.dtypes import GrepResult, LsResult
from langchain_mongodb_deepagents_vfs.errors import (
    ErrorCode,
    adapter_boundary,
    user_message,
)


@pytest.mark.unit
class TestErrorCode:
    def test_all_codes_have_messages(self):
        from langchain_mongodb_deepagents_vfs.errors import _CODE_MESSAGES

        for code in ErrorCode:
            assert code in _CODE_MESSAGES, f"{code} has no message"

    def test_code_values_unique(self):
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values))


@pytest.mark.unit
class TestUserMessage:
    def test_includes_code(self):
        msg = user_message(ErrorCode.E5001_GREP_FAILED)
        assert "E5001" in msg

    def test_includes_detail(self):
        msg = user_message(ErrorCode.E5001_GREP_FAILED, "timeout")
        assert "timeout" in msg


@pytest.mark.unit
class TestAdapterBoundary:
    def _make_instance(self, debug: bool = False) -> object:
        class FakeBackend:
            self_debug = debug

            def __init__(self, debug: bool):
                self.debug = debug

            @adapter_boundary(ErrorCode.E5001_GREP_FAILED)
            def grep(self, pattern: str, path: str = "", glob: str = "") -> GrepResult:
                raise ValueError("boom")

            @adapter_boundary(ErrorCode.E5001_GREP_FAILED)
            def grep_ok(self, pattern: str, path: str = "") -> GrepResult:
                return GrepResult(matches=[])

            @adapter_boundary(ErrorCode.E5003_LS_FAILED)
            def ls(self, path: str) -> LsResult:
                raise RuntimeError("ls broken")

        return FakeBackend(debug=debug)

    def test_exception_returns_error_dto(self):
        obj = self._make_instance(debug=False)
        result = obj.grep("query")
        assert result is not None
        assert result.error is not None
        assert "E5001" in result.error

    def test_success_passes_through(self):
        obj = self._make_instance(debug=False)
        result = obj.grep_ok("query")
        assert result.error is None
        assert result.matches == []

    def test_debug_mode_reraises(self):
        obj = self._make_instance(debug=True)
        with pytest.raises(ValueError, match="boom"):
            obj.grep("query")

    def test_ls_error_returns_ls_result(self):
        obj = self._make_instance(debug=False)
        result = obj.ls("/")
        assert result.error is not None
        assert "E5003" in result.error
