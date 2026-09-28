"""Unit tests for the Chunker module."""

from __future__ import annotations

import pytest

from langchain_mongodb_deepagents_vfs.chunker import Chunker
from langchain_mongodb_deepagents_vfs.dtypes import Chunk
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode


@pytest.mark.unit
class TestChunkerPlainText:
    def test_basic_chunk_returns_list(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("docs/sample.txt", sample_text_bytes)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_type(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("docs/sample.txt", sample_text_bytes)
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_source_path_preserved(self, chunker, sample_text_bytes):
        path = "docs/my_file.txt"
        chunks = chunker.chunk(path, sample_text_bytes)
        for c in chunks:
            assert c.source_path == path

    def test_chunk_indices_sequential(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("sample.txt", sample_text_bytes)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_content_non_empty(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("sample.txt", sample_text_bytes)
        for c in chunks:
            assert c.content.strip() != ""

    def test_line_start_non_negative(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("sample.txt", sample_text_bytes)
        for c in chunks:
            assert c.line_start >= 0

    def test_char_offsets_ordered(self, chunker, sample_text_bytes):
        chunks = chunker.chunk("sample.txt", sample_text_bytes)
        for c in chunks:
            assert c.char_start <= c.char_end

    def test_token_limit_respected(self):
        chunker = Chunker(token_limit=10, overlap=2)
        # 100 tokens of text
        text = ("hello world " * 50).encode("utf-8")
        chunks = chunker.chunk("big.txt", text)
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        for c in chunks:
            token_count = len(enc.encode(c.content))
            assert token_count <= 12  # small margin for boundary rounding

    def test_empty_file_returns_no_chunks(self, chunker):
        chunks = chunker.chunk("empty.txt", b"")
        assert chunks == []

    def test_md_extension_treated_as_text(self, chunker):
        data = b"# Heading\n\nSome markdown content."
        chunks = chunker.chunk("README.md", data)
        assert len(chunks) > 0

    def test_unknown_extension_falls_back_to_text(self, chunker):
        data = b"Some plain text in a weird extension."
        chunks = chunker.chunk("file.xyz", data)
        assert len(chunks) > 0


@pytest.mark.unit
class TestChunkerPDF:
    def test_pdf_extraction(self, chunker, sample_pdf_bytes):
        chunks = chunker.chunk("report.pdf", sample_pdf_bytes)
        # A minimal PDF may have 0 or 1 chunks depending on content
        assert isinstance(chunks, list)

    def test_pdf_page_number_set(self, chunker, sample_pdf_bytes):
        chunks = chunker.chunk("report.pdf", sample_pdf_bytes)
        for c in chunks:
            assert c.page_number >= 0

    def test_pdf_page_count_bomb_rejected(self, chunker, sample_pdf_bytes, monkeypatch):
        """A page-count bomb is refused before per-page extraction runs.

        PDF content streams are Flate-compressed, so a small file can declare
        enormous content — the same bomb class _guard_ooxml blocks for Office
        formats, which has no archive metadata equivalent for PDF.
        """
        import langchain_mongodb_deepagents_vfs.chunker as chunker_mod

        monkeypatch.setattr(chunker_mod, "_MAX_PDF_PAGES", 0)
        with pytest.raises(AdapterError) as ei:
            chunker.chunk("bomb.pdf", sample_pdf_bytes)
        assert ei.value.code == ErrorCode.E9002_CHUNKER_FAILED

    def test_pdf_text_expansion_bomb_rejected(
        self, chunker, sample_pdf_bytes, monkeypatch
    ):
        """Extraction stops once the decompressed text passes the budget."""
        import langchain_mongodb_deepagents_vfs.chunker as chunker_mod

        monkeypatch.setattr(chunker_mod, "_MAX_EXTRACTED_CHARS", 1)
        with pytest.raises(AdapterError) as ei:
            chunker.chunk("bomb.pdf", sample_pdf_bytes)
        assert ei.value.code == ErrorCode.E9002_CHUNKER_FAILED


@pytest.mark.unit
class TestChunkerMagicBytes:
    def test_pdf_magic_bytes_detected(self, chunker, sample_pdf_bytes):
        # Rename to unknown extension — should still parse as PDF
        chunks = chunker.chunk("report.dat", sample_pdf_bytes)
        assert isinstance(chunks, list)


@pytest.mark.unit
class TestChunkerErrors:
    def test_unsupported_format_still_falls_back(self, chunker):
        # We expect graceful fallback, not an exception, for unknown text-like content
        data = b"Some content"
        chunks = chunker.chunk("file.abc123", data)
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Office formats: xlsx / xls / pptx / ppt
# ---------------------------------------------------------------------------

import tiktoken  # noqa: E402

_ENC = tiktoken.get_encoding("cl100k_base")


def _within_token_budget(chunks, limit=128, slack=4):
    """Token budget check matching the chunker fixture's token_limit=128."""
    for c in chunks:
        assert len(_ENC.encode(c.content)) <= limit + slack, (
            f"chunk {c.chunk_index} exceeded token budget"
        )


@pytest.mark.unit
class TestChunkerXLSX:
    def test_xlsx_dispatches_per_sheet(self, chunker, sample_xlsx_bytes):
        chunks = chunker.chunk("reports/q1.xlsx", sample_xlsx_bytes)
        assert len(chunks) > 0
        # Two sheets → at least two distinct page_numbers
        assert {c.page_number for c in chunks} >= {0, 1}

    def test_xlsx_sheet_title_emitted(self, chunker, sample_xlsx_bytes):
        chunks = chunker.chunk("reports/q1.xlsx", sample_xlsx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "# Sheet: Revenue" in joined
        assert "# Sheet: Notes" in joined

    def test_xlsx_cell_values_preserved(self, chunker, sample_xlsx_bytes):
        chunks = chunker.chunk("reports/q1.xlsx", sample_xlsx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "EMEA" in joined and "APAC" in joined and "AMER" in joined
        assert "rankFusion" in joined  # from Notes sheet

    def test_xlsx_chunk_invariants(self, chunker, sample_xlsx_bytes):
        chunks = chunker.chunk("reports/q1.xlsx", sample_xlsx_bytes)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.char_start <= c.char_end
            assert c.line_start >= 0
            assert c.source_path == "reports/q1.xlsx"
        _within_token_budget(chunks)

    def test_xlsx_magic_byte_fallback(self, chunker, sample_xlsx_bytes):
        # Strip extension — should still parse via ZIP-sniffing fallback
        chunks = chunker.chunk("reports/q1.dat", sample_xlsx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "# Sheet: Revenue" in joined


@pytest.mark.unit
class TestChunkerXLS:
    def test_xls_dispatches_per_sheet(self, chunker, sample_xls_bytes):
        chunks = chunker.chunk("legacy/q1.xls", sample_xls_bytes)
        assert len(chunks) > 0
        assert {c.page_number for c in chunks} >= {0, 1}

    def test_xls_sheet_title_emitted(self, chunker, sample_xls_bytes):
        chunks = chunker.chunk("legacy/q1.xls", sample_xls_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "# Sheet: Revenue" in joined
        assert "# Sheet: Notes" in joined

    def test_xls_cell_values_preserved(self, chunker, sample_xls_bytes):
        chunks = chunker.chunk("legacy/q1.xls", sample_xls_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "EMEA" in joined and "rankFusion" in joined

    def test_xls_chunk_invariants(self, chunker, sample_xls_bytes):
        chunks = chunker.chunk("legacy/q1.xls", sample_xls_bytes)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.char_start <= c.char_end
        _within_token_budget(chunks)

    def test_xls_magic_byte_fallback(self, chunker, sample_xls_bytes):
        # OLE2 magic should trigger xls extraction even with wrong extension
        chunks = chunker.chunk("legacy/q1.bin", sample_xls_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "EMEA" in joined


@pytest.mark.unit
class TestChunkerPPTX:
    def test_pptx_dispatches_per_slide(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.pptx", sample_pptx_bytes)
        assert len(chunks) > 0
        # Two slides → distinct page_numbers
        assert {c.page_number for c in chunks} >= {0, 1}

    def test_pptx_text_extracted(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.pptx", sample_pptx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "Hybrid Retrieval Overview" in joined
        assert "RRF" in joined

    def test_pptx_table_extracted(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.pptx", sample_pptx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "nDCG@10" in joined
        assert "Hybrid (RRF)" in joined

    def test_pptx_notes_extracted(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.pptx", sample_pptx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "rankFusion" in joined  # speaker notes

    def test_pptx_chunk_invariants(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.pptx", sample_pptx_bytes)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.char_start <= c.char_end
        _within_token_budget(chunks)

    def test_pptx_magic_byte_fallback(self, chunker, sample_pptx_bytes):
        chunks = chunker.chunk("decks/intro.dat", sample_pptx_bytes)
        joined = "\n".join(c.content for c in chunks)
        assert "Hybrid Retrieval Overview" in joined


@pytest.mark.unit
class TestChunkerPPTLegacy:
    """Legacy binary .ppt has no Python writer; test the helpers directly."""

    def test_scan_ole_strings_extracts_ascii(self):
        blob = b"\x00\x01garbage\x00" + b"HelloWorld" + b"\xff\xfe\x00"
        out = Chunker._scan_ole_strings(blob, min_len=4)
        assert "HelloWorld" in out

    def test_scan_ole_strings_extracts_utf16le(self):
        # "Slide Title" encoded as UTF-16LE
        utf16 = "Slide Title".encode("utf-16-le")
        blob = b"\x00\x00\x00" + utf16 + b"\xff\xff"
        out = Chunker._scan_ole_strings(blob, min_len=4)
        assert "Slide Title" in out

    def test_scan_ole_strings_ignores_short_runs(self):
        blob = b"\x00ab\x00cd\x00ef\x00"  # all runs <4
        out = Chunker._scan_ole_strings(blob, min_len=4)
        assert out.strip() == ""

    def test_ppt_invalid_bytes_raise_adapter_error(self, chunker):
        # Not a valid OLE2 file — olefile should reject it
        with pytest.raises(AdapterError) as ei:
            chunker.chunk("legacy/deck.ppt", b"definitely not an OLE2 file")
        assert ei.value.code in (
            ErrorCode.E9002_CHUNKER_FAILED,
            ErrorCode.E9003_FORMAT_UNSUPPORTED,
        )


@pytest.mark.unit
class TestChunkerZipBombGuard:
    @staticmethod
    def _zip_bomb() -> bytes:
        """A tiny archive that inflates far past the expansion-ratio limit."""
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("word/document.xml", b"\x00" * (10 * 1024 * 1024))
        return buf.getvalue()

    def test_guard_rejects_high_expansion_ratio(self):
        with pytest.raises(AdapterError) as ei:
            Chunker._guard_ooxml(self._zip_bomb())
        assert ei.value.code == ErrorCode.E9002_CHUNKER_FAILED

    def test_docx_extraction_guarded(self, chunker):
        # Reaches _extract_docx via the docx extension and is rejected pre-parse.
        with pytest.raises(AdapterError) as ei:
            chunker.chunk("bomb.docx", self._zip_bomb())
        assert ei.value.code == ErrorCode.E9002_CHUNKER_FAILED
