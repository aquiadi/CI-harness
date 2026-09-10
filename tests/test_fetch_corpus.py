"""A 2xx is a statement about the exchange, not about the body.

The corpus is the one input every downstream number depends on. A zero-byte
file that the manifest calls OK does not fail anything: it ingests to no
chunks, retrieves nothing, and produces a corpus hash that looks exactly as
valid as a real one. That is the failure this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.fetch_corpus import _reuse_existing, body_rejection

from evalgate.config import SourceDocument

PDF = b"%PDF-1.7\nreal document bytes"


def source(filename: str = "doc.pdf") -> SourceDocument:
    return SourceDocument(
        id="doc",
        title="A document",
        url="https://example.invalid/doc.pdf",
        filename=filename,
        kind="pdf",
        required=True,
    )


def test_a_real_pdf_is_accepted() -> None:
    assert body_rejection(PDF, "application/pdf", "pdf") is None


def test_an_empty_body_is_rejected() -> None:
    """The 0-byte case: sha256 e3b0c442... is the hash of nothing."""
    assert body_rejection(b"", "application/pdf", "pdf") == "empty response body"


def test_an_html_holding_page_is_rejected() -> None:
    """EUR-Lex answers 202 with HTML while it renders the PDF."""
    rejection = body_rejection(b"<!DOCTYPE html><html>...", "text/html", "pdf")
    assert rejection is not None
    assert "not a PDF" in rejection
    assert "text/html" in rejection


def test_the_rejection_quotes_the_body_so_the_cause_is_visible() -> None:
    """'not a PDF' alone does not tell you it was a login page or an error."""
    rejection = body_rejection(b"<html><title>Error 500</title>", "text/html", "pdf")
    assert rejection is not None and "Error 500" in rejection


def test_a_non_pdf_source_is_not_held_to_the_pdf_check() -> None:
    assert body_rejection(b"id,value\n1,2\n", "text/csv", "csv") is None


def test_a_cached_empty_file_is_deleted_rather_than_reused(tmp_path: Path) -> None:
    """Otherwise one bad download is permanent.

    `_reuse_existing` only checked that the path existed, so a zero-byte file
    was reported as cached on every subsequent run and never re-fetched.
    """
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"")
    assert _reuse_existing(source(), path, None) is None
    assert not path.exists(), "a rejected cache entry must not survive to the next run"


def test_a_cached_real_pdf_is_still_reused(tmp_path: Path) -> None:
    """The check must not defeat caching for documents that are fine."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(PDF)
    reused = _reuse_existing(source(), path, None)
    assert reused is not None and reused.bytes == len(PDF)
    assert path.exists()


@pytest.mark.parametrize("body", [b"", b"<!DOCTYPE html>", b'{"error": "not found"}'])
def test_nothing_that_is_not_a_pdf_reaches_the_manifest(body: bytes) -> None:
    assert body_rejection(body, "", "pdf") is not None
