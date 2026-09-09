"""Loading source documents into normalised text.

PDFs are read with pypdf; plain text and markdown are read directly. Extraction
is deliberately dumb -- no layout reconstruction, no OCR, no table parsing --
because a clever extractor that behaves differently on two runs would be a
source of unexplained variance in every downstream number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from evalgate.hashing import sha256_text

# Collapse runs of spaces and tabs, and runs of three or more newlines, without
# touching paragraph structure: chunkers depend on blank lines as separators.
_HORIZONTAL_WS = re.compile("[ \\t\u00a0]+")  # includes U+00A0, common in PDF extraction
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
# PDF extraction commonly leaves a hyphen at a line break inside a word.
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n(\w)")

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIXES = frozenset({".pdf"})


class IngestError(RuntimeError):
    """Raised when a source document cannot be turned into text."""


@dataclass(frozen=True, slots=True)
class Document:
    """One source document as normalised text."""

    id: str
    title: str
    path: Path
    text: str

    @property
    def text_hash(self) -> str:
        """Hash of the extracted text, not of the source bytes."""
        return sha256_text(self.text)


def normalise(text: str) -> str:
    """Collapse extraction noise without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    text = _HORIZONTAL_WS.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise IngestError("pypdf is required to read PDF sources") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(page.strip() for page in pages):
        raise IngestError(f"no extractable text in {path} (is it a scan?)")
    return "\n\n".join(pages)


def load_document(path: Path, doc_id: str, title: str) -> Document:
    """Load one document from disk."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        raw = extract_pdf_text(path)
    elif suffix in TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8")
    else:
        raise IngestError(f"unsupported source type {suffix!r}: {path}")
    text = normalise(raw)
    if not text:
        raise IngestError(f"document is empty after normalisation: {path}")
    return Document(id=doc_id, title=title, path=path, text=text)
