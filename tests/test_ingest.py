from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalgate.corpus.loading import load_corpus, load_local_corpus
from evalgate.ingest.documents import IngestError, load_document, normalise


def test_normalise_collapses_horizontal_whitespace() -> None:
    assert normalise("a   \t b") == "a b"


def test_normalise_preserves_paragraph_breaks() -> None:
    assert normalise("a\n\nb") == "a\n\nb"


def test_normalise_collapses_excess_blank_lines() -> None:
    assert normalise("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_rejoins_words_split_across_lines() -> None:
    assert normalise("obli-\ngation") == "obligation"


def test_normalise_strips_non_breaking_spaces() -> None:
    assert normalise("a\u00a0\u00a0b") == "a b"


def test_normalise_is_idempotent(fixture_corpus_dir: Path) -> None:
    text = (fixture_corpus_dir / "fixture_reg_a.md").read_text(encoding="utf-8")
    assert normalise(normalise(text)) == normalise(text)


def test_load_document_records_a_text_hash(fixture_corpus_dir: Path) -> None:
    document = load_document(fixture_corpus_dir / "fixture_reg_a.md", "a", "A")
    assert len(document.text_hash) == 64
    assert document.text.startswith("SYNTHETIC TEST DOCUMENT")


def test_unsupported_suffix_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "doc.docx"
    path.write_bytes(b"x")
    with pytest.raises(IngestError, match="unsupported source type"):
        load_document(path, "d", "D")


def test_empty_document_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(IngestError, match="empty after normalisation"):
        load_document(path, "d", "D")


def test_local_corpus_is_ordered_and_hashed(fixture_corpus_dir: Path) -> None:
    corpus = load_local_corpus("fixture", fixture_corpus_dir)
    assert [document.id for document in corpus.documents] == sorted(
        document.id for document in corpus.documents
    )
    assert corpus.size == 3
    assert corpus.corpus_hash == load_local_corpus("fixture", fixture_corpus_dir).corpus_hash


def test_local_corpus_hash_changes_with_content(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "corpus"
    copy.mkdir()
    for path in fixture_corpus_dir.glob("*.md"):
        (copy / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    before = load_local_corpus("fixture", copy).corpus_hash
    (copy / "fixture_reg_a.md").write_text("SYNTHETIC. Article 1\nChanged body.", encoding="utf-8")
    assert load_local_corpus("fixture", copy).corpus_hash != before


def test_empty_local_corpus_is_refused(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match=r"no .* documents under"):
        load_local_corpus("fixture", tmp_path)


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no corpus manifest"):
        load_corpus("cbam", tmp_path / "raw", tmp_path / "manifest.json")


def write_manifest_json(path: Path, entries: list[dict[str, object]]) -> Path:
    """A manifest built for the test rather than borrowed from the repo."""
    path.write_text(
        json.dumps({"corpus": "cbam", "generated_at": "2026-01-01T00:00:00Z", "entries": entries}),
        encoding="utf-8",
    )
    return path


def manifest_entry(doc_id: str, status: str, **extra: object) -> dict[str, object]:
    return {
        "id": doc_id,
        "title": doc_id,
        "url": f"https://example.invalid/{doc_id}.pdf",
        "filename": f"{doc_id}.pdf",
        "kind": "pdf",
        "required": True,
        "status": status,
        "fetched_at": "2026-01-01T00:00:00Z",
        **extra,
    }


def test_manifest_with_no_fetched_documents_is_refused(tmp_path: Path) -> None:
    """A manifest with no successful fetches must fail loudly.

    This is the path a fresh clone hits before `make corpus`. It builds its own
    manifest rather than reading the committed one: that file is data, and once
    a real corpus was fetched it stopped describing an empty one, so borrowing
    it made this test fail for a reason unrelated to what it asserts.
    """
    manifest_path = write_manifest_json(
        tmp_path / "manifest.json",
        [
            manifest_entry("reg_a", "missing", error="HTTP 404"),
            manifest_entry("reg_b", "error", error="ConnectError"),
        ],
    )
    with pytest.raises(IngestError, match="records no successfully fetched documents"):
        load_corpus("cbam", tmp_path / "raw", manifest_path)


def test_a_manifest_that_promises_a_file_that_is_not_there_is_refused(tmp_path: Path) -> None:
    """The other half: OK in the manifest but nothing on disk.

    Reached by cloning a repo whose manifest is committed while the PDFs it
    names are gitignored, so it is the first thing a new contributor hits.
    """
    manifest_path = write_manifest_json(
        tmp_path / "manifest.json",
        [manifest_entry("reg_a", "ok", sha256="0" * 64, bytes=1024)],
    )
    with pytest.raises(IngestError, match="is missing; run `make corpus`"):
        load_corpus("cbam", tmp_path / "raw", manifest_path)
