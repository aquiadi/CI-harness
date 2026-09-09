from __future__ import annotations

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


def test_manifest_with_no_fetched_documents_is_refused(repo_root: Path, tmp_path: Path) -> None:
    """A manifest with no successful fetches must fail loudly.

    The committed manifest records every source as unreachable from the build
    sandbox, so this is the path a fresh clone hits before `make corpus`.
    """
    manifest_path = repo_root / "data" / "corpus" / "manifest.json"
    with pytest.raises(IngestError, match="records no successfully fetched documents"):
        load_corpus("cbam", tmp_path / "raw", manifest_path)
