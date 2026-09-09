from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalgate.config import load_config
from evalgate.corpus.manifest import (
    CorpusManifest,
    FetchStatus,
    ManifestEntry,
    load_manifest_if_present,
    read_manifest,
    utc_now_iso,
    write_manifest,
)


def entry(source_id: str, status: FetchStatus, digest: str | None = None) -> ManifestEntry:
    return ManifestEntry(
        id=source_id,
        title=f"title {source_id}",
        url=f"https://example.invalid/{source_id}.pdf",
        filename=f"{source_id}.pdf",
        kind="pdf",
        required=True,
        status=status,
        sha256=digest,
    )


def test_round_trip(tmp_path: Path) -> None:
    manifest = CorpusManifest(
        corpus="cbam",
        generated_at=utc_now_iso(),
        entries=[entry("a", FetchStatus.OK, "a" * 64), entry("b", FetchStatus.MISSING)],
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest


def test_ok_and_failed_partition() -> None:
    manifest = CorpusManifest(
        corpus="c",
        generated_at=utc_now_iso(),
        entries=[
            entry("a", FetchStatus.OK, "a" * 64),
            entry("b", FetchStatus.MISSING),
            entry("c", FetchStatus.ERROR),
        ],
    )
    assert [e.id for e in manifest.ok_entries] == ["a"]
    assert [e.id for e in manifest.failed_entries] == ["b", "c"]


def test_corpus_hash_ignores_failed_entries() -> None:
    ok_only = CorpusManifest(
        corpus="c", generated_at=utc_now_iso(), entries=[entry("a", FetchStatus.OK, "a" * 64)]
    )
    with_failure = CorpusManifest(
        corpus="c",
        generated_at=utc_now_iso(),
        entries=[entry("a", FetchStatus.OK, "a" * 64), entry("b", FetchStatus.ERROR)],
    )
    assert ok_only.corpus_hash() == with_failure.corpus_hash()


def test_corpus_hash_changes_when_a_document_changes() -> None:
    first = CorpusManifest(
        corpus="c", generated_at=utc_now_iso(), entries=[entry("a", FetchStatus.OK, "a" * 64)]
    )
    second = CorpusManifest(
        corpus="c", generated_at=utc_now_iso(), entries=[entry("a", FetchStatus.OK, "b" * 64)]
    )
    assert first.corpus_hash() != second.corpus_hash()


def test_corpus_hash_is_independent_of_generation_time() -> None:
    entries = [entry("a", FetchStatus.OK, "a" * 64)]
    assert (
        CorpusManifest(
            corpus="c", generated_at="2020-01-01T00:00:00Z", entries=entries
        ).corpus_hash()
        == CorpusManifest(
            corpus="c", generated_at="2030-01-01T00:00:00Z", entries=entries
        ).corpus_hash()
    )


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text('{"corpus":"c","generated_at":"t","entries":[],"surprise":1}', encoding="utf-8")
    with pytest.raises(ValidationError):
        read_manifest(path)


def test_load_missing_manifest_returns_none(tmp_path: Path) -> None:
    assert load_manifest_if_present(tmp_path / "absent.json") is None


def test_committed_manifest_is_valid_and_records_every_real_source(repo_root: Path) -> None:
    real = load_config(overrides=["corpus=cbam"])
    manifest = load_manifest_if_present(repo_root / "data" / "corpus" / "manifest.json")
    assert manifest is not None, "run `make corpus` first"
    assert {e.id for e in manifest.entries} == {s.id for s in real.corpus.sources}
