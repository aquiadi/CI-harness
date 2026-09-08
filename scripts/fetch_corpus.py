#!/usr/bin/env python
"""Download the pinned corpus and write its SHA256 manifest.

Usage:
    make corpus
    make corpus ARGS="corpus.refetch=true"
    uv run python scripts/fetch_corpus.py corpus=cbam

The PDFs land in ``data/corpus/raw/`` and are gitignored. The manifest is
committed. A URL that 404s, times out or is blocked by a network policy is
recorded with its failure and does not stop the run: a partial corpus is a
condition the rest of the pipeline reports on, not a crash.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
from rich.console import Console

from evalgate.config import CorpusConfig, SourceDocument, load_config, resolve_path, typed_node
from evalgate.corpus.manifest import (
    CorpusManifest,
    FetchStatus,
    ManifestEntry,
    load_manifest_if_present,
    utc_now_iso,
    write_manifest,
)
from evalgate.hashing import sha256_file, short

console = Console()

# 404 and 410 mean the document is not there; anything else is a transport or
# server problem worth retrying.
_MISSING_STATUSES = frozenset({404, 410})


def _entry(
    source: SourceDocument,
    status: FetchStatus,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    http_status: int | None = None,
    content_type: str | None = None,
    error: str | None = None,
    fetched_at: str | None = None,
) -> ManifestEntry:
    """Build a manifest entry from a source spec and an outcome."""
    return ManifestEntry(
        id=source.id,
        title=source.title,
        url=source.url,
        filename=source.filename,
        kind=source.kind,
        required=source.required,
        status=status,
        sha256=sha256,
        bytes=size_bytes,
        http_status=http_status,
        content_type=content_type,
        error=error,
        fetched_at=fetched_at or utc_now_iso(),
    )


def _reuse_existing(
    source: SourceDocument, path: Path, prior: ManifestEntry | None
) -> ManifestEntry | None:
    """Return a manifest entry for an already-downloaded file, if it is intact."""
    if not path.is_file():
        return None
    digest = sha256_file(path)
    if prior is not None and prior.sha256 is not None and prior.sha256 != digest:
        console.print(
            f"[yellow]{source.id}: on-disk hash {short(digest)} differs from manifest "
            f"{short(prior.sha256)}; re-downloading[/yellow]"
        )
        return None
    return _entry(
        source,
        FetchStatus.OK,
        sha256=digest,
        size_bytes=path.stat().st_size,
        http_status=prior.http_status if prior else None,
        content_type=prior.content_type if prior else None,
        fetched_at=prior.fetched_at if prior else None,
    )


def fetch_one(
    client: httpx.Client,
    source: SourceDocument,
    dest_dir: Path,
    max_attempts: int,
) -> ManifestEntry:
    """Fetch one source document, returning its manifest entry either way."""
    path = dest_dir / source.filename
    last_error = ""
    last_status: int | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(source.url)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            last_status = response.status_code
            if response.status_code in _MISSING_STATUSES:
                console.print(
                    f"[yellow]{source.id}: HTTP {response.status_code}, recorded missing[/yellow]"
                )
                return _entry(
                    source,
                    FetchStatus.MISSING,
                    http_status=response.status_code,
                    error=f"HTTP {response.status_code}",
                )
            if response.is_success:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                digest = sha256_file(path)
                console.print(
                    f"[green]{source.id}[/green]: {len(response.content):,} bytes, "
                    f"sha256 {short(digest)}"
                )
                return _entry(
                    source,
                    FetchStatus.OK,
                    sha256=digest,
                    size_bytes=len(response.content),
                    http_status=response.status_code,
                    content_type=response.headers.get("content-type"),
                )
            last_error = f"HTTP {response.status_code}"

        if attempt < max_attempts:
            delay = float(2 ** (attempt - 1))
            console.print(
                f"[yellow]{source.id}: {last_error}; "
                f"retry {attempt}/{max_attempts - 1} in {delay:.0f}s[/yellow]"
            )
            time.sleep(delay)

    console.print(f"[red]{source.id}: {last_error}, recorded as error[/red]")
    return _entry(source, FetchStatus.ERROR, http_status=last_status, error=last_error)


def main(argv: list[str] | None = None) -> int:
    """Fetch every configured source and rewrite the manifest."""
    cfg = load_config(overrides=list(argv if argv is not None else sys.argv[1:]))
    corpus = typed_node(cfg, "corpus", CorpusConfig)
    dest_dir = resolve_path(cfg, "paths.corpus_raw_dir")
    manifest_path = resolve_path(cfg, "paths.corpus_manifest_path")
    dest_dir.mkdir(parents=True, exist_ok=True)

    prior = load_manifest_if_present(manifest_path)
    entries: list[ManifestEntry] = []

    headers = {"User-Agent": corpus.user_agent, "Accept": "*/*"}
    with httpx.Client(
        follow_redirects=True,
        timeout=corpus.request_timeout_s,
        headers=headers,
    ) as client:
        for source in corpus.sources:
            if not corpus.refetch:
                reused = _reuse_existing(
                    source,
                    dest_dir / source.filename,
                    prior.entry(source.id) if prior else None,
                )
                if reused is not None:
                    console.print(
                        f"[dim]{source.id}: cached, sha256 {short(reused.sha256 or '')}[/dim]"
                    )
                    entries.append(reused)
                    continue
            entries.append(fetch_one(client, source, dest_dir, corpus.max_attempts))

    manifest = CorpusManifest(corpus=corpus.name, generated_at=utc_now_iso(), entries=entries)
    write_manifest(manifest_path, manifest)

    ok = manifest.ok_entries
    failed = manifest.failed_entries
    console.print(
        f"\n{len(ok)}/{len(entries)} documents present, corpus_hash={short(manifest.corpus_hash())}"
    )
    if failed:
        console.print("[yellow]unavailable:[/yellow]")
        for entry in failed:
            flag = "required" if entry.required else "optional"
            console.print(f"  {entry.id} ({flag}): {entry.status} {entry.error or ''}")
    console.print(f"manifest written to {manifest_path}")

    # A missing optional document is fine. A missing required document is a
    # condition the caller should see, but it is not a crash: the manifest is
    # still written, and the rest of the pipeline reports on what is present.
    missing_required = [e for e in failed if e.required]
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
