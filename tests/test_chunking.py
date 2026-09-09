from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from omegaconf import DictConfig

from evalgate.chunking.base import Chunk, Chunker
from evalgate.config import load_config
from evalgate.ingest.documents import Document, load_document, normalise
from evalgate.ingest.tokens import RegexTokenEstimator
from evalgate.pipeline import build_chunker, build_tokenizer

CHUNKERS = ["fixed_token", "recursive_structural", "section_aware"]
TOKENIZER = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)


def chunker_for(name: str, target: int = 96) -> Chunker:
    cfg: DictConfig = load_config(overrides=[f"chunker={name}", f"chunker.target_tokens={target}"])
    return build_chunker(cfg, build_tokenizer(cfg))


@pytest.fixture
def document(fixture_corpus_dir: Path) -> Document:
    path = fixture_corpus_dir / "fixture_reg_a.md"
    return load_document(path, "fixture_reg_a", "Fixture Regulation A")


@pytest.mark.parametrize("name", CHUNKERS)
def test_chunks_are_substrings_at_their_recorded_offsets(name: str, document: Document) -> None:
    for chunk in chunker_for(name).chunk(document):
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text


@pytest.mark.parametrize("name", CHUNKERS)
def test_chunk_ids_are_unique_and_ordinally_ordered(name: str, document: Document) -> None:
    chunks = chunker_for(name).chunk(document)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.id.startswith(f"{document.id}#") for chunk in chunks)


@pytest.mark.parametrize("name", CHUNKERS)
def test_no_chunk_is_empty_or_untrimmed(name: str, document: Document) -> None:
    for chunk in chunker_for(name).chunk(document):
        assert chunk.text == chunk.text.strip()
        assert chunk.text


@pytest.mark.parametrize("name", CHUNKERS)
def test_every_token_of_the_document_appears_in_some_chunk(name: str, document: Document) -> None:
    """No chunker may silently drop text; a lost obligation is unrecoverable."""
    covered: set[int] = set()
    for chunk in chunker_for(name).chunk(document):
        covered.update(
            start
            for start, _ in TOKENIZER.spans(document.text)
            if chunk.char_start <= start < chunk.char_end
        )
    assert covered == {start for start, _ in TOKENIZER.spans(document.text)}


@pytest.mark.parametrize("name", CHUNKERS)
def test_chunking_is_deterministic(name: str, document: Document) -> None:
    first = chunker_for(name).chunk(document)
    second = chunker_for(name).chunk(document)
    assert first == second


@pytest.mark.parametrize("name", CHUNKERS)
def test_fingerprint_changes_with_target_size(name: str) -> None:
    assert chunker_for(name, 96).fingerprint() != chunker_for(name, 192).fingerprint()


def test_section_aware_never_puts_two_articles_in_one_chunk(document: Document) -> None:
    """The invariant the strategy exists for: one article's obligations per chunk."""
    chunks = chunker_for("section_aware", 512).chunk(document)
    for chunk in chunks:
        headings = [line for line in chunk.text.splitlines() if line.strip().startswith("Article ")]
        assert len(headings) <= 1, chunk.id


def test_section_aware_labels_name_a_heading_inside_the_chunk(document: Document) -> None:
    for chunk in chunker_for("section_aware", 512).chunk(document):
        if chunk.section:
            assert chunk.section in " ".join(chunk.text.split())


def test_section_aware_keeps_a_container_heading_with_its_article(document: Document) -> None:
    """CHAPTER II belongs with the article it introduces, not the one before it."""
    chunks = chunker_for("section_aware", 512).chunk(document)
    holder = next(chunk for chunk in chunks if "CHAPTER II" in chunk.text)
    assert holder.section == "Article 3"


def test_section_aware_does_not_treat_a_wrapped_date_as_a_heading(document: Document) -> None:
    """A wrapped date at the start of a line is prose, not a numbered heading."""
    chunks = chunker_for("section_aware", 512).chunk(document)
    assert not any(chunk.text.startswith("31 July") for chunk in chunks)


def test_fixed_token_windows_respect_the_target(document: Document) -> None:
    chunks = chunker_for("fixed_token", 96).chunk(document)
    assert len(chunks) > 1
    # Estimation is approximate, so allow the slack of one atom's worth.
    assert all(chunk.n_tokens <= 96 + 2 for chunk in chunks)


@pytest.mark.parametrize("name", CHUNKERS)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(body=st.text(alphabet=st.characters(codec="ascii"), min_size=1, max_size=800))
def test_chunker_never_raises_on_arbitrary_text(name: str, body: str) -> None:
    text = normalise(body)
    if not text:
        return
    document = Document(id="d", title="d", path=Path("d.md"), text=text)
    chunks: list[Chunk] = chunker_for(name).chunk(document)
    assert all(chunk.text for chunk in chunks)
