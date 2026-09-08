# evalgate: repo constitution

Read this before changing anything. It is short on purpose.

## What this repo is

A RAG system over EU CBAM regulatory documents whose actual product is the
evaluation harness: a validated LLM judge, retrieval ablations, a
cost/latency/quality Pareto frontier, and a CI gate that blocks a pull request
when quality regresses.

The RAG application is intentionally unremarkable. When a change makes the
application cleverer at the cost of making a measurement less trustworthy, the
measurement wins.

## Architecture

    scripts/fetch_corpus.py   pinned URLs -> data/corpus/raw/ + SHA256 manifest
    evalgate.chunking         PDF text -> chunks. Pluggable, config-selected.
    evalgate.embeddings       local (bge) | api | hashed (deterministic)
    evalgate.retrieval        dense | bm25 | hybrid RRF | +cross-encoder rerank
    evalgate.generation       answer with citations, via the Anthropic API
    evalgate.judging          three-axis LLM judge, tool-use structured output
    evalgate.evaluation       retrieval metrics, judge metrics, cost, latency
    evalgate.gate             baseline comparison; the thing that fails the build
    evalgate.serve            thin FastAPI layer over the same pipeline

Data flows one way: corpus -> chunks -> index -> retrieval -> generation ->
judgment -> run record -> report -> gate. Nothing downstream writes upstream.

## Invariants

1. **Config is the only source of values.** No path, model name, k value,
   threshold, weight or prompt string is written in a `.py` file. If you are
   about to type a number into Python, it belongs in `configs/`.
2. **Prompts live in `prompts/` and are hashed.** Every run record carries the
   SHA256 of every prompt it used.
3. **Runs are immutable.** A run directory under `runs/` is written once and
   never edited. Re-running produces a new directory.
4. **Every measurement carries its fingerprint**: config hash, prompt hashes,
   corpus hash. A number without a fingerprint is not a result.
5. **Determinism by default.** Local embeddings, temperature 0, sorted-key
   hashing, fixed seeds. Sources of nondeterminism must be named and justified.
6. **Replay is the default API mode.** Live calls happen when someone asks for
   them, in `record` or `live` mode, never implicitly in a test or in PR CI.

## Never do these

- **Never parse judge output with a regex.** Judge output arrives through a
  tool-use call validated against a pydantic schema. If validation fails, retry
  with backoff; after the configured attempt limit, fail loudly. A default
  score on parse failure silently manufactures data and is the single most
  dangerous thing you could add to this repo.
- **Never compare runs across different prompt hashes, corpus hashes or config
  fingerprints.** The reporting layer must refuse to place them in the same
  table rather than footnote the difference.
- **Never let the gate pass on missing data.** Absent metrics fail the gate.
  "No baseline found" is a failure, not a pass.
- **Never commit a number that was typed by hand.** Every figure in a report or
  in the README is generated from a run artifact.
- **Never widen a threshold to make CI green.** Thresholds change only as a
  deliberate, separately-reviewed decision recorded in `docs/DECISIONS.md`.
- **Never make PR CI spend money.** PR CI replays cassettes. Live API calls
  belong in the nightly workflow.
- **Never label synthetic data as measured data.** Anything not produced by a
  real run against the real corpus says so, in the artifact and in the report.
- **Never skip or delete a failing test to get green.** Fix the cause.

## Conventions

- Python 3.12, `uv` for dependencies, `make` as the only interface.
- `make check` (ruff format check + ruff lint + mypy strict + pytest) must pass
  before every commit.
- Full type hints. `Any` at a library boundary needs a comment saying why.
- Docstrings say why, not what. The code already says what.
- No notebooks. No emoji. No marketing language in docs or reports.
- Every non-obvious choice gets an entry in `docs/DECISIONS.md`, which is
  append-only: correct an old entry with a new one, do not rewrite history.
