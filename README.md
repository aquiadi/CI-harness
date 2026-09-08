# evalgate

A retrieval-augmented question answering system over EU CBAM regulatory
documents, whose actual product is the evaluation harness around it: an LLM
judge validated against human labels, retrieval ablations on a
cost/latency/quality Pareto frontier, and a CI gate that fails a pull request
when quality regresses.

The RAG application is deliberately ordinary. The measurement is the point.

Status: under construction. Headline numbers, the judge calibration report and
the Pareto table land with milestones M3-M5 and are written here from measured
run artifacts, not by hand.

## Quick start

    make dev      # install runtime + dev dependencies (uv, Python 3.12)
    make check    # ruff format check, ruff lint, mypy strict, pytest
    make corpus   # download the pinned sources, write data/corpus/manifest.json
    make config   # print the resolved config and every hash it implies

`make help` lists the rest. The Makefile is the only interface you should need.

## Tooling we deliberately skipped

No DVC. No MLflow. Both would be ceremony here, and the reason is worth stating
because "we use DVC" is easier to put on a slide than "we thought about it".

**Data versioning (DVC).** The things that need versioning are the eval sets
and the corpus pin. The eval sets are small JSONL files -- 120 retrieval slots,
150 answer pairs -- that live in git, diff as text in review, and are the sort
of artifact where a reviewer genuinely wants to see the line-level change. The
corpus is large and not ours to redistribute, so what is versioned is
`data/corpus/manifest.json`: one SHA256 per source document, which pins the
corpus exactly and reproduces it from public URLs. A content-addressed blob
store in front of that would add a remote to configure and a cache to
invalidate, in exchange for versioning bytes we deliberately do not commit.

**Experiment tracking (MLflow).** Every run writes one immutable parquet under
`runs/<timestamp>-<confighash>/`, keyed by a fingerprint of the config that
produced it, alongside the prompt hashes and corpus hash it ran under. Reports
are generated from those files. That makes the run history queryable with
pandas, diffable in git, and portable in a tarball, with no server to stand up,
no schema migration, and no second source of truth to reconcile when the
tracking server and the artifacts disagree. A tracking server earns its keep
when many people run many experiments concurrently and need a shared UI. That
is not this repo.

Both decisions are recorded with their tradeoffs in `docs/DECISIONS.md`.

## Repository layout

    configs/      hydra config tree; all paths, models, k values, thresholds
    prompts/      versioned prompt bodies, hashed into every run record
    src/evalgate/ the library
    scripts/      corpus acquisition
    tests/        pytest suite, including retrieval property tests
    docs/         DECISIONS.md, append-only
    runs/         immutable per-run parquet artifacts
    reports/      generated markdown reports

## License

MIT.
