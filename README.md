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

    make dev                                   # install (uv, Python 3.12)
    make check                                 # format, lint, types, tests
    make index ARGS="embedder=hashed"          # build an index, no downloads
    make label ARGS="embedder=hashed"          # rate answers on three axes

That works on a fresh clone with no network access and no API key: the default
corpus is the committed synthetic one and `embedder=hashed` is a deterministic
local embedder that needs no model download.

For the real setup:

    make ml            # add sentence-transformers + torch for bge embeddings
    make corpus        # fetch and pin the real EU documents by SHA256
    make index ARGS="corpus=cbam"

`make help` lists the rest. The Makefile is the only interface you should need.

## Corpus

The default corpus is **synthetic** -- five documents written for this
repository in the structural style of the CBAM instruments, marked as such in
every file, so that the harness has something to measure where the real sources
are unreachable. See `data/corpus/synthetic/README.md`. `corpus=cbam` selects
the real EU instruments, pinned by SHA256 in `data/corpus/manifest.json`. Every
run record and every report names the corpus it used.

## The CI split: replay on every pull request, live once a night

This is the most interesting engineering decision in the project, so it gets
the most space.

An evaluation harness that calls a model has a problem: the thing that makes it
valuable -- running on every pull request -- is also what makes it expensive and
flaky. Running the full suite live on every push costs money per push, takes
minutes, and fails for reasons that have nothing to do with the change under
review: a rate limit, a timeout, a model that is one token less deterministic
than it was yesterday. Teams respond by running the suite nightly instead, and
a gate that runs after the merge is not a gate.

evalgate splits the difference:

**Every pull request replays.** All model access goes through one client stack
with three modes. In `replay` -- the default, and what CI runs -- a call is
served from a committed cassette: one JSON file per call, holding the request
and the response, reviewable in a diff. An unmatched call is an error, never a
network request. The result is a full evaluation, including generation, judging
and the quality gate, that is deterministic, takes seconds, costs nothing, and
cannot fail because of somebody else's rate limit. A change that alters a
prompt, a model or a config surfaces as a cassette miss with the command to
re-record, rather than as a surprise invoice.

**One scheduled job runs live.** `.github/workflows/nightly.yml` fetches the
real corpus, installs the local encoder, and runs the same suite against the
real API with its own frozen baseline (`baseline.live.json`). If the gate fails
it opens an issue. Drift that only a live model can show -- a provider-side
model update, a change in refusal behaviour, latency creep -- is found there, at
a cost of one run a day rather than one per push.

The two halves catch different things and neither substitutes for the other.
Replay catches "this change made retrieval worse", which is most changes. Live
catches "the world moved underneath us", which is rarer and slower-moving. The
split is what makes it affordable to gate every pull request on quality at all.

`make record` re-records the cassettes; the recording is a reviewable diff, so
a change in what the model says is something a human sees rather than something
that silently shifts a number.

*State of the cassettes in this repository:* empty. The environment this repo
was built in had no API credentials (`docs/DECISIONS.md`, D-0009), so no live
call was ever made and there was nothing honest to record. The replay path is
exercised by unit tests, and the committed gate runs on the offline stack --
the extractive generator and the rule-based judge -- which needs no cassettes at
all. Recording them is one command for anyone with a key.

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
