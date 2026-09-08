# Decisions

Append-only. One entry per non-obvious choice, with the tradeoff stated. To
change a past decision, add a new entry that supersedes it; do not edit the old
one.

Format: `D-NNNN` / date / decision / why / what it costs.

---

## D-0001 -- Makefile as the only interface, uv underneath

2026-09-08, M0

Every workflow is a make target; every make target is a thin wrapper over
`uv run`. No target contains logic beyond argument plumbing.

Why: one place to look, and the interface stays stable while the
implementation moves. `uv` gives a locked, reproducible environment in one
command and resolves fast enough that CI can install from scratch.

Cost: an extra indirection when debugging, and `uv` is younger than pip. The
lockfile is committed, so a `uv` regression pins rather than breaks us.

## D-0002 -- All configuration in hydra, zero constants in Python

2026-09-08, M0

Paths, model names, k values, thresholds, prompt references and sweep matrices
live in `configs/`. Python reads them. The one bootstrap exception is
`evalgate.rootdir`, which locates `configs/` in the first place.

Why: the repo's claim is that its measurements are trustworthy. That requires
knowing exactly what produced a number, which requires the full parameter set
to be one serialisable object that can be hashed into the run record. Constants
scattered across modules cannot be hashed and quietly invalidate comparisons.

Cost: indirection. Reading a value means finding its YAML. Structured-config
schemas in `evalgate/config.py` mitigate this by typing and validating the tree
at compose time, so a typo fails immediately rather than at use.

## D-0003 -- Prompts are versioned files, hashed into every run

2026-09-08, M0

Prompts are markdown under `prompts/`, loaded by relative path, hashed with
SHA256 at load. Substitution uses `string.Template` (`${var}`), strictly.

Why: a prompt edit changes results as surely as a model swap. Recording the
hash makes "these two runs are comparable" a checkable claim instead of a
belief. `${var}` rather than `str.format` so that JSON examples, markdown and
currency symbols inside prompt bodies need no escaping.

Cost: prompts cannot be composed programmatically at call sites; a variant is
a new file. That is the intended pressure -- variants should be visible in
`git log`, not synthesised at runtime.

## D-0004 -- Heavy model dependencies live behind an optional extra

2026-09-08, M0

`sentence-transformers` and `torch` are in the `ml` extra, not the default
dependency set. `make dev` (and therefore lint, types, tests and replay CI) does
not install them.

Why: PR CI runs on every push and must be fast and free. Installing a
multi-gigabyte torch wheel to run a lint job is a tax paid many times a day for
no benefit. The pipeline's default embedder needs the extra; the deterministic
one (D-0005) does not, and that is what CI and the tests use.

Cost: `mypy` cannot see `sentence_transformers` types in the default
environment, so those modules carry an explicit per-module override rather than
a global `ignore_missing_imports`, and the local-embedder code path is
type-checked but not exercised by the default test run. It is exercised by
tests marked `slow` and by the nightly workflow.

## D-0005 -- A deterministic, dependency-free embedder is a first-class backend

2026-09-08, M0

`embedder=hashed` is a signed random projection of hashed token n-grams:
real vectors, real arithmetic, no model download, byte-identical on every
machine.

Why: three things need embeddings that cost nothing and never vary -- the test
suite, replay-mode CI, and anyone reproducing the repo behind a network policy
that blocks model hosts. Making the deterministic path a real backend rather
than a test double means CI exercises the same retrieval code as production.

Cost: retrieval quality under `embedder=hashed` is much worse than under
`embedder=local`, so any number produced with it is labelled with the embedder
that produced it and is never presented as the system's quality.

## D-0006 -- The corpus is pinned by manifest; the PDFs are not committed

2026-09-08, M0

`scripts/fetch_corpus.py` downloads the pinned URLs into a gitignored
directory and writes `data/corpus/manifest.json`: per document, the SHA256 of
the bytes actually received. A URL that 404s or is unreachable is recorded with
its failure and the run continues.

Why: the documents are large and redistributing them is not our call, but their
identity is exactly what has to be pinned -- a silent upstream revision of a
guidance PDF would otherwise appear as unexplained movement in eval numbers.
The manifest is small, diffable, and reproduces the corpus from public URLs.
Recording failures rather than raising keeps a partial corpus a measurable
condition rather than an outage.

Cost: reproduction depends on the URLs staying up. If EUR-Lex reorganises,
the manifest documents exactly what is missing but cannot recover it. The
`corpus_hash` covers only successfully fetched documents, so a corpus that
later gains a previously-unreachable document hashes differently -- correct,
but it means old runs become incomparable rather than merely incomplete.

## D-0007 -- No DVC, no MLflow

2026-09-08, M0

Eval sets are small JSONL files versioned in git. Run artifacts are immutable
parquet under `runs/<timestamp>-<confighash>/`. Reports are generated from
those files. No data-versioning layer, no tracking server.

Why: DVC exists to version large binary artifacts that cannot live in git. Our
versioned artifacts are a few hundred JSONL lines that a reviewer genuinely
wants to read as a diff, plus a manifest of hashes. Pointing DVC at them adds a
remote to configure and a cache to invalidate in exchange for nothing. MLflow
exists so that many people running many experiments concurrently get a shared
UI and a queryable store; here, a directory of parquet keyed by config hash is
already queryable with pandas, diffable, and portable in a tarball, with no
server to run and no second source of truth to reconcile against the artifacts.

Cost: no run-comparison UI out of the box, and cross-run queries are pandas
rather than SQL. If this repo grew to several concurrent contributors sweeping
in parallel, that calculus would change and this entry should be superseded.

## D-0008 -- Hydra compose API rather than `@hydra.main`

2026-09-08, M0

Config is composed by `initialize_config_dir` + `compose` inside a
hand-rolled verb dispatcher, not by decorating each entrypoint.

Why: evalgate is a multi-command CLI, and `@hydra.main` takes over `sys.argv`
and creates its own timestamped output directory. We manage run directories
ourselves under `runs/`, keyed by config hash rather than by wall-clock alone,
and a second output-directory convention would be a competing source of truth.

Cost: we lose hydra's logging setup and multirun launcher. `make ablate`
implements its own sweep, which is a few dozen lines and gives control over the
run-directory naming that the gate depends on.

## D-0009 -- What the sandbox could not measure, and how that is handled

2026-09-08, M0

The environment this repo was built in reaches PyPI and GitHub only. EUR-Lex,
DG TAXUD, huggingface.co and download.pytorch.org are refused by the egress
policy, and no Anthropic API credential is available to it. Consequences,
recorded here rather than papered over:

- `data/corpus/manifest.json` as first committed records all six sources as
  `error` with the proxy's refusal. The fetcher is correct; the network is not
  open. Running `make corpus` anywhere with normal internet access fills it in.
- The real corpus could not be downloaded, so the pipeline is exercised against
  a clearly-labelled synthetic corpus (`corpus=cbam_synthetic`, added in M1)
  whose documents carry a "not law" header. Every artifact and report states
  which corpus produced it. Synthetic-corpus numbers are never presented as
  measurements of the real system.
- Generation and judging could not make live calls. The record/replay cassette
  layer required by M5 is therefore also what makes M3 and M4 runnable here:
  committed cassettes drive the pipeline, and the reports say plainly which
  numbers came from replay rather than from a live run.
- Using the host session's own Anthropic credentials to run the eval suite was
  available and deliberately not done: spending a user's session quota on
  hundreds of eval calls is not something to do without being asked.

Cost: the headline numbers in the README are produced under conditions the
README states explicitly, and are weaker than what the same code produces
against the real corpus with a real key. That is the honest trade; the
alternative is a repo whose numbers cannot be traced to how they were made.
