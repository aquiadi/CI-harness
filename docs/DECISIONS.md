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

## D-0010 -- Chunk sizes are measured with a scaled regex word count

2026-09-09, M1

Chunk boundaries are decided by counting regex word atoms and multiplying by a
configured `tokens_per_word` (1.3 for English legal prose), not by the
generation model's real tokenizer.

Why: the real BPE tokenizer is behind an API call. Counting tokens for every
candidate chunk boundary would cost money per ablation and make chunking depend
on the network. What the ablations require is that "512 tokens" means the same
thing in every run and on every machine, which a deterministic local estimate
delivers and an API call does not.

Cost: the estimate is off by a roughly constant factor, so a chunk labelled 512
tokens is not exactly 512 to Anthropic's tokenizer. That matters for context
budgeting, so billed tokens are always read from API usage and never estimated.
The ratio is config, so recalibrating it is a config change and a new config
hash, which correctly marks old runs as incomparable.

## D-0011 -- Exact brute-force vector search, no ANN index

2026-09-09, M1

LanceDB is used as a file-based store and searched exhaustively. No IVF/PQ
index is created.

Why: this corpus is thousands of chunks, where an exact scan costs
milliseconds. An approximate index would buy nothing measurable here and would
introduce recall that varies with index build parameters and insertion order --
a class of "the numbers moved and nobody changed anything" bug that is
expensive to diagnose and fatal to a repo whose product is trustworthy
measurement.

Cost: this does not scale. At a million chunks the scan dominates latency and
an ANN index becomes necessary; at that point the honest move is to measure
the recall the approximation costs and record it, not to adopt it silently.

## D-0012 -- Hybrid retrieval fuses ranks (RRF), not scores

2026-09-09, M1

The hybrid retriever combines the dense and sparse lists with reciprocal rank
fusion rather than a weighted sum of normalised scores.

Why: cosine similarity and BM25 are not on a comparable scale, and normalising
them (min-max over what window? z-score over which corpus?) introduces a
parameter that has to be re-tuned per corpus and defended per result. RRF uses
only ranks: one parameter with a well-understood effect, and no way for one
retriever's score distribution to dominate the fusion.

Cost: RRF discards score magnitude, so a case where the dense retriever is
extremely confident and the sparse one is weakly ranked is fused the same as
one where both are marginal. Where magnitude matters, the cross-encoder rerank
stage is the place to recover it, and the ablations measure whether it does.

## D-0013 -- BM25 parameters live in their own config group

2026-09-09, M1

`k1`, `b`, stopwords and stemmer are in `configs/sparse/`, not in
`configs/retriever/bm25.yaml`.

Why: bm25s precomputes term scores at index time, so these are properties of
the index, not of the query. Keeping them under the retriever would mean the
dense and sparse retrievers disagreed about which index they shared, and the
ablation sweep would rebuild an identical index once per retriever.

Cost: one more config group, and a slightly surprising place to look for a
BM25 knob. The comment in the group file says why it lives there.

## D-0014 -- A query with no signal returns k results, not zero

2026-09-09, M1

If a query embeds to a zero vector (no word characters at all), dense search
scores every chunk zero and falls through to the shared tie-break, returning
k results in chunk-id order -- matching what BM25 already does for an
out-of-vocabulary query.

Why: this was found by a hypothesis property test, not by design. LanceDB drops
the undefined cosine distances and returns nothing, so the same query produced
0 results from the dense retriever and k from the sparse one. Whatever the
right answer is, retrievers disagreeing about the shape of their output is not
it: downstream code would have to special-case one backend. Uniform behaviour
with an explicit, tested rule beats an accident of a library's NaN handling.

Cost: a caller cannot distinguish "no signal" from "genuinely weak matches" by
result count alone; they must look at the scores, which are exactly zero in the
degenerate case. Real eval questions never hit this path.

## D-0015 -- Gold evidence is a verbatim span, not a chunk id

2026-09-09, M2

A retrieval eval slot stores `gold_spans`: sentences copied verbatim from the
source document. A retrieved chunk counts as a hit when it contains one, with
whitespace collapsed on both sides. `gold_chunk_ids` is still recorded, for the
reference chunking named alongside it, because it is what the human reviewer
looked at -- but no metric reads it.

Why: chunk ids are a property of the chunker (`{doc_id}#{ordinal}`), and the
entire ablation matrix varies the chunker. An eval set keyed on one chunker's
ids can score exactly one configuration, which would make the eval set useless
for the thing it exists to support. A span is a property of the corpus, so one
eval set scores every configuration.

Cost: a span that straddles a chunk boundary is a miss for every chunker that
splits it, which is harsher than a human would be. That is deliberate -- a
chunk holding half an obligation is a real defect -- but it means recall is
sensitive to chunk size in a way that a fuzzier matcher would hide. Matching is
also case-sensitive, because in regulatory text "Article" and "article" are
different things.

## D-0016 -- A synthetic corpus, labelled as synthetic everywhere

2026-09-09, M2

`data/corpus/synthetic/` holds five documents written for this repository in
the structural style of the CBAM instruments, and `corpus=cbam_synthetic` is
the default. Every file opens with "SYNTHETIC DOCUMENT - NOT LAW", the
directory carries a README saying the same, and every run record and report
names the corpus that produced it.

Why: the build environment cannot reach EUR-Lex (D-0009), and a harness with
nothing to measure demonstrates nothing. Making it the default is the choice
that lets a fresh clone run `make index` with no network and no credentials,
which is what a reproduction guide has to be able to promise. `corpus=cbam`
selects the real documents once `make corpus` has fetched them.

Cost: the headline numbers are measured over text this repository wrote, which
is easier than the real instruments in ways that are hard to quantify -- the
vocabulary is smaller, the cross-references are shallower, and no sentence is
400 words long. The numbers are therefore optimistic relative to the real
corpus, and the README says so rather than implying otherwise. The mitigation
is that nothing about the harness depends on which corpus it runs over: point
it at the real one and every number recomputes.

## D-0017 -- The labelling CLI uses rich, not textual

2026-09-09, M2

`make label` is a prompt-and-print loop built on `rich`, not a full-screen
`textual` application.

Why: the requirement is that a labelling session never loses work. A
line-oriented loop that appends and fsyncs after every rating satisfies that by
construction, works over SSH and in a CI log, and is testable by feeding it a
list of keystrokes -- which is how the resumption and interruption behaviour is
actually tested here. A full-screen app would look better and would put the
guarantee behind a widget event loop.

Cost: no mouse, no split panes, no live progress bar. For an hour of rating 1-5
scores, the loop is not the bottleneck.

One thing this cost us and is worth writing down: rich interprets square
brackets as markup, so the first version silently deleted every `[chunk#id]`
citation from the answer being rated -- the exact text the citation-correctness
axis is about. Answer, question and context are now rendered as `Text`, which
does not interpret markup.

## D-0018 -- Record/replay arrived in M2, not M5

2026-09-09, M2

The cassette layer was built when retrieval-candidate generation needed a model
call, two milestones before the regression gate that motivated it.

Why: the alternative was a temporary stub client that would have been deleted
in M5, and a stub is a second code path that the tests exercise and production
does not. Building the real thing once means the eval loop, the judge, the
ablation sweep and the gate all share one client stack, and that stack is
exercised from the first model call in the repo.

Cost: M2 depends on machinery whose reason for existing only becomes obvious in
M5, so the commit ordering reads oddly in `git log`. This entry is the
explanation.

## D-0019 -- Seed labels are attributed and are not independent

2026-09-09, M2

`data/eval/seed_labels.jsonl` carries a rating for each of the 15 seed answers,
attributed to `seed-author` with a note recording the flaw each answer was
written to contain. The seed answers are deliberately varied: correct answers,
an unsupported number, a wrong citation, a missing citation, an overclaim, an
off-topic answer, and a correct refusal.

Why: judge calibration needs labelled data to be runnable at all, and a judge
validated only against good answers has not been validated, because it never
had to detect anything.

Cost, stated plainly: the same author wrote the answer and its score, so these
are not independent human labels, and agreement measured against them says
almost nothing about whether the judge agrees with a person. They exist to make
the calibration pipeline runnable and testable before anyone has labelled.
`reports/judge_calibration.md` states this wherever they are the source, and
the report distinguishes them from labels produced through `make label`.

## D-0020 -- The judgment schema is built from config at runtime

2026-09-09, M3

The pydantic model the judge's tool call is validated against is constructed by
`build_judgment_model(axes, scale_min, scale_max)` from the values in
`configs/judge/`, rather than written out with a fixed 1-5 bound and three
named fields.

Why: the axes and the scale are configuration, and the invariant is that no
threshold or scale appears in Python. Generating the model means the tool
schema the API validates against, the bounds validation applies, and the
columns the report prints all follow from one place. Adding a fourth axis is a
config change.

Cost: the model is dynamic, so a type checker cannot see its fields, and code
reading a judgment goes through `model_dump()` rather than attribute access.
That is a small, contained loss confined to two functions.

## D-0021 -- A rule-based baseline judge ships alongside the LLM judge

2026-09-09, M3

`judge=heuristic` scores the same three axes by rule: content-word coverage
against the retrieved context for groundedness, question-answer overlap with
explicit refusal handling for relevance, and citation validity for citation
correctness.

Why: the first question a sceptical reader asks about an LLM judge is whether
it beats a cheap rule. Without a baseline in the calibration report there is no
way to answer, and a judge that agrees with people no better than string
matching does is not worth its latency or its bill. The baseline is also immune
by construction to the position and self-preference biases the probes look for,
which makes it a useful control when reading those sections.

It has a second effect that was not the reason for building it but is worth
recording: because it needs no credentials, `make report` produces a real
calibration report on a fresh clone with no API key. That is how the numbers
currently in `reports/judge_calibration.md` were produced, and the report says
which judge produced them.

Cost: a second scoring implementation to keep aligned with the axes. It is
about ninety lines and its rules are documented in full in its module
docstring, because a baseline whose behaviour is unclear is not a baseline.

## D-0022 -- Judge scores and human labels join on the answer hash

2026-09-09, M3

A judge score is compared to a human label only when both carry the same
`example_id` **and** the same `answer_hash`.

Why: an answer changes whenever the generator, the prompt or the retrieval
changes. Joining on the question alone would compare a judge's opinion of one
answer with a person's opinion of a different answer to the same question, and
report the result as agreement. That is a number that looks right and is
meaningless, which is the worst kind.

Cost: re-generating answers invalidates every existing label, and the report
then shows "no overlapping labelled items" rather than a stale kappa. That is
the correct behaviour and it is tested, but it does mean human labelling effort
is tied to a specific answer set.

## D-0023 -- Reports carry no timestamp

2026-09-09, M3

Generated reports record an inputs digest -- a hash of the judgments and labels
they were computed from -- instead of a generation time.

Why: a report is a pure function of the artifacts it reads. Regenerating one
whose inputs have not changed should produce a byte-identical file and leave
the working tree clean, so that a diff in `reports/` always means the
measurements moved. A timestamp would make every regeneration a diff and train
readers to ignore them.

Cost: "when was this produced" is not answerable from the file itself. It is
answerable from `git log`, which is the better record anyway.

## D-0024 -- Undefined kappa is reported as undefined, never as zero

2026-09-09, M3

When either rater gives the same score to every item, Cohen's kappa divides by
zero. The report prints "undefined" with the reason instead of a number.

Why: 0.0 means "agreement no better than chance", which is a finding. Undefined
means "this data cannot answer the question", which is a different finding.
Printing the first when the second is true is a quiet lie, and this repo's only
real asset is that its numbers can be trusted. The seed label set is skewed
enough to hit this case, so it is not hypothetical.

Cost: consumers of the report have to handle a non-numeric cell. The gate
handles it by failing rather than by coercing (see M5).

## D-0025 -- Citations are extracted with a regex; judge output never is

2026-09-09, M3

`evalgate.citations` finds `[doc#0007]` in a generated answer with a regular
expression. `CLAUDE.md` forbids parsing judge output with a regex. These are
not in tension, and the module docstring says why.

The rule exists because a half-matching regex over a judge's prose invents a
score that looks real and cannot be detected downstream. Citation extraction is
the opposite situation: the input is an answer, the output is a set of chunk
ids, and every extracted id is then checked against the context that was
actually retrieved. A missed or spurious match surfaces as a citation the
context does not contain, which is measured. Judge scores still arrive only
through a tool call validated against a schema.

Cost: an answer that writes a citation in some other format is scored as
uncited. That is a defensible reading -- the generation prompt specifies the
format -- but it does mean the citation axis is partly measuring format
compliance. The heuristic judge's citation score in particular should be read
that way.
