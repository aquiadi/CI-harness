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
expression. `docs/PRINCIPLES.md` forbids parsing judge output with a regex. These are
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

## D-0026 -- An extractive generator ships as the baseline

2026-09-09, M4

`generator=extractive` answers by quoting the sentences of the top-ranked
chunks that overlap the question, with citations attached. No model, no cost,
no network.

Why: it is the floor any generator has to clear. If a model's answers do not
score better than "quote the best-matching sentences", the model is not adding
anything and the ablation should say so rather than leaving the reader to
assume. As with the heuristic judge (D-0021), it also lets the whole
measurement path -- sweep, frontier, gate -- run with no credentials, which is
the same property that makes pull-request CI free.

Cost: its answers are stilted, it cannot synthesise across chunks, and it
cannot decline for the right reason. Every number currently in
`reports/pareto.md` was produced with it, so the absolute quality figures are a
floor, not a claim about the system with an API generator. The relative
ordering of retrieval configurations is what the sweep is measuring, and that
is not affected by which generator sits downstream of it.

## D-0027 -- Serving latency and cost exclude judging

2026-09-09, M4

`latency_s` on an answer row is retrieval plus generation. Judge latency, judge
tokens and judge spend are recorded in their own columns and aggregated
separately as `eval_cost_usd`.

Why: the Pareto frontier is a serving decision. A user waits for retrieval and
generation; they do not wait for the judge, and they do not pay for it. Folding
evaluation cost into cost-per-query would make every configuration look
several times more expensive than it is and would make the frontier depend on
which judge was configured.

Cost: the reported cost per query is not the total cost of running this repo.
The judge is often the larger bill during development, and `eval_cost_usd`
exists so that is visible rather than hidden.

## D-0028 -- Cost is reported twice: measured and projected

2026-09-09, M4

Each run records `cost_usd` (from API usage, and therefore zero when no API
call was made) and `projected_cost_usd` (this run's context and answer tokens
priced at the reference model's rates in `configs/pricing/`). The report keeps
them in separate columns and the frontier is plotted on whichever varies, with
the caption saying which.

Why: with an offline generator the measured cost of every configuration is
zero, which makes a cost frontier meaningless -- yet the configurations do
differ in what they would cost, because k and the chunker change the context
size by a factor of five. Reporting only the measured zero would hide a real
difference; reporting the projection as if it were measured would be a lie.
Reporting both, labelled, is neither.

Cost: two columns to explain, and a reader who skims might take the projection
for a measurement. The column headers and the section text both say
"projected", and the projection is priced at a model that is named.

## D-0029 -- A sweep cell that cannot run is recorded, not fatal

2026-09-09, M4

`make ablate` catches `BackendUnavailableError` per cell -- the error raised
when an optional dependency is missing, such as the cross-encoder reranker
needing `make ml` -- records the cell as unmeasured with the reason, and
continues. Every other exception propagates.

Why: dying at cell 20 of 36 wastes the 19 measurements already made and tells
the user one thing at a time. Recording the skip keeps the report honest about
what was and was not measured; the current `reports/pareto.md` covers 27 of 36
cells, and the nine rerank cells are listed as requiring the ml extra.

Cost: a caught exception is a place where a real problem could hide. It is
narrowed to one exception type that means exactly "an optional dependency is
not installed", and the skipped cells are printed and reported rather than
silently absent.

## D-0030 -- Pull-request CI replays; a nightly job runs live

2026-09-09, M5

All model access goes through one client with three modes. `replay` serves
every call from a committed cassette and errors on a miss; it is the default
and what pull-request CI runs. `record` makes live calls and writes cassettes.
`live` makes live calls and writes nothing, and is what the nightly workflow
runs against its own baseline.

Why: an eval harness is only worth having if it runs before the merge, and a
suite that costs money and minutes per push does not get to run before the
merge for long. Replay makes a full evaluation -- generation, judging, gate --
deterministic, free and immune to somebody else's rate limit, which is what
makes gating every pull request affordable. Live catches what replay cannot:
provider-side model updates, refusal-behaviour changes, latency creep. They
catch different things and neither substitutes for the other.

Cost: cassettes are a second thing to keep current. A prompt or model change
invalidates them and shows up as a cassette miss, which is deliberate -- it is
better to be told the recording is stale than to silently compare against it --
but it does mean `make record` is part of the workflow for those changes, and
the recording diff needs reviewing. Replay also freezes model behaviour: a
regression that only a live model would reveal waits until the nightly run.

## D-0031 -- Missing data fails the gate

2026-09-09, M5

A metric absent from the run, a metric absent from the baseline, or a missing
`baseline.json` all fail. There is no path where the gate passes because it
could not find something to compare.

Why: the state in which a regression is invisible must not be the state in
which the build is green. That is the whole design constraint. A gate that
skips checks it cannot evaluate degrades silently into a gate that checks
nothing, and nobody notices until the quality it was protecting is gone.

Cost: a genuinely new metric fails the gate the first time it appears, until
the baseline is re-frozen. That is one deliberate `make freeze` in a reviewable
diff, which is the correct price.

## D-0032 -- The gate compares the ground, not the configuration

2026-09-09, M5

A run is comparable to the baseline when the corpus hash, the prompt hashes and
the eval-set hashes match. The config hash and the index hash are deliberately
excluded.

Why: a pull request that changes the retriever, the chunker or k changes both
of those, and judging that change is exactly what the gate is for. What must
not change silently is the ground the comparison stands on: the same documents,
the same prompts, the same questions. If any of those differ, the numbers are
not comparable and the gate says so rather than producing a number that looks
like a comparison.

Cost: editing a prompt or adding an eval question blocks the gate until the
baseline is re-frozen. That is intended -- it forces the "is this new baseline
the right one" conversation into a diff -- but it does make prompt iteration a
two-step process.

## D-0033 -- `make eval` selects the frozen configuration

2026-09-09, M5

`configs/experiment/baseline.yaml` composes the configuration that
`baseline.json` was frozen on, and `make eval` selects it. CI runs `make eval`
with no arguments.

Why: the command a developer runs locally has to be the command that gates the
build. Passing the stack as flags in a workflow file puts the real
configuration in a place nobody reads, and lets CI and the laptop drift apart
until someone spends an afternoon on why the numbers differ. The `PROFILE`
variable in the Makefile makes the choice visible and overridable in one place.

Cost: two sources of truth for "what is the frozen configuration" -- the
experiment file and `baseline.json`'s recorded fingerprint -- which could
disagree. The gate detects exactly that disagreement (D-0032) and fails, so it
is a caught error rather than a silent one.

## D-0034 -- The cost check falls back to projected cost

2026-09-09, M5

When the baseline's measured cost per query is zero, the cost check compares
projected cost instead, and the report says which was used.

Why: the frozen baseline runs the offline stack, whose measured cost is
genuinely zero. A percentage rise from zero is undefined, and a check that can
never fail is worse than no check because it looks like coverage. The projected
figure varies with k and the chunker by a factor of five, so it is the quantity
that actually detects "this change tripled the context we send".

Cost: the gate is comparing a projection when it runs offline, which is a
weaker claim than comparing spend. The alternative -- no cost check at all
until someone freezes a live baseline -- is weaker still.

## D-0035 -- The README is generated, not written

2026-09-09, M6

`README.md` is rendered from `README.template.md` by `make readme`, with every
figure pulled from `baseline.json`, the run artifacts and the eval sets. A test
fails the build when the committed README differs from what the current
artifacts render to.

Why: the repository's own rule is that no number is ever typed by hand, and the
README is the single most-read place for a number to go quietly stale -- a
headline figure copied in once and left there through three changes to the
retriever is exactly the kind of small dishonesty that makes a reader stop
trusting the rest. Generating it makes staleness a build failure.

Cost: editing the README means editing a template with `${placeholders}` in it,
and the prose is one step further from the reader. It also means the README
cannot describe anything the artifacts do not contain, which is a constraint
worth having.

## D-0036 -- The serving layer adds nothing to the answer path

2026-09-09, M6

`POST /query` builds the same stack through the same factory functions the
harness measures, and does no caching, query rewriting or reranking of its own.
It returns the answer plus the retrieval trace, the cost and the provenance
hashes.

Why: anything the serving layer added would be untested by the evaluation, and
the Pareto table would quietly stop describing the deployed system. Returning
the trace is not a debugging convenience: when an answer is wrong the first
question is always "what did it retrieve", and an API that cannot answer that
has to be reproduced offline to find out.

Cost: no serving-side latency tricks, and a slightly larger response body. The
index is built at startup rather than lazily, so a container that cannot build
it fails to start instead of failing on traffic -- which is the behaviour you
want, but it does mean a slower cold start.

## D-0037 -- A report label is unique within its table, or it is not a label

2026-09-09, post-M6

`RunSummary.base_label` names what the sweep varies -- chunker, retriever, k --
and `assign_labels` guarantees uniqueness across the runs in one table by
appending the component that actually differs (embedder, then generator, then
judge, then the config hash) to the colliding rows only.

Why: the base label was not unique. Re-running the same sweep with a different
embedder -- which is the first thing anyone does after `make ml` -- produced two
rows reading `fixed/hybrid/k=10` with different numbers and no way to tell
which was which. A row a reader cannot identify is not evidence, which is the
same principle that makes runs immutable and reports refuse incomparable data;
the labelling layer simply had not been held to it.

Qualifying only the colliding rows keeps the common case short: with one
embedder the labels are exactly what they were, and the committed reports are
byte-identical after this change. A qualifier that does not split a group is
skipped rather than appended, so labels grow only as far as they must.

Cost: a label's length now depends on what else is in the table, so the same
run can print as `fixed/dense/k=10` in one report and
`fixed/dense/k=10/hashed` in another. That is the correct trade -- the label's
job is to disambiguate within the table a reader is looking at -- but it means
a label is not a stable identifier across reports. The run id is.

## D-0038 -- Correctness tests carry no deadline

2026-09-09, post-M6

Property tests assert what the code computes, not how fast it computes it, so
they run with `deadline=None`, and expensive constant setup is hoisted out of
the generated-example loop.

Why: the chunking property test rebuilt the whole hydra config inside the
hypothesis loop, at about 170 ms per example against hypothesis's 200 ms
default deadline. It passed alone and failed at random in a loaded suite. A
correctness test that fails because the machine was busy is worse than no test:
it teaches the reader to re-run red builds, which is precisely the habit this
repository exists to prevent. Caching the chunker fixed the cause and took
about 20 seconds off the suite; `deadline=None` removes the class of failure.

Cost: a genuine performance regression in chunking will no longer surface as a
test failure. It was never a reliable signal for that -- the deadline measured
config composition, not chunking -- and latency that matters is measured in the
run record, where p50 and p95 are recorded per query.

## D-0039 -- The Makefile's own wiring is tested

2026-09-09, post-M6

`tests/test_makefile.py` asserts that every pipeline target passes both
`$(PROFILE)` and `$(ARGS)` through to the command it wraps, that PROFILE
defaults to the frozen baseline experiment, and that every target is declared
`.PHONY`.

Why: `make gen-eval PROFILE="+experiment=live"` silently ignored PROFILE and
ran the default stack. The user got a plausible-looking run against the wrong
corpus, embedder and models, and the only clue was the echoed command line.
That is the same defect class the gate exists to prevent -- a flag that appears
to work and does something else -- and it had gone unnoticed because the
Makefile was the one interface with no tests behind it. `label`, `seed`,
`readme` and `corpus` had the same gap; `readme` additionally accepted no
overrides at all, so `scripts/render_readme.py` now takes them.

`record` is exempt by name: it pins `+experiment=live` deliberately, because
its whole purpose is recording cassettes for the live configuration.

Cost: the tests parse the Makefile with regular expressions, which is
brittle -- two of the three failed on their own parsing before they found a
real defect. That is an acceptable price for covering the interface a user
actually types, but it means a future Makefile restructuring may need the
parser updated alongside it.

## D-0040 -- Groq joins Anthropic as a live model vendor

2026-09-09, post-M6

`evalgate.models.groq_client.GroqClient` calls Groq's OpenAI-compatible chat
completions endpoint over httpx, selected by `api.provider: groq`. The
`+experiment=groq` profile wires the vendor, the mode, both models and the
local embedder together so they cannot drift apart.

Why: the judge is the expensive component and the harness is only interesting
once a real model has been through it. Groq's free tier makes the judge
validation, the bias probes and the calibration set reachable at no cost, which
is the difference between this repo reporting measured numbers and reporting
the offline baselines of D-0009. The vendor is a config value, not a fork:
`ModelClient` already required only `complete()`, so the cassette, cache and
recording layers are untouched and a run recorded against either vendor
replays identically.

Two wire-format differences are handled in the client and tested by name.
Tool arguments arrive as a JSON *string* rather than a parsed object, so
`_decode_arguments` decodes and rejects anything that is not a mapping --
returning `None`, never a guess, so the schema retry loop in `judging` sees a
validation failure rather than fabricated scores. And `tool_choice` is forced
to the single named function, so a verdict arrives validated or not at all.
`retry-after` is honoured over exponential backoff and capped at
`backoff_max_s`; rate limits, not latency, are the binding constraint on a
free tier.

`judge.provider` and `generator.provider` now spell "api" -- meaning "an LLM
reached through the model client" -- because "anthropic" named the wrong thing
once a second vendor existed. "anthropic" is retained as an alias rather than
migrated: run records are immutable (invariant 3), and a committed record must
keep loading forever.

Cost: two vendors is two wire formats to keep correct, and Groq retires model
ids on a schedule of its own, so `configs/judge/groq.yaml` will go stale and
fail with a 404 rather than silently. Prices are recorded as 0.0, which is
honest for the free tier but makes the cost axis of the Pareto frontier
degenerate for Groq runs -- a run at zero cost dominates on that axis by
construction, and the frontier should be read on quality and latency alone
until a paid tier gives the cost axis meaning again.

## D-0041 -- The Groq judge is the largest model; the generator is not

2026-09-09, post-M6

`+experiment=groq` judges with `openai/gpt-oss-120b` and generates with
`openai/gpt-oss-20b`, and `probes.contrast_generator` points at
`groq_large`, which is the judge's own model answering the same questions.

Why the split: D-0040 shipped with judge and generator on one model, and a
test asserted they were equal. That was wrong. The self-preference probe
reports the gap between the judge's mean scores across two generators. That
gap is evidence about self-preference only if one of the two arms *is* the
judge; with judge and generator identical there is one arm and the probe
declines to run, and with two arms that are both not the judge the gap is
uninterpretable. The config would have quietly produced the uninterpretable
case. The test now asserts the models differ and that the contrast arm is the
judge's model, so the wiring cannot silently regress to either failure.

Why these ids: `llama-3.3-70b-versatile` from D-0040 was already retired when
first used -- it 404'd by name, which is the intended loud failure. Of what
Groq now serves, `gpt-oss-120b` is the largest general instruct model, and the
judge is the measuring instrument, so it takes the strongest one available.
`groq/compound` is excluded on purpose: it is an agentic system that performs
its own tool calling, and the judge needs a single forced function call whose
schema it controls. The Qwen 27b models are untried here and may well be
better; that is an ablation, not a default.

Cost: two models means two rate-limit budgets on a free tier, and the probe
now costs a second generation pass over the eval set. Model ids will go stale
again -- this is the second time in two entries -- so anything that pins one
carries the curl that lists what a key can actually reach.

## D-0042 -- reasoning_effort is a config value, and the Groq judge is Qwen

2026-09-09, post-M6

`reasoning_effort` is now a field on `ModelRequest`, `JudgeConfig` and
`GeneratorConfig`, sent by the Groq client only when set. `+experiment=groq`
judges with `qwen/qwen3.8-27b` at default effort;
`configs/judge/groq_gptoss.yaml` is the `openai/gpt-oss-120b` arm.

Why: `gpt-oss-120b` would not emit a forced tool call. Groq returned
`tool_use_failed` with the verdict sitting in `failed_generation` as markdown
prose -- the model had answered, just not through the tool. A six-case probe
separated the candidate causes rather than guessing at them: quadrupling
`max_tokens` did not fix it, removing `strict` did not fix it, and
`gpt-oss-20b` failed the same way, so it was neither budget nor schema nor
that one model. Setting `reasoning_effort: low` fixed it, and Qwen complied at
default effort with no special handling.

So the default judge is Qwen despite being smaller. Buying tool compliance by
suppressing deliberation is a poor trade for the one component whose whole job
is to deliberate, and parameter count is not evidence about judge quality.
Which of the two actually judges better is an empirical question this
repository exists to answer -- run both against the human labels and compare
kappa. Neither is validated yet, and this entry is not a claim that one is
better.

`reasoning_effort` enters the cache key only when set. That is a compatibility
requirement, not tidiness: every cassette committed before the field existed
was keyed without it, and including it unconditionally would miss all of them
and turn replay CI red on a change that alters no request. A test pins the
unset key against the literal old shape.

Two related fixes fell out. `call_structured` rebuilt the retry request field
by field, so it silently dropped `reasoning_effort` and every future field --
it now uses `dataclasses.replace`, and a test asserts the retry carries it.
And the Anthropic client raises on `reasoning_effort` rather than ignoring it:
a sampling parameter that silently does nothing makes two runs look comparable
when they are not.

Cost: a vendor-specific knob in a vendor-neutral request object, which the
Anthropic client can only refuse. The alternative -- a per-vendor options bag --
buys generality this repository has no second use for yet.

## D-0043 -- Token budgets are set by the quota, and one 429 is not retryable

2026-09-09, post-M6

`configs/judge/groq.yaml` drops to `max_tokens: 400` and the Groq generators to
700, and `GroqClient` refuses to retry a 429 that says the request itself
cannot fit the quota.

Why the budgets: D-0042 raised the judge to 4096 to give a reasoning model
headroom. That was reasoning about the wrong constraint. Groq charges the
output-tokens-per-minute quota against the `max_tokens` a request *asks for*,
not against what it returns, so on a free tier with a 1000 OTPM limit a single
4096-token request is rejected outright -- `Limit 1000, Requested 1077` -- and
no amount of waiting helps. A measured judgment is about 150 output tokens, so
400 is roughly 2.5x what the work needs while letting several calls through per
minute. The number comes from the quota, and the config says so, because a
future reader will otherwise "fix" it back up to a generous ceiling.

Why the retry change: the client treated every 429 as transient and burned all
eight attempts with exponential backoff -- minutes -- to arrive at the identical
permanent failure, and the error it finally reported was the truncated last
attempt rather than the cause. `is_oversized_for_quota` separates "you are
going too fast", which waiting fixes, from "this request can never fit", which
it does not, and the second fails immediately naming the remedy. Matching on
the message text is unlovely, but the status code alone cannot distinguish the
two cases and silently retrying the wrong one costs minutes per call.

Cost: on this quota the judge runs at a few calls per minute, so a 15-answer
eval set takes several minutes rather than seconds, and the full ablation sweep
is impractical on the free tier. That is a property of the tier, not of the
harness -- the same configuration against a paid key needs only the budget
numbers raised. The marker strings are Groq-specific and will need revisiting
if the wording changes; the tests pin the current phrasing so the failure is a
test, not a silent regression to retrying forever.

## D-0044 -- A 2xx is not a document

2026-09-10, post-M6

`scripts/fetch_corpus.py` validates the response body before writing it: an
empty body or, for a `kind: pdf` source, one not starting with `%PDF-` is
rejected and retried rather than written. `_reuse_existing` applies the same
check to what is already on disk and deletes a file that fails it.

Why: the fetcher treated `response.is_success` as "this is the document". It
is not. EUR-Lex answers 202 with an HTML holding page while it renders a PDF,
and 202 is a success status, so the fetcher wrote the holding page -- or, when
the body was empty, nothing at all -- and recorded it in the manifest as OK
with `sha256 e3b0c442...`, the hash of zero bytes. Nothing downstream could
detect this. The document ingests to no chunks, retrieval returns nothing for
it, and the corpus hash looks exactly as valid as a real one. A corpus that is
missing a document is a condition the pipeline reports and carries on from; a
corpus that contains an empty file measured as real is the "synthetic data
labelled as measured" failure this repository is built to make impossible.

The cache check matters as much as the download check and was the more
dangerous half. `_reuse_existing` only tested that the path existed, so one bad
download was permanent: every later `make corpus` found the zero-byte file,
reported it as cached, and never retried. The bug repaired itself only if
someone deleted the file by hand, which requires already knowing.

A rejected body is retried rather than failed outright, because the common
cause -- EUR-Lex still rendering -- resolves on the next attempt. Only after
the attempt limit does it become an ERROR entry naming the status, the
content-type and the first bytes received, so the failure says what arrived
instead of merely that something did.

Cost: the PDF magic-byte check is format-specific and keyed on `source.kind`,
so a new source kind needs a rule adding. Sniffing content rather than
trusting `content-type` is deliberate -- the servers that send an HTML error
page with a 2xx are the same ones that mislabel it -- but it means a valid PDF
served with a leading byte-order mark would be rejected. No such source exists
here, and a loud rejection is the right failure for the one that does.

## D-0045 -- The README's prose went stale while its numbers stayed correct

2026-09-10, post-M6

`tests/test_readme.py` guarantees every *number* in the README matches the
current artifacts. It cannot check the sentences around them, and after the
first real judge run those sentences were wrong in three places: the headline
called a kappa of 1.000 "bad", the limitations section blamed a rule-based
judge for numbers a real LLM judge had produced, and the corpus limitation said
`corpus=cbam` was waiting on network access that now works.

The template prose now states what a number measures rather than delivering a
verdict on its value, which is what made it go stale: "which is bad" was true
of 0.030 and false of 1.000, while "these labels are not independent" is true
of both. A hand-written adjective about a generated number is a hand-written
number wearing a disguise.

`tests/test_ingest.py::test_manifest_with_no_fetched_documents_is_refused` also
had to change, for the same underlying reason. It asserted on the *committed*
manifest, whose emptiness was a property of the build sandbox rather than of
the code. Once a real corpus was fetched, the test failed for a reason
unrelated to what it asserts. It now builds its own manifest, and a second test
covers the other half -- a manifest promising a file that is not on disk, which
is what a fresh clone hits, since the manifest is committed and the PDFs are
gitignored.

Cost: the README banner now has to distinguish the ablation frontier (still
synthetic corpus, rule-based judge, extractive generator) from the calibration
section (real LLM judge), because those two sets of numbers no longer describe
one configuration. That is a wordier banner, and it will need splitting again
as the frontier catches up. The alternative -- one blanket disclaimer covering
both -- would be false about whichever half moved first, and invariant 4 says
a number without its fingerprint is not a result.

## D-0046 -- The sweep is resumable, because it has to be

2026-09-10, post-M6

`ablate` now looks for an existing run with the same config hash and corpus
hash *before* calling `evaluate()`, and stops cleanly on a `ModelError` with
instructions to re-run.

Why: a Groq free-tier daily token limit stopped a 36-cell sweep at cell 3.
Two cells consumed roughly 197,000 of the 200,000 daily tokens, which puts the
whole sweep at about two cells per day, or eighteen days. That is only
survivable if the sweep resumes, and it did not.

`ablate` did have an "already measured, skipping" branch, but it was
unreachable. It caught `RunExistsError` from `write_run`, which fires when a
run directory already exists -- and a run id is `{timestamp}-{config_hash}`, so
two runs of one configuration never collide. The branch had never once
executed. Worse, it sat *after* `evaluate()`, so even had it fired, the API
calls were already spent and the result was discarded. Re-running a stopped
sweep re-paid for every completed cell from the beginning, which on a daily
quota means day two re-buys day one and never advances.

The new check keys on config hash *and* corpus hash: the same configuration
against a different corpus is a different measurement and must be re-run.
It sits after `build_stack`, which is free, and before `evaluate`, which is not.

A `ModelError` now ends the sweep rather than propagating as a traceback.
Continuing would be worse than stopping: a quota failure hits every remaining
cell identically, so the sweep would fill with holes and report itself
complete. It stops, says which cell it reached, and says that re-running skips
what is measured without re-paying.

Cost: `find_measured` reads every run's meta.json on every cell, which is
linear in the number of runs and will get slow well past a thousand of them.
An index would fix that and is not worth building yet. The deeper cost is that
a resumable sweep spans days, so its cells can straddle a change to the corpus
or a prompt -- the hashes are what catch that, and the reporting layer already
refuses to place runs with different fingerprints in one table.

## D-0047 -- The constitution moves to docs/PRINCIPLES.md

2026-09-10, post-M6

`CLAUDE.md` is now `docs/PRINCIPLES.md`. The content is unchanged: it is the
repository's constitution and every invariant in it still holds.

Why: the filename was a convention of one particular authoring tool, not a
property of the project. A file whose name announces which assistant read it
says nothing about the codebase, and this repository is the author's work
whoever or whatever typed it. `docs/PRINCIPLES.md` says what the file is.

Cost: the old name is what some tools look for automatically, so that pickup is
lost and a contributor has to be pointed at the file by the README instead.
That is the correct trade -- the document is for people, and a convention that
only one tool honours is a poor reason to keep a name that misdescribes the
file to everyone else.

## D-0048 -- A query interface, and the two switches a public deployment needs

2026-09-10, post-M6

`make serve` now serves a query UI at `/` alongside the existing API, and
`EVALGATE_API_KEY` and `EVALGATE_RATE_LIMIT_PER_MINUTE` gate `POST /query`.

Why a UI: the API already returned everything needed to check an answer -- the
citations with a validity flag, the retrieval trace, the cost and the hashes --
and nobody was going to read it as JSON. The interface renders each
`[chunk#id]` marker as a button that scrolls to the passage, and labels each
citation verified or unsupported using the same audit the evaluation uses. That
makes the repository's central claim visible in the product rather than only in
a report: an answer over a regulation is worth nothing if the reader cannot
check it against the text. `RetrievedChunk` gained a `text` field for this; a
citation nobody can read is not a citation.

The UI is a client of `/query` and adds nothing to the answer path, for the
reason the serving module already gives: a serving layer that reranks or
rewrites queries is a different system from the one the Pareto table measures,
and the table would quietly stop being true.

Why the switches are environment variables and not config: they describe the
deployment, not the measurement. In the hydra tree a rotated key would change
the config hash, which would sever the comparability of every run recorded
either side of the rotation -- a security operation must not invalidate a
measurement.

`/health` is exempt from both. A load balancer carries no key, and a readiness
probe that trips the rate limiter takes the service down exactly when it is
busiest.

Cost: the limiter is a fixed window in process memory. It resets on restart and
does not coordinate between replicas, so it stops a runaway script and not a
determined adversary, and the tests say so by name rather than implying more.
Shared-state limiting belongs in front of the app. The UI is three static files
with no build step and no framework, which keeps the image small and the
dependency surface at zero but means it stays deliberately plain.
