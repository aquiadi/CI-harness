# The Makefile is the only interface. Every target is a thin wrapper over a
# hydra entrypoint; no behaviour lives here beyond argument plumbing.

UV ?= uv
RUN := $(UV) run
PY := $(RUN) python
ARGS ?=

# Which stack the pipeline targets run. The default is the configuration frozen
# in baseline.json: the offline stack, so every target works on a fresh clone
# with no network and no API key. `PROFILE="+experiment=live"` selects the real
# system (real documents, bge embeddings, API generator and judge).
PROFILE ?= +experiment=baseline

.DEFAULT_GOAL := help
.PHONY: help install dev ml test check fmt lint types config seed gen-eval judge freeze record corpus ingest index \
        eval ablate label readme report gate-demo serve docker clean

help:  ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "};{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the venv and install runtime dependencies
	$(UV) sync --no-dev

dev:  ## Install runtime + dev dependencies (what CI uses)
	$(UV) sync

ml:  ## Add the local-model extra (sentence-transformers, torch)
	$(UV) sync --extra ml

fmt:  ## Format and apply safe lint fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint:  ## Lint without writing
	$(RUN) ruff format --check .
	$(RUN) ruff check .

types:  ## Type-check (mypy strict)
	$(RUN) mypy

test:  ## Run the test suite, excluding slow and live tests
	$(RUN) pytest -m "not slow and not live"

check: lint types test  ## The pre-commit gate. Never commit red.

config:  ## Print the resolved config plus the hashes it implies
	$(PY) -m evalgate.cli config $(PROFILE) $(ARGS)

gen-eval:  ## Draft retrieval eval candidates with a model, for review
	$(PY) -m evalgate.cli gen-eval $(ARGS)

seed:  ## Materialise the hand-written seed eval sets into data/eval/
	$(PY) scripts/seed_evalsets.py $(ARGS)

record:  ## Re-record the API cassettes that replay-mode CI plays back
	$(PY) -m evalgate.cli eval +experiment=live api=record $(ARGS)

corpus:  ## Download pinned source documents and write the SHA256 manifest
	$(PY) scripts/fetch_corpus.py $(ARGS)

ingest:  ## PDF -> text -> chunks
	$(PY) -m evalgate.cli ingest $(PROFILE) $(ARGS)

index:  ## Build the vector + sparse indexes (no-op if the corpus is unchanged)
	$(PY) -m evalgate.cli index $(PROFILE) $(ARGS)

judge:  ## Score the answer eval set with the configured judge
	$(PY) -m evalgate.cli judge $(PROFILE) $(ARGS)

freeze:  ## Measure the current config and write baseline.json
	$(PY) -m evalgate.cli freeze $(PROFILE) $(ARGS)

eval:  ## Run the eval suite against baseline.json; nonzero exit on regression
	$(PY) -m evalgate.cli eval $(PROFILE) $(ARGS)

ablate:  ## Sweep the config matrix, one immutable parquet per run
	$(PY) -m evalgate.cli ablate $(PROFILE) $(ARGS)

label:  ## Terminal labelling CLI (resumable, saves incrementally)
	$(PY) -m evalgate.cli label $(ARGS)

readme:  ## Regenerate README.md from README.template.md and the artifacts
	$(PY) scripts/render_readme.py

report:  ## Regenerate reports/ from the run artifacts
	$(PY) -m evalgate.cli report $(PROFILE) $(ARGS)

gate-demo:  ## Prove the gate fails on a deliberately degraded retriever
	$(PY) -m evalgate.cli gate-demo $(PROFILE) $(ARGS)

serve:  ## Run the FastAPI layer
	$(RUN) uvicorn evalgate.serve.app:app --host 0.0.0.0 --port 8000

docker:  ## Build the serving image
	docker build -t evalgate:local .

clean:  ## Remove caches and build detritus
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
