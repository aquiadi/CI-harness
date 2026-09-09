"""Configuration schema and loader.

All configuration lives in ``configs/`` as YAML. This module declares the
dataclass schema hydra validates against, composes the config, and derives the
fingerprint that identifies a run.

Pluggable components (chunker, embedder, retriever) are declared with
``_target_`` and instantiated by hydra, so adding a strategy means adding a
YAML file and a class -- never editing a dispatch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from evalgate.hashing import hash_obj
from evalgate.rootdir import config_dir, find_repo_root

SCHEMA_NAME = "evalgate_schema"


@dataclass
class PathsConfig:
    """Filesystem layout. Every path in the codebase is resolved from here."""

    root: str = MISSING
    data_dir: str = MISSING
    corpus_raw_dir: str = MISSING
    corpus_processed_dir: str = MISSING
    corpus_manifest_path: str = MISSING
    index_dir: str = MISSING
    eval_dir: str = MISSING
    runs_dir: str = MISSING
    reports_dir: str = MISSING
    cache_dir: str = MISSING
    prompts_dir: str = MISSING
    cassette_dir: str = MISSING
    baseline_path: str = MISSING


@dataclass
class SourceDocument:
    """One pinned source document in the corpus."""

    id: str = MISSING
    title: str = MISSING
    url: str = MISSING
    filename: str = MISSING
    kind: str = "pdf"
    required: bool = True


@dataclass
class CorpusConfig:
    """The pinned document set."""

    name: str = MISSING
    sources: list[SourceDocument] = field(default_factory=list)
    # When set, documents are read from this directory instead of the
    # downloaded PDFs. Used for corpora that are not fetched from a URL.
    local_dir: str | None = None
    # Filenames to skip when loading from local_dir (a corpus directory may
    # legitimately contain a README that is not part of the corpus).
    exclude: list[str] = field(default_factory=list)
    request_timeout_s: float = 60.0
    max_attempts: int = 3
    refetch: bool = False
    user_agent: str = "evalgate-corpus-fetcher/0.1 (+https://github.com/aquiadi/CI-harness)"


@dataclass
class SparseConfig:
    """BM25 index-time parameters."""

    name: str = MISSING
    k1: float = 1.2
    b: float = 0.75
    stopwords: str = "en"
    stemmer: str = "none"


@dataclass
class GeneratorConfig:
    """The answering model."""

    provider: str = MISSING
    model: str = MISSING
    prompt: str = MISSING
    max_tokens: int = 1024
    temperature: float = 0.0
    input_usd_per_mtok: float = MISSING
    output_usd_per_mtok: float = MISSING
    # Extractive generator only.
    max_sentences: int = 3
    max_chunks: int = 3
    min_overlap: int = 2


@dataclass
class PricingConfig:
    """Reference rates for projecting what a configuration would cost."""

    model: str = MISSING
    input_usd_per_mtok: float = MISSING
    output_usd_per_mtok: float = MISSING


@dataclass
class JudgeConfig:
    """The judging model. Deliberately separate from the generator."""

    provider: str = MISSING
    model: str = MISSING
    prompt: str = MISSING
    max_tokens: int = 1024
    temperature: float = 0.0
    max_attempts: int = 3
    axes: list[str] = field(default_factory=list)
    scale_min: int = 1
    scale_max: int = 5
    input_usd_per_mtok: float = MISSING
    output_usd_per_mtok: float = MISSING
    # Which answers to grade: the eval set's reference answers, or the answers
    # produced by a run under runs/.
    answers_from: str = "reference"
    run_id: str | None = None


@dataclass
class ProbesConfig:
    """Judge bias probes."""

    position_swap: bool = True
    length_bias: bool = True
    self_preference: bool = True
    # The second generator used for the self-preference probe, as a config
    # group name under configs/generator/.
    contrast_generator: str = "haiku"
    worst_disagreements: int = 10


@dataclass
class ApiConfig:
    """How model calls reach (or do not reach) the network."""

    mode: str = "replay"
    cache_enabled: bool = True
    cache_path: str = MISSING
    max_retries: int = 3
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 30.0
    timeout_s: float = 120.0


@dataclass
class GateConfig:
    """Regression thresholds. Not one of these numbers appears in Python."""

    baseline_path: str = MISSING
    quality_drop_pct: float = 2.0
    p95_latency_rise_pct: float = 20.0
    cost_per_query_rise_pct: float = 15.0
    composite_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalSetsConfig:
    """Eval data locations and sampling."""

    retrieval_path: str = MISSING
    answers_path: str = MISSING
    human_labels_path: str = MISSING
    seed_labels_path: str = MISSING
    judge_scores_path: str = MISSING
    limit: int | None = None


@dataclass
class EvalGenConfig:
    """Model-drafted retrieval candidates, for human review."""

    prompt: str = MISSING
    model: str = MISSING
    max_tokens: int = 2048
    temperature: float = 0.0
    max_attempts: int = 3
    target_slots: int = 120
    sample_chunks: int = 80
    max_questions_per_chunk: int = 2
    min_chunk_tokens: int = 60
    input_usd_per_mtok: float = MISSING
    output_usd_per_mtok: float = MISSING


@dataclass
class LabelConfig:
    """The terminal labelling CLI."""

    mode: str = "answers"
    labeller: str = MISSING
    answers_from: str = "reference"
    run_id: str | None = None
    context_k: int = 5
    show_gold: bool = False
    relabel: bool = False


@dataclass
class AblationConfig:
    """The sweep matrix for `make ablate`."""

    chunkers: list[str] = field(default_factory=list)
    retrievers: list[str] = field(default_factory=list)
    k_values: list[int] = field(default_factory=list)
    rerank: list[bool] = field(default_factory=list)


@dataclass
class RootConfig:
    """Top-level config node."""

    seed: int = 0
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    tokenizer: Any = MISSING
    chunker: Any = MISSING
    embedder: Any = MISSING
    sparse: SparseConfig = field(default_factory=SparseConfig)
    retriever: Any = MISSING
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    probes: ProbesConfig = field(default_factory=ProbesConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    evalsets: EvalSetsConfig = field(default_factory=EvalSetsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    evalgen: EvalGenConfig = field(default_factory=EvalGenConfig)


def register_schema() -> None:
    """Register the dataclass schema so hydra type-checks the composed config."""
    store = ConfigStore.instance()
    store.store(name=SCHEMA_NAME, node=RootConfig)


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("repo_root"):
        OmegaConf.register_new_resolver("repo_root", lambda: str(find_repo_root()))


def load_config(
    overrides: list[str] | None = None,
    config_name: str = "config",
) -> DictConfig:
    """Compose the config tree, applying dotted-path overrides.

    The compose API is used rather than ``@hydra.main`` because evalgate is a
    multi-command CLI and manages its own output directories under ``runs/``.
    """
    register_schema()
    _register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"composed config is not a DictConfig: {type(cfg).__name__}")
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Resolve interpolations and return a plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))


def typed_node[T](cfg: DictConfig, dotted: str, kind: type[T]) -> T:
    """Convert a config node into its dataclass, failing loudly on a mismatch."""
    node = OmegaConf.select(cfg, dotted, throw_on_missing=True)
    if node is None:
        raise KeyError(f"config key not found: {dotted}")
    obj = OmegaConf.to_object(node)
    if not isinstance(obj, kind):
        raise TypeError(f"config key {dotted!r} is not a {kind.__name__}: {type(obj).__name__}")
    return obj


def resolve_path(cfg: DictConfig, dotted: str) -> Path:
    """Resolve a dotted config key to an absolute filesystem path."""
    value = OmegaConf.select(cfg, dotted, throw_on_missing=True)
    if not isinstance(value, str):
        raise TypeError(f"config key {dotted!r} is not a path string: {value!r}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (find_repo_root() / path).resolve()


# Keys that change the meaning of a measurement. `evalsets` is deliberately
# absent: it holds file paths, and where the eval set lives cannot change a
# result. What the eval set CONTAINS certainly can, so its content hash is
# recorded in the run record instead. Everything outside this set --
# where files live, whether the API is replayed, how many workers run -- must
# not change results, and is excluded so that runs stay comparable across
# machines. If a run's fingerprint differs, its numbers are not comparable.
FINGERPRINT_KEYS = (
    "seed",
    "corpus",
    "tokenizer",
    "chunker",
    "embedder",
    "sparse",
    "retriever",
    "generator",
    "judge",
)


def fingerprint(cfg: DictConfig) -> dict[str, Any]:
    """Return the result-affecting subset of the config."""
    resolved = to_container(cfg)
    subset = {key: resolved[key] for key in FINGERPRINT_KEYS if key in resolved}
    corpus = subset.get("corpus")
    if isinstance(corpus, dict):
        # Fetch mechanics do not affect results; the document set does.
        subset["corpus"] = {
            "name": corpus.get("name"),
            "sources": corpus.get("sources"),
        }
    return subset


def config_hash(cfg: DictConfig) -> str:
    """Hash of the result-affecting config subset."""
    return hash_obj(fingerprint(cfg))
