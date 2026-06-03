"""
Experiment runner — orchestrates benchmark evaluation across all techniques.

Runs each technique on each task category, collects ReliabilityRun records,
and saves results for analysis and plotting.

Module size note
----------------
At ~1500 LOC this module bundles a few orchestration concerns: parallel
task execution, per-technique cache loading, live plotting, and the
SIGALRM watchdog. A clean split lives on the v0.4 roadmap:

    runner/core.py             # BenchmarkRunner + ExperimentConfig
    runner/parallel.py         # ThreadPoolExecutor orchestration
    runner/cache.py            # cache.json read/write
    runner/live_plots.py       # soft-imported plotting

Deliberately deferred until after the first public release; the
benchmark CLI is the primary consumer and the surface is paper-stable.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .channel import AgentChannel, QualityScorer
from .models import (
    CombiningStrategy,
    HARQMode,
    ReliabilityRun,
    TaskCategory,
    TaskItem,
)
from .techniques import (
    ACMLearnedRouter,
    ACMRouter,
    BestOfNBaseline,
    ChainOfVerificationBaseline,
    CISCBaseline,
    DiversityEnsemble,
    DiversityMRCDiscreteN,
    FECService,
    FountainDecoder,
    HARQService,
    MixtureOfAgentsBaseline,
    SelectionCombiningN,
    SelfConsistencyBaseline,
    SelfRefineBaseline,
    SoftACMRouter,
    SoftDiversityMRC,
    SoftDiversityMRCDiscreteN,
    SoftFountainDecoder,
    TurboDecoder,
    WeightedBoNBaseline,
)

logger = logging.getLogger(__name__)


@contextmanager
def _task_timeout(seconds: int, name: str = "task"):
    """Raise TimeoutError if the wrapped block runs longer than `seconds`.

    Uses SIGALRM to interrupt CPU-bound hangs (e.g. catastrophic regex
    backtracking) AND blocking I/O hangs (e.g. dead Ollama sockets) cleanly.

    SIGALRM is only safe on the main thread of the main interpreter — Python
    raises ``ValueError: signal only works in main thread`` otherwise. This
    context manager **degrades to a no-op off the main thread** instead of
    crashing, because BenchmarkRunner is occasionally embedded in a worker
    thread (notebooks, FastAPI dev runs). For library-mode timeout
    enforcement, see ``ReliabilityModule`` which uses wall-clock checks that
    are thread-safe by construction.

    No-op on:
      * platforms without SIGALRM (e.g. Windows)
      * non-main threads (logged at DEBUG once per call site)
      * ``seconds <= 0``
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        logger.debug(
            "_task_timeout(%s, %ds) skipped: not on main thread; "
            "use the library facade (ReliabilityModule) for thread-safe "
            "timeouts.",
            name, seconds,
        )
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"{name} exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ExperimentConfig:
    """Configuration for a full benchmark run."""

    def __init__(
        self,
        # Models to use
        models: list[dict[str, Any]] | None = None,
        # Judge model for quality scoring
        judge_model: str = "gpt-4o-mini",
        # Which techniques to run
        techniques: list[str] | None = None,
        # Which task categories
        categories: list[str] | None = None,
        # Output directory
        output_dir: str = "results",
        # Global settings
        base_url: str | None = None,
        api_key: str | None = None,
        # Judge-specific settings (fall back to global if not set)
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        # Extra request-body kwargs for the judge (parallels per-model `extra_body`).
        # Use to enable/disable thinking on the judge, e.g. for Ollama qwen3:
        #   judge_extra_body: {chat_template_kwargs: {enable_thinking: false}}
        # or for Anthropic extended thinking:
        #   judge_extra_body: {thinking: {type: enabled, budget_tokens: 4096}}
        judge_extra_body: dict | None = None,
        # How QualityScorer combines the LLM judge with deterministic
        # checks on tasks that carry a non-None score_mode (set by the
        # benchmark loaders for structured-reference benchmarks like MMLU,
        # GSM8K). One of:
        #   "blended" (default): final = 0.6 * deterministic + 0.4 * judge.
        #     Treats the regex/numeric check as ground truth on correctness
        #     and the judge as a reasoning-quality signal on the prose. This
        #     diverges from the paper's pure-judge scoring on MC/numeric
        #     benchmarks — use "judge" below to reproduce paper numbers.
        #   "exact": pure deterministic, no judge call. Returns {0, 1}.
        #     Cheapest and noise-free; loses the reasoning-quality signal.
        #   "judge": pure judge, ignores score_mode. Matches paper behavior.
        # Tasks WITHOUT a score_mode (free-form like HumanEval, creative)
        # always go through the pure judge regardless of strategy — there's
        # no deterministic check to blend with.
        score_strategy: str = "blended",
        # Difficulty filter
        min_difficulty: str | None = None,
        # ACM routing table
        acm_table: list[dict[str, Any]] | None = None,
        # Optional per-category ACM routing tables. Shape:
        # {category_name: [ {difficulty_range, technique, ...}, ... ], ...}
        # When present, ACM routes per (category, difficulty) instead of
        # difficulty alone. Unknown categories fall back to acm_table.
        acm_category_tables: dict[str, list[dict[str, Any]]] | None = None,
        # acm_learned: path to a JSON file produced by the upstream ACM
        # router trainer (not shipped in the open-source release).
        # Required to run the "acm_learned" technique; ignored otherwise.
        # Per-technique dispatch parameters (rounds, branches, code rate)
        # come from `acm_learned_dispatch_defaults` or fall back to
        # agentcodec.techniques.acm_learned.DEFAULT_DISPATCH_PARAMS.
        acm_learned_weights: str | None = None,
        acm_learned_dispatch_defaults: dict[str, dict[str, Any]] | None = None,
        # Critic model for iterative techniques (HARQ-IR, Turbo)
        # "same"  = same channel model (communication-faithful, default)
        # "judge" = use the judge model as critic
        # "<model_name>" = use a specific model as critic
        critic_model: str = "same",
        # Voter / aggregator model for the vote-style baselines
        # (self_consistency, best_of_n, weighted_bon, cisc, mixture_of_agents,
        # diversity_sc_N, diversity_mrc_discrete_N). This LLM fuses the N
        # samples into one answer.
        # "primary" = use the primary generator, models[0] (default; matches
        #             the historical hard-wired behavior — bit-identical)
        # "judge"   = reuse the judge model as the voter
        # "<model_name>" = use a specific model as the voter
        voter_model: str = "primary",
        # Iterative technique control (HARQ-IR, Turbo)
        # early_exit=False: run all rounds (communication-faithful, default)
        # early_exit=True: allow score-plateau and empty-critique early stopping
        early_exit: bool = False,
        # Per-technique result caching
        # cache_dir: directory where per-technique JSON files are stored
        cache_dir: str = "results/cache",
        # rerun: list of techniques to run fresh (others loaded from cache if available)
        # "all" = rerun everything, ignoring cache
        # [] (empty) = rerun only techniques with no cached file
        rerun: list[str] | str | None = None,
        # When True, the runner swaps each channel's `temperature` to the
        # category-specific value (from the model config's
        # `category_temperatures` map) before running a technique on a task,
        # then restores the base temperature. When False (default), every
        # task is generated at the model's configured `temperature`.
        # The judge and the optional critic are NOT affected — they keep
        # their own fixed temperatures so scoring/critique stays consistent.
        per_category_temperature: bool = False,
        # Number of independent repeats per (task, technique) pair.
        # Default 1 = single run per pair (current behavior).
        # When > 1, each pair is evaluated this many times so that within-task
        # variance (from temperature sampling, judge noise, etc.) can be
        # estimated and reported alongside the paired-Wilcoxon test. Each run
        # is tagged with a `repeat_idx` in [0, repeat_runs) and baselines are
        # matched per (task_id, repeat_idx) when normalizing costs.
        repeat_runs: int = 1,
        # CISC baseline (Taubenfeld et al. 2025) settings. Single dict so the
        # YAML stays grouped:
        #   csi_source: "verbal_100" (default, single-step paper App. B
        #     prompt; works on any backend) or "response_probability"
        #     (intrinsic logprob; backend must expose token logprobs).
        #   softmax_temperature: float | None — scales c_i in the paper's
        #     Definition 3.1 normalization step. If None, defaults to the
        #     median of Figure 8 for the chosen csi_source (verbal_100 → 8,
        #     response_probability → 0.1). For per-model fidelity, set the
        #     value from Figure 8 (e.g. Mistral-22B verbal_100 → 10).
        #   num_samples: int — N reasoning paths per task (default 5).
        cisc: dict | None = None,
        # Number of (task, repeat_idx) workers to run concurrently within a
        # single technique. Default 1 = sequential (current behavior, bit-
        # identical). When > 1, the runner uses a ThreadPoolExecutor so
        # network-bound LLM calls overlap. Per-task state (judge accumulator,
        # category-temperature override) is kept per-thread so concurrent
        # workers don't contaminate each other's accounting.
        # Notes:
        #   - SIGALRM-based per-task watchdog is disabled in parallel mode
        #     (Python's signal module only works on the main thread). HTTP
        #     timeouts on the channel client (240/300s) still apply.
        #   - Tune to backend concurrency limits (e.g. Ollama's parallel
        #     request budget, OpenAI's RPM tier) — too high will throttle.
        parallel_tasks: int = 1,
        # Soft-method aggregation settings: applies CISC's Def 3.1 softmax-
        # with-T normalization (paper §3, Appendix C) to four soft-output
        # techniques whose existing combining math sums raw weights:
        #   - diversity_mrc_soft       (logprob weights → synthesizer)
        #   - diversity_mrc_discrete_N_soft (logprob weights → cluster sum)
        #   - diversity_mrc_discrete_N (judge weights → cluster sum)
        #   - fountain                 (judge weights → synthesizer display)
        # Per-csi_source T defaults from Figure 8 of the paper (logprob,
        # verbal_100) or a documented heuristic (judge). Override per
        # technique only if needed.
        #   enabled: bool                — master on/off switch.
        #   T_logprob, T_judge, T_verbal_100: float — per-source T values.
        soft_normalization: dict | None = None,
        # Optional per-category prompt augmentation. Maps a category name
        # ("qa" | "reasoning" | "creative" | "code") to a small dict that
        # steers generation for tasks in that category WITHOUT touching the
        # text the scorer/judge see (so deterministic extraction stays
        # intact). Each entry picks one mode:
        #   {mode: "as_is"}                      — send the prompt verbatim.
        #   {mode: "system_prompt",              — prepend a system message to
        #    system_prompt: "<text>"}             each generator call.
        #   {mode: "user_prompt_template",       — rewrite the user turn;
        #    user_prompt_template: "...{prompt}..."} `{prompt}` is the only
        #                                          placeholder, replaced with
        #                                          the original task prompt.
        # Categories not listed default to "as_is". Applied per-task in
        # `_augment_task_for_category`, which returns a *new* TaskItem so the
        # shared task objects (reused across techniques / repeats / parallel
        # workers) are never mutated. The judge and the deterministic scorer
        # always receive the original `prompt`/`reference`.
        category_prompts: dict[str, dict[str, Any]] | None = None,
    ):
        self.models = models or [
            {"model": "gpt-4o-mini", "temperature": 0.7},
            {"model": "gpt-4o", "temperature": 0.7},
        ]
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.judge_extra_body = judge_extra_body
        if score_strategy not in ("blended", "exact", "judge"):
            raise ValueError(
                f"score_strategy must be 'blended', 'exact', or 'judge', "
                f"got {score_strategy!r}"
            )
        self.score_strategy = score_strategy
        self.min_difficulty = min_difficulty
        self.acm_table = acm_table
        self.acm_category_tables = acm_category_tables
        self.acm_learned_weights = acm_learned_weights
        self.acm_learned_dispatch_defaults = acm_learned_dispatch_defaults
        self.critic_model = critic_model
        self.voter_model = voter_model
        self.early_exit = early_exit
        self.cache_dir = cache_dir
        self.per_category_temperature = per_category_temperature
        if not isinstance(repeat_runs, int) or repeat_runs < 1:
            raise ValueError(f"repeat_runs must be a positive int, got {repeat_runs!r}")
        self.repeat_runs = repeat_runs
        if not isinstance(parallel_tasks, int) or parallel_tasks < 1:
            raise ValueError(f"parallel_tasks must be a positive int, got {parallel_tasks!r}")
        self.parallel_tasks = parallel_tasks
        # CISC settings — validate eagerly so YAML typos surface at startup,
        # not mid-benchmark. Unrecognized keys are rejected so silent typos
        # (e.g. `csi-source` vs `csi_source`) don't fall back to defaults.
        cisc = cisc or {}
        _CISC_ALLOWED = {"csi_source", "softmax_temperature", "num_samples"}
        unknown = set(cisc) - _CISC_ALLOWED
        if unknown:
            raise ValueError(
                f"Unknown cisc.* config keys: {sorted(unknown)}. "
                f"Allowed: {sorted(_CISC_ALLOWED)}."
            )
        self.cisc = cisc

        # Soft-normalization settings — same eager-validation policy as
        # cisc.* so YAML typos surface at startup. Defaults from CISC
        # paper's Figure 8 medians (logprob, verbal_100) and documented
        # heuristic (judge).
        soft_normalization = soft_normalization or {}
        _SOFT_ALLOWED = {"enabled", "T_logprob", "T_judge", "T_verbal_100"}
        unknown = set(soft_normalization) - _SOFT_ALLOWED
        if unknown:
            raise ValueError(
                f"Unknown soft_normalization.* config keys: {sorted(unknown)}. "
                f"Allowed: {sorted(_SOFT_ALLOWED)}."
            )
        self.soft_normalization = {
            "enabled": soft_normalization.get("enabled", True),
            "T_logprob": soft_normalization.get("T_logprob", 0.1),
            "T_judge": soft_normalization.get("T_judge", 0.5),
            "T_verbal_100": soft_normalization.get("T_verbal_100", 8.0),
        }

        # Per-category prompt augmentation — eager validation so a typo in
        # the YAML (wrong category, unknown mode, missing required text,
        # malformed template) fails at startup, not mid-benchmark.
        self.category_prompts = self._validate_category_prompts(category_prompts)
        # Normalize rerun config
        if rerun is None or rerun == []:
            self.rerun: list[str] | str = []
        elif isinstance(rerun, str):
            self.rerun = rerun  # "all"
        else:
            self.rerun = list(rerun)
        self.techniques = techniques or [
            "baseline",
            "diversity_sc", "diversity_mrc", "diversity_egc",
            "harq_cc", "harq_ir",
            "turbo",
            "fountain",
            "fec_0.75", "fec_0.50", "fec_0.33",
            "acm",
        ]
        self.categories = categories or ["qa", "reasoning", "creative", "code"]
        self.output_dir = output_dir
        self.base_url = base_url
        self.api_key = api_key

    # Canonical category keys + the augmentation modes we accept.
    _VALID_CATEGORIES = frozenset({"qa", "reasoning", "creative", "code"})
    _AUG_MODES = frozenset({"as_is", "system_prompt", "user_prompt_template"})

    @classmethod
    def _validate_category_prompts(
        cls, category_prompts: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Validate the `category_prompts` block and return a normalized copy.

        Returns an empty dict when nothing is configured (the no-op default).
        Raises ValueError on any malformed entry so YAML typos surface at
        construction time rather than silently doing nothing mid-run.
        """
        if not category_prompts:
            return {}
        if not isinstance(category_prompts, dict):
            raise ValueError(
                f"category_prompts must be a mapping of category -> spec, "
                f"got {type(category_prompts).__name__}"
            )
        normalized: dict[str, dict[str, Any]] = {}
        for cat, spec in category_prompts.items():
            if cat not in cls._VALID_CATEGORIES:
                raise ValueError(
                    f"category_prompts: unknown category {cat!r}. "
                    f"Allowed: {sorted(cls._VALID_CATEGORIES)}."
                )
            if not isinstance(spec, dict):
                raise ValueError(
                    f"category_prompts[{cat!r}] must be a dict, got "
                    f"{type(spec).__name__}"
                )
            mode = spec.get("mode", "as_is")
            if mode not in cls._AUG_MODES:
                raise ValueError(
                    f"category_prompts[{cat!r}].mode = {mode!r} is invalid. "
                    f"Allowed: {sorted(cls._AUG_MODES)}."
                )
            allowed_keys = {"mode", "system_prompt", "user_prompt_template"}
            unknown = set(spec) - allowed_keys
            if unknown:
                raise ValueError(
                    f"category_prompts[{cat!r}]: unknown keys {sorted(unknown)}. "
                    f"Allowed: {sorted(allowed_keys)}."
                )
            if mode == "system_prompt":
                text = spec.get("system_prompt")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"category_prompts[{cat!r}].mode='system_prompt' "
                        f"requires a non-empty 'system_prompt' string."
                    )
                normalized[cat] = {"mode": mode, "system_prompt": text}
            elif mode == "user_prompt_template":
                tmpl = spec.get("user_prompt_template")
                if not isinstance(tmpl, str) or "{prompt}" not in tmpl:
                    raise ValueError(
                        f"category_prompts[{cat!r}].mode='user_prompt_template' "
                        f"requires a 'user_prompt_template' string containing "
                        f"the '{{prompt}}' placeholder."
                    )
                normalized[cat] = {"mode": mode, "user_prompt_template": tmpl}
            else:  # as_is
                normalized[cat] = {"mode": "as_is"}
        return normalized


class BenchmarkRunner:
    """
    Runs all configured experiments and collects results.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: list[dict[str, Any]] = []
        # Guards shared mutable state in parallel mode (parallel_tasks>1):
        # self.results / per-technique tech_results / done_keys / completed
        # counter / cache + intermediate file writes. Acquired briefly after
        # each worker finishes a (task, repeat) — never held across an LLM
        # call, so it does not serialize the parallelism.
        self._results_lock = threading.Lock()

        # Fail fast if any model in the config is not in MODEL_COSTS. Otherwise
        # _estimate_cost falls back to the "default" $2/$8 stub, which silently
        # contaminates cost numbers in the cache. We require every model to be
        # explicitly priced in agentcodec/channel.py before a run can start.
        from .channel import MODEL_COSTS
        unknown_models: list[tuple[str, str]] = []
        for m in config.models:
            name = m["model"]
            if name not in MODEL_COSTS:
                unknown_models.append((name, "channel"))
        if config.judge_model and config.judge_model not in MODEL_COSTS:
            unknown_models.append((config.judge_model, "judge"))
        critic = getattr(config, "critic_model", None)
        if critic and critic not in ("same", "judge") and critic not in MODEL_COSTS:
            unknown_models.append((critic, "critic"))
        voter = getattr(config, "voter_model", None)
        if voter and voter not in ("primary", "judge") and voter not in MODEL_COSTS:
            unknown_models.append((voter, "voter"))
        if unknown_models:
            lines = [f"  - {role}: {name!r}" for name, role in unknown_models]
            raise ValueError(
                "Refusing to run: the following models are not in MODEL_COSTS "
                "(would fall back to the $2/$8 default stub and produce wrong "
                "costs in the cache):\n" + "\n".join(lines) +
                "\n\nAdd entries to MODEL_COSTS in agentcodec/channel.py with "
                "the correct (input_per_1M, output_per_1M) pricing, then retry."
            )

        # Create channels
        self.channels: dict[str, AgentChannel] = {}
        for m in config.models:
            ch_kwargs = dict(
                model=m["model"],
                temperature=m.get("temperature", 0.7),
                base_url=m.get("base_url", config.base_url),
                api_key=m.get("api_key", config.api_key),
                extra_body=m.get("extra_body"),
                category_temperatures=m.get("category_temperatures"),
            )
            if "max_tokens" in m:
                ch_kwargs["max_tokens"] = m["max_tokens"]
            ch = AgentChannel(**ch_kwargs)
            self.channels[m["model"]] = ch

        # Create scorer (judge can have its own base_url/api_key).
        # Use judge-specific settings if provided; otherwise fall back to
        # global settings — BUT not for Anthropic models, which need base_url=None
        # to use the native SDK instead of being routed to e.g. Ollama.
        from .channel import _is_anthropic_model
        judge_is_anthropic = _is_anthropic_model(config.judge_model)

        if hasattr(config, 'judge_base_url') and config.judge_base_url is not None:
            judge_base_url = config.judge_base_url
        elif judge_is_anthropic:
            judge_base_url = None  # Anthropic SDK handles its own endpoint
        else:
            judge_base_url = config.base_url

        if hasattr(config, 'judge_api_key') and config.judge_api_key is not None:
            judge_api_key = config.judge_api_key
        elif judge_is_anthropic:
            judge_api_key = None  # uses ANTHROPIC_API_KEY env var
        elif judge_base_url != config.base_url:
            # Judge is on a different provider than the channels (e.g. judge
            # on api.openai.com, channels on local Ollama). The channels'
            # api_key ("ollama") must NOT leak into OpenAI requests — pass
            # None so channel.py picks up OPENAI_API_KEY from the env.
            judge_api_key = None
        else:
            judge_api_key = config.api_key

        self.scorer = QualityScorer(
            judge_model=config.judge_model,
            base_url=judge_base_url,
            api_key=judge_api_key,
            extra_body=getattr(config, "judge_extra_body", None),
            score_strategy=getattr(config, "score_strategy", "blended"),
        )

        # Build critic channel for iterative techniques (HARQ, Turbo)
        # "same" = resolved per-technique to the channel model (default)
        # "judge" = reuse the judge model as critic
        # "<model_name>" = specific model
        self.critic_channel: AgentChannel | None = None
        if config.critic_model == "judge":
            self.critic_channel = self.scorer.judge
        elif config.critic_model != "same":
            critic_is_anthropic = _is_anthropic_model(config.critic_model)
            self.critic_channel = AgentChannel(
                model=config.critic_model,
                temperature=0.2,
                base_url=None if critic_is_anthropic else config.base_url,
                api_key=None if critic_is_anthropic else config.api_key,
            )

        # Build voter/aggregator channel for the vote-style baselines.
        # "primary" = resolved per-technique to the primary generator (default,
        #             bit-identical to the historical hard-wired behavior)
        # "judge" = reuse the judge model as the voter
        # "<model_name>" = specific model
        voter_model = getattr(config, "voter_model", "primary")
        self.voter_channel: AgentChannel | None = None
        if voter_model == "judge":
            self.voter_channel = self.scorer.judge
        elif voter_model != "primary":
            voter_is_anthropic = _is_anthropic_model(voter_model)
            self.voter_channel = AgentChannel(
                model=voter_model,
                temperature=0.0,
                base_url=None if voter_is_anthropic else config.base_url,
                api_key=None if voter_is_anthropic else config.api_key,
            )

        # Output path
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Per-technique cache directory
        self.cache_path = Path(config.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _should_rerun(self, tech_name: str) -> bool:
        """Check whether a technique should be rerun or can use cached results."""
        rerun = self.config.rerun
        if rerun == "all":
            return True
        if isinstance(rerun, list) and tech_name in rerun:
            return True
        # No cache file → must run
        if not self._cache_file(tech_name).exists():
            return True
        return False

    def _cache_file(self, tech_name: str) -> Path:
        """Path to the per-technique cache file."""
        # Sanitize tech name for filesystem (e.g. fec_0.50 → fec_0.50.json)
        return self.cache_path / f"{tech_name}.json"

    def _load_cached(self, tech_name: str) -> list[dict[str, Any]]:
        """Load cached results for a technique. Returns [] if not found."""
        path = self._cache_file(tech_name)
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", [])
            logger.info(f"  Loaded {len(results)} cached results from {path.name}")
            return results
        except Exception as e:
            logger.warning(f"  Failed to load cache {path}: {e}")
            return []

    def _save_cached(self, tech_name: str, results: list[dict[str, Any]]):
        """Save per-technique results to cache.

        Dedupes by (task_id, repeat_idx) before writing — keeping the LAST
        entry per key. This is a defense-in-depth safety net against any code
        path that accidentally passes duplicates (historical ACM caches grew
        to 4× the expected size because of such a bug).
        """
        path = self._cache_file(tech_name)
        # Dedupe: last-write-wins per (task_id, repeat_idx).
        seen: dict[tuple[Any, int], int] = {}
        for i, r in enumerate(results):
            key = (r.get("task_id"), r.get("repeat_idx", 0))
            seen[key] = i  # overwrites earlier idx for same key
        deduped = [results[i] for i in sorted(seen.values())]
        if len(deduped) != len(results):
            logger.warning(
                f"  _save_cached({tech_name}): dropped "
                f"{len(results) - len(deduped)} duplicate (task_id, repeat_idx) "
                f"entries before write"
            )
        try:
            with open(path, "w") as f:
                json.dump({
                    "technique": tech_name,
                    "num_results": len(deduped),
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": {
                        "models": self.config.models,
                        "judge_model": self.config.judge_model,
                        "critic_model": self.config.critic_model,
                        "voter_model": getattr(self.config, "voter_model", "primary"),
                        "score_strategy": getattr(self.config, "score_strategy", "blended"),
                    },
                    "results": deduped,
                }, f, indent=2)
            logger.info(f"  Cached {len(deduped)} results → {path}")
        except Exception as e:
            logger.warning(f"  Failed to save cache {path}: {e}")

    def _resave_caches_normalized(self, techniques: list[str]):
        """Re-save cache files with normalized cost data (cost_overhead, etc.).

        Cache files are initially saved after each technique finishes but BEFORE
        cross-technique normalization (which needs baseline data). This method
        re-writes them after normalization so cached results include cost_overhead.
        """
        from collections import defaultdict

        by_tech: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_tech[r.get("technique", "")].append(r)

        for tech in techniques:
            tech_results = by_tech.get(tech, [])
            if tech_results:
                self._save_cached(tech, tech_results)

        logger.info(f"Re-saved {len(techniques)} cache files with normalized costs")

    def run_all(
        self,
        tasks: list[TaskItem],
        live_plots: bool = True,
        plot_dir: str = "plots",
    ) -> list[dict[str, Any]]:
        """
        Run all configured techniques on all provided tasks.
        Techniques with cached results are loaded instead of rerun,
        unless listed in config.rerun or config.rerun == "all".

        Args:
            live_plots: If True, regenerate plots after each technique finishes.
            plot_dir: Directory to save plots to.
        """
        # Filter tasks by configured categories
        active_categories = {TaskCategory(c) for c in self.config.categories}
        filtered_tasks = [t for t in tasks if t.category in active_categories]

        # Always need baseline for cost normalization
        techniques_to_run = list(self.config.techniques)
        if "baseline" not in techniques_to_run:
            techniques_to_run.insert(0, "baseline")
            logger.info("Auto-adding baseline for cost normalization")

        # Determine force-rerun set (wipes partial cache so technique starts over)
        rerun_cfg = self.config.rerun
        if rerun_cfg == "all":
            force_rerun = set(techniques_to_run)
        elif isinstance(rerun_cfg, list):
            force_rerun = set(rerun_cfg)
        else:
            force_rerun = set()

        repeats = max(1, int(self.config.repeat_runs))

        # For each technique, load any existing partial cache and build the
        # set of (task_id, repeat_idx) keys already done. Tasks already in the
        # cache are skipped — this lets a killed run resume from where it left off.
        # Force-rerun techniques get their cache wiped before starting.
        tech_existing: dict[str, list[dict[str, Any]]] = {}
        tech_done_keys: dict[str, set[tuple[Any, int]]] = {}
        total_experiments = 0
        for tech in techniques_to_run:
            if tech in force_rerun:
                cf = self._cache_file(tech)
                if cf.exists():
                    try:
                        backups_dir = cf.parent / "backups"
                        backups_dir.mkdir(exist_ok=True)
                        backup_path = backups_dir / f"{cf.name}.{int(time.time())}"
                        cf.rename(backup_path)
                        logger.info(f"  Backed up cache to {backup_path.relative_to(cf.parent.parent)} (forced rerun)")
                    except Exception as e:
                        logger.warning(f"  Could not back up {cf}: {e}")
                tech_existing[tech] = []
                tech_done_keys[tech] = set()
            else:
                existing = self._load_cached(tech)
                # Only count non-error results as "done" — errors will be retried.
                done: set[tuple[Any, int]] = {
                    (r.get("task_id"), r.get("repeat_idx", 0))
                    for r in existing if "error" not in r
                }
                tech_existing[tech] = existing
                tech_done_keys[tech] = done
            # Count remaining experiments (tasks not already cached)
            for task in filtered_tasks:
                for repeat_idx in range(repeats):
                    if (task.id, repeat_idx) not in tech_done_keys[tech]:
                        total_experiments += 1

        loaded_from_cache = sum(len(tech_existing[t]) for t in techniques_to_run)
        completed = 0

        logger.info(
            f"Starting benchmark: {len(techniques_to_run)} techniques × "
            f"{len(filtered_tasks)} tasks × {repeats} repeat(s). "
            f"Loaded {loaded_from_cache} cached results; "
            f"{total_experiments} remaining experiments to run."
        )
        if repeats > 1:
            logger.info(
                f"repeat_runs={repeats}: each (task, technique) will be "
                f"evaluated {repeats} times for within-task variance estimation."
            )

        # Intermediate results file — updated after every experiment
        intermediate_path = self.output_path / "benchmark_intermediate.json"

        # Seed self.results with cached data up front, so live plots /
        # normalization passes see the full picture from experiment 0.
        for tech in techniques_to_run:
            self.results.extend(tech_existing[tech])

        # Run each technique (skipping tasks whose results are already cached).
        # Cache is saved after every task so a killed run can resume exactly
        # where it left off — no more losing a whole technique's progress.
        for tech_name in techniques_to_run:
            done_keys = tech_done_keys[tech_name]
            remaining = [
                (task, r) for task in filtered_tasks for r in range(repeats)
                if (task.id, r) not in done_keys
            ]
            cached_count = len(tech_existing[tech_name])
            if not remaining:
                logger.info(
                    f"\n{'='*60}\nTechnique: {tech_name} "
                    f"(fully cached, {cached_count} results)\n{'='*60}"
                )
                continue
            logger.info(
                f"\n{'='*60}\nTechnique: {tech_name} "
                f"({cached_count} cached, {len(remaining)} remaining)\n{'='*60}"
            )
            # Start from existing cached rows; new results are appended and
            # the whole list is re-saved after each task.
            tech_results: list[dict[str, Any]] = list(tech_existing[tech_name])

            parallel = max(1, int(self.config.parallel_tasks))
            pending = [
                (task, r) for task in filtered_tasks for r in range(repeats)
                if (task.id, r) not in done_keys
            ]

            def _record(task: TaskItem, repeat_idx: int, result: dict[str, Any]) -> None:
                """Append result + persist cache + intermediate. Lock-protected."""
                nonlocal completed
                with self._results_lock:
                    self.results.append(result)
                    tech_results.append(result)
                    done_keys.add((task.id, repeat_idx))
                    completed += 1
                    repeat_tag = f" [rep {repeat_idx+1}/{repeats}]" if repeats > 1 else ""
                    logger.info(
                        f"  Task: {task.id} ({task.category.value}){repeat_tag}"
                        f" → [{completed}/{total_experiments}] "
                        f"quality={result.get('final_quality', 0):.3f}, "
                        f"diversity_gain={result.get('diversity_gain', 0):+.3f}, "
                        f"calls={result.get('num_llm_calls', 0)}, "
                        f"cost=${result.get('total_cost_usd', 0):.4f}"
                    )
                    self._save_cached(tech_name, tech_results)
                    self._save_intermediate(intermediate_path, completed, total_experiments)

            if parallel == 1:
                # Sequential: bit-identical to pre-parallel behavior.
                for task, repeat_idx in pending:
                    try:
                        result = self._run_one(tech_name, task, repeat_idx, repeats)
                    except Exception as e:
                        logger.error(
                            f"    → FAILED [{task.id} rep {repeat_idx}]: {e} "
                            f"— stopping simulation (error result NOT saved to cache)"
                        )
                        raise
                    _record(task, repeat_idx, result)
            else:
                # Parallel: dispatch (task, repeat) workers to a thread pool.
                # Threads are appropriate (LLM calls are I/O-bound) and let
                # workers share the channel/scorer instances; per-thread state
                # (TLS temperature override, TLS judge accumulator) keeps
                # accounting isolated.
                logger.info(
                    f"  Running with parallel_tasks={parallel} "
                    f"(SIGALRM watchdog disabled in parallel mode; HTTP "
                    f"timeouts on the channel client still apply)"
                )
                with ThreadPoolExecutor(
                    max_workers=parallel,
                    thread_name_prefix=f"agentcodec-{tech_name}",
                ) as ex:
                    futures = {
                        ex.submit(self._run_one, tech_name, task, repeat_idx, repeats):
                            (task, repeat_idx)
                        for task, repeat_idx in pending
                    }
                    try:
                        for fut in as_completed(futures):
                            task, repeat_idx = futures[fut]
                            try:
                                result = fut.result()
                            except Exception as e:
                                logger.error(
                                    f"    → FAILED [{task.id} rep {repeat_idx}]: {e} "
                                    f"— cancelling remaining workers and stopping"
                                )
                                for f in futures:
                                    f.cancel()
                                raise
                            _record(task, repeat_idx, result)
                    except BaseException:
                        # Ensure no zombie futures on error/Ctrl-C — the with-
                        # block will join workers, but cancel queued ones first.
                        for f in futures:
                            f.cancel()
                        raise

            # Update plots after each technique finishes
            if live_plots:
                self._update_live_plots(plot_dir, tech_name, completed, total_experiments)

        # Normalize costs against per-task baselines.
        # This adds cost_overhead, baseline_cost_usd, etc. to self.results.
        self._normalize_costs()

        # Re-save cache files WITH normalized cost data.
        # Previously, cache was saved before normalization so cost_overhead
        # was always missing from cached results. When plots loaded from cache,
        # _ensure_normalized could only recompute if total_cost_usd > 0 — but
        # some techniques (or model configs) may have zero cost data, causing
        # permanent cost_overhead=0 in plots. By saving normalized data to
        # cache, this is fixed once and for all.
        self._resave_caches_normalized(techniques_to_run)

        # Save final results
        final_path = self._save_results()
        # Clean up intermediate file
        if intermediate_path.exists():
            intermediate_path.unlink()
            logger.info(f"Removed intermediate file (final results at {final_path})")
        return self.results

    def _run_one(
        self,
        tech_name: str,
        task: TaskItem,
        repeat_idx: int,
        repeats: int,
    ) -> dict[str, Any]:
        """Execute a single (task, repeat_idx) for tech_name and return its result dict.

        Centralises the per-experiment flow used by both sequential and
        parallel dispatch in `run_all`. Re-raises on failure so the caller
        decides whether to abort the whole run.

        Per-task isolation in parallel mode (parallel_tasks > 1):
          * Judge accumulator is thread-local on the scorer, so calling
            `collect_judge_outputs()` here only resets/reads this worker's
            own list — never another concurrent task's.
          * `_category_temperature_swap` writes to a per-thread override on
            each channel rather than mutating the shared `temperature` field.

        SIGALRM watchdog is bypassed in parallel mode because Python's signal
        module only works on the main thread. Channel HTTP timeouts
        (240/300s) and the per-attempt retry loop in `transmit()` still
        guard against stuck calls.
        """
        timeout_s = int(os.environ.get("AGENTCODEC_TASK_TIMEOUT_S", "600"))
        max_attempts = int(os.environ.get("AGENTCODEC_TASK_MAX_ATTEMPTS", "3"))
        watchdog_s = timeout_s if int(self.config.parallel_tasks) <= 1 else 0

        run = None
        for attempt in range(max_attempts):
            try:
                # Reset this thread's judge output accumulator before each
                # attempt — a partial-run's judge calls shouldn't leak into
                # the retry's accounting.
                self.scorer.collect_judge_outputs()
                with _task_timeout(
                    watchdog_s,
                    name=f"task {task.id} ({tech_name})",
                ):
                    with self._category_temperature_swap(task):
                        run = self._run_technique(
                            tech_name, self._augment_task_for_category(task)
                        )
                break
            except TimeoutError as te:
                if attempt + 1 < max_attempts:
                    logger.warning(
                        f"    → TIMEOUT (attempt "
                        f"{attempt + 1}/{max_attempts}): {te} — retrying"
                    )
                else:
                    logger.error(
                        f"    → TIMEOUT (final, {max_attempts} attempts): {te}"
                    )
                    raise

        # Collect all judge LLM calls made during this run (this thread only).
        run.judge_outputs = self.scorer.collect_judge_outputs()
        run.compute_metrics()  # recompute to include judge costs

        # Guardrail: detect silent channel failures.
        # transmit() catches exceptions and returns an AgentOutput with
        # text="[ERROR: ...]" and token_count=0 so the technique can keep
        # running. If *every* generator/overhead call errored out (connection
        # refused, model not loaded, etc.), the run is garbage — fail loudly
        # instead of caching it.
        self._validate_run_or_raise(run, tech_name, task)

        result = self._run_to_dict(run)
        if repeats > 1:
            result["repeat_idx"] = repeat_idx
        return result

    def _update_live_plots(self, plot_dir: str, tech_name: str, completed: int, total: int):
        """Regenerate plots from current results after a technique finishes.

        Plotting helpers live in ``agentcodec.plots`` and are an optional
        install (``pip install 'agentcodec[benchmark]'``). The import is
        wrapped in try/except so the library gracefully degrades when the
        plotting deps aren't installed — no plots, no crash.
        """
        try:
            from agentcodec.plots import _model_info_text, plot_all_from_results
            valid_results = [r for r in self.results if "error" not in r]
            if len(valid_results) < 2:
                return  # not enough data for meaningful plots
            model_info = _model_info_text({
                "models": self.config.models,
                "judge_model": self.config.judge_model,
                "critic_model": self.config.critic_model,
                "voter_model": getattr(self.config, "voter_model", "primary"),
            })
            plot_all_from_results(valid_results, plot_dir, model_info=model_info)
            logger.info(
                f"  📊 Live plots updated ({completed}/{total}) — "
                f"after {tech_name}"
            )
        except Exception as e:
            logger.debug(f"Live plot update failed (non-fatal): {e}")

    def _save_intermediate(self, path: Path, completed: int, total: int):
        """Save intermediate results so progress is not lost on crash."""
        try:
            with open(path, "w") as f:
                json.dump(
                    {
                        "status": "in_progress",
                        "progress": f"{completed}/{total}",
                        "results": self.results,
                        "summary": self._compute_summary(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")

    def _validate_run_or_raise(self, run, tech_name: str, task) -> None:
        """
        Detect silent channel failures and raise so the run is recorded as an
        error instead of cached as garbage.

        A `transmit()` call that hits the channel's except branch returns an
        AgentOutput with text starting "[ERROR:" and token_count=0. If every
        generator/overhead output looks like that, the backend was down and
        this run is meaningless — raise to surface it.
        """
        gen_out = list(run.individual_outputs) + list(run.overhead_outputs)
        if not gen_out:
            return

        def _is_errored(o) -> bool:
            text = (o.text or "")
            return o.token_count == 0 and text.startswith("[ERROR:")

        errored = [o for o in gen_out if _is_errored(o)]
        if len(errored) == len(gen_out):
            sample_err = errored[0].text[:200] if errored else "unknown"
            raise RuntimeError(
                f"[{tech_name}/{task.id}] all {len(gen_out)} generator/overhead "
                f"calls failed at the channel level — backend likely unreachable. "
                f"First error: {sample_err}"
            )

        # Partial failure: also flag if we produced zero content tokens across
        # the entire generator path (catches edge cases where transmit returned
        # empty content without raising).
        total_gen_tokens = sum(o.token_count for o in gen_out)
        if total_gen_tokens == 0:
            raise RuntimeError(
                f"[{tech_name}/{task.id}] generator/overhead calls produced "
                f"0 tokens across {len(gen_out)} outputs — backend likely "
                f"returning empty responses."
            )

    def _category_temperature_swap(self, task: TaskItem):
        """
        Context manager that scopes each generator channel's effective
        temperature to the value resolved for `task.category`, restoring
        the channel default on exit.

        Implementation detail: writes to a per-thread override
        (`AgentChannel._tls.temperature_override`) instead of mutating the
        shared `channel.temperature` field. That keeps concurrent (task,
        repeat) workers in parallel mode (parallel_tasks>1) isolated — two
        tasks of different categories running on the same channel each see
        their own temperature with no race.

        No-op when `per_category_temperature` is False or no channel has a
        `category_temperatures` map. The judge and critic channels are NOT
        touched — they keep their fixed scoring/critique temperatures.
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if not self.config.per_category_temperature:
                yield
                return
            overridden: list = []
            for ch in self.channels.values():
                if ch.category_temperatures:
                    ch._tls.temperature_override = ch.temperature_for_category(task.category)
                    overridden.append(ch)
            try:
                yield
            finally:
                for ch in overridden:
                    try:
                        del ch._tls.temperature_override
                    except AttributeError:
                        pass
        return _ctx()

    def _augment_task_for_category(self, task: TaskItem) -> TaskItem:
        """Return a TaskItem whose *generator* prompt is augmented per the
        configured `category_prompts` rule for `task.category`.

        Returns the task unchanged when no rule applies (or mode is
        "as_is"). Otherwise returns a NEW TaskItem (via dataclasses.replace)
        so the shared task objects — reused across techniques, repeats, and
        parallel workers — are never mutated.

        Only `request` (what techniques send to the model) is rewritten.
        `prompt` and `reference` are left untouched so the deterministic
        scorers (last-number / letter / code extraction) and the LLM judge
        always grade against the original task text.
        """
        rules = self.config.category_prompts
        if not rules:
            return task
        spec = rules.get(task.category.value)
        if not spec or spec.get("mode", "as_is") == "as_is":
            return task

        from dataclasses import replace as _dc_replace

        from .messages import ChatRequest

        base_req = task.request or ChatRequest.from_prompt(task.prompt)
        mode = spec["mode"]
        if mode == "system_prompt":
            # Prepend (or replace) the system message for generation only.
            new_req = base_req.with_system(spec["system_prompt"])
        elif mode == "user_prompt_template":
            # Rewrite the user turn, substituting the original prompt text.
            new_user = spec["user_prompt_template"].replace("{prompt}", task.prompt)
            new_req = base_req.with_user(new_user)
        else:  # pragma: no cover — guarded by _validate_category_prompts
            return task

        # New TaskItem: augmented request for generation, original prompt /
        # reference / metadata / score_mode preserved for scoring.
        return _dc_replace(task, request=new_req)

    def _run_technique(self, tech_name: str, task: TaskItem) -> ReliabilityRun:
        """Dispatch to the appropriate technique implementation."""
        channels_list = list(self.channels.values())
        primary = channels_list[0]
        # Voter/aggregator for vote-style baselines. None (default voter_model
        # "primary") falls back to the primary generator — bit-identical to the
        # previous hard-wired behavior.
        voter = self.voter_channel or primary

        if tech_name == "baseline":
            return self._run_baseline(task, primary)

        # --- Prior-method baselines (reproduced at matched inference budget) ---

        elif tech_name == "self_consistency":
            svc = SelfConsistencyBaseline(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
                voter=voter,
            )
            return svc.run(task)

        elif tech_name == "self_refine":
            svc = SelfRefineBaseline(
                channel=primary,
                scorer=self.scorer,
                max_rounds=3,
            )
            return svc.run(task)

        elif tech_name == "chain_of_verification":
            svc = ChainOfVerificationBaseline(
                channel=primary,
                scorer=self.scorer,
                num_verification_questions=3,
            )
            return svc.run(task)

        elif tech_name == "diversity_sc_N":
            svc = SelectionCombiningN(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
            )
            return svc.run(task)

        elif tech_name == "diversity_mrc_discrete_N":
            sn = self.config.soft_normalization
            svc = DiversityMRCDiscreteN(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
                voter=voter,
                softmax_normalize=sn["enabled"],
                softmax_temperature=sn["T_judge"],
            )
            return svc.run(task)

        elif tech_name == "best_of_n":
            svc = BestOfNBaseline(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
            )
            return svc.run(task)

        elif tech_name == "weighted_bon":
            svc = WeightedBoNBaseline(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
                voter=voter,
            )
            return svc.run(task)

        elif tech_name == "cisc":
            cisc_cfg = self.config.cisc
            svc = CISCBaseline(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=cisc_cfg.get("num_samples", 5),
                voter=voter,
                csi_source=cisc_cfg.get("csi_source", "verbal_100"),
                softmax_temperature=cisc_cfg.get("softmax_temperature"),
            )
            return svc.run(task)

        elif tech_name == "mixture_of_agents":
            svc = MixtureOfAgentsBaseline(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
                aggregator=voter,
            )
            return svc.run(task)

        # --- Soft-output techniques (must be checked before generic diversity_*) ---

        elif tech_name == "diversity_mrc_soft":
            sn = self.config.soft_normalization
            svc = SoftDiversityMRC(
                channels=channels_list,
                scorer=self.scorer,
                softmax_normalize=sn["enabled"],
                softmax_temperature=sn["T_logprob"],
            )
            return svc.run(task, synthesizer=primary)

        elif tech_name == "diversity_mrc_discrete_N_soft":
            sn = self.config.soft_normalization
            svc = SoftDiversityMRCDiscreteN(
                channels=channels_list,
                scorer=self.scorer,
                num_samples=5,
                voter=voter,
                softmax_normalize=sn["enabled"],
                softmax_temperature=sn["T_logprob"],
            )
            return svc.run(task)

        elif tech_name == "fountain_soft":
            svc = SoftFountainDecoder(
                channels=channels_list,
                scorer=self.scorer,
                max_samples=8,
            )
            return svc.run(task)

        elif tech_name == "acm_soft":
            svc = SoftACMRouter(
                channels=self.channels,
                scorer=self.scorer,
            )
            return svc.run(task)

        elif tech_name.startswith("diversity_"):
            strategy = tech_name.split("_", 1)[1]
            combining = CombiningStrategy(strategy)
            svc = DiversityEnsemble(
                channels=channels_list,
                scorer=self.scorer,
                combining=combining,
            )
            return svc.run(task, synthesizer=primary)

        elif tech_name == "diversity_spatial":
            # Pure spatial diversity — different models, same prompt
            svc = DiversityEnsemble(
                channels=channels_list,
                scorer=self.scorer,
                combining=CombiningStrategy.MRC,
            )
            return svc.run(task, synthesizer=primary)

        elif tech_name == "diversity_frequency":
            # Pure frequency diversity — same model, different prompts
            from .techniques.diversity import DEFAULT_PROMPT_VARIANTS
            svc = DiversityEnsemble(
                channels=[primary],
                scorer=self.scorer,
                combining=CombiningStrategy.MRC,
                prompt_variants=DEFAULT_PROMPT_VARIANTS,
            )
            return svc.run(task, synthesizer=primary)

        elif tech_name == "diversity_time":
            # Pure time diversity — same model, different temperatures
            svc = DiversityEnsemble(
                channels=[primary],
                scorer=self.scorer,
                combining=CombiningStrategy.MRC,
                temperature_spread=[0.3, 0.5, 0.7, 0.9],
            )
            return svc.run(task, synthesizer=primary)

        elif tech_name.startswith("harq_"):
            mode = HARQMode(tech_name.split("_", 1)[1])
            svc = HARQService(
                channel=primary,
                scorer=self.scorer,
                mode=mode,
                max_rounds=5,
                critic_channel=self.critic_channel,  # None → defaults to channel
                early_exit=self.config.early_exit,
            )
            return svc.run(task)

        elif tech_name == "turbo":
            generator = channels_list[0]
            # Resolve critic channel for turbo decoding.
            #
            # Real turbo codes use two SISO decoders of EQUAL capability
            # separated by an interleaver. The extrinsic information exchange
            # only works when both decoders can reliably decode — a weaker
            # decoder 2 injects noise that prevents convergence.
            #
            # critic_model="same" (default) → use generator model as critic.
            # This matches real turbo codes (same trellis for both decoders).
            # Diversity comes from the interleaver (different prompting), not
            # from model mismatch.
            #
            # BUG FIX: Previously, "same" fell through to channels_list[1]
            # (the second, often weaker model), causing cross-model noise
            # injection that degraded quality with each iteration.
            if self.critic_channel:
                critic = self.critic_channel
            else:
                critic = None  # defaults to generator inside TurboDecoder
            svc = TurboDecoder(
                generator=generator,
                critic=critic,
                scorer=self.scorer,
                max_iterations=5,
                # Match HARQ-IR threshold — turbo should not have a higher
                # bar than HARQ since both are iterative refinement techniques.
                quality_threshold=0.85,
                early_exit=self.config.early_exit,
            )
            return svc.run(task)

        elif tech_name == "fountain":
            sn = self.config.soft_normalization
            svc = FountainDecoder(
                channels=channels_list,
                scorer=self.scorer,
                max_samples=8,
                softmax_normalize=sn["enabled"],
                softmax_temperature=sn["T_judge"],
            )
            return svc.run(task)

        elif tech_name.startswith("fec_"):
            rate = float(tech_name.split("_", 1)[1])
            svc = FECService(
                channel=primary,
                scorer=self.scorer,
                code_rate=rate,
            )
            return svc.run(task)

        elif tech_name == "acm":
            acm_table = self._build_acm_table() if self.config.acm_table else None
            category_tables = self._build_acm_category_tables() if self.config.acm_category_tables else None
            svc = ACMRouter(
                channels=self.channels,
                scorer=self.scorer,
                acm_table=acm_table,
                critic_channel=self.critic_channel,
                category_tables=category_tables,
            )
            return svc.run(task)

        elif tech_name == "acm_learned":
            weights_path = self.config.acm_learned_weights
            if not weights_path:
                raise ValueError(
                    "Technique 'acm_learned' requires `acm_learned_weights` "
                    "in the config — path to a JSON produced by the upstream "
                    "ACM router trainer (not shipped in the open-source "
                    "release)."
                )
            svc = ACMLearnedRouter(
                channels=self.channels,
                scorer=self.scorer,
                router_weights=weights_path,
                dispatch_defaults=self.config.acm_learned_dispatch_defaults,
                critic_channel=self.critic_channel,
            )
            run = svc.run(task)
            # Ensure the top-level technique tag is stable for cache lookups,
            # even though the inner dispatch tags it as e.g. "acm_learned_harq_ir".
            run.config["routed_tagged_technique"] = run.technique
            run.technique = "acm_learned"
            return run

        else:
            raise ValueError(f"Unknown technique: {tech_name}")

    def _build_acm_table(self) -> list[Any]:
        """Build ACMProfile list from config dicts."""
        return self._entries_to_profiles(self.config.acm_table or [])

    def _build_acm_category_tables(self) -> dict[str, list[Any]]:
        """Build per-category ACMProfile tables from config dicts."""
        return {
            cat: self._entries_to_profiles(entries)
            for cat, entries in (self.config.acm_category_tables or {}).items()
        }

    def _entries_to_profiles(self, entries: list[dict[str, Any]]) -> list[Any]:
        from .techniques.acm import ACMProfile
        # Default model: first configured channel (per-entry override still
        # wins). Lets category tables omit "model" when all profiles share
        # the generator, which is the common case.
        default_model = next(iter(self.channels.keys())) if self.channels else ""
        profiles: list[Any] = []
        for entry in entries:
            profiles.append(ACMProfile(
                name=entry.get("name", ""),
                difficulty_range=tuple(entry.get("difficulty_range", [0.0, 1.0])),
                model=entry.get("model", default_model),
                technique=entry["technique"],
                code_rate=entry.get("code_rate", 1.0),
                num_branches=entry.get("num_branches", 1),
                max_rounds=entry.get("max_rounds", 1),
                estimated_cost_multiplier=entry.get("estimated_cost_multiplier", 1.0),
            ))
        return profiles

    def _run_baseline(self, task: TaskItem, channel: AgentChannel) -> ReliabilityRun:
        """Baseline: single agent, no redundancy (uncoded transmission)."""
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="baseline",
            config={"model": channel.model},
        )
        out = channel.transmit(task.request)
        out.quality_score = self.scorer.score(task.prompt, out.text, reference=task.reference, task=task)
        run.individual_outputs = [out]
        run.combined_output = out.text
        run.final_quality = out.quality_score
        run.compute_metrics()
        return run

    def _normalize_costs(self):
        """
        Normalize cost metrics against per-task baselines.

        Instead of raw cost (which makes hard tasks look "expensive and bad"),
        we compute cost_overhead = technique_cost / baseline_cost for the same
        task. This is analogous to Eb/N0 normalization in communications —
        it accounts for the "channel conditions" (task difficulty).
        """
        from collections import defaultdict

        # Step 1: Find baseline cost/quality per (task, repeat_idx).
        # When repeat_runs=1 (default), repeat_idx is absent from result dicts
        # and everything reduces to a single key per task (preserving the
        # original behavior bit-for-bit). When repeat_runs>1, each repeat gets
        # its own matched baseline so within-task variance is not conflated
        # with cross-repeat variance.
        def _bl_key(r: dict) -> tuple:
            return (r["task_id"], r.get("repeat_idx", 0))

        baselines: dict[tuple, dict[str, float]] = {}

        by_task: dict[tuple, list[dict]] = defaultdict(list)
        for r in self.results:
            if "error" not in r:
                by_task[_bl_key(r)].append(r)

        for key, runs in by_task.items():
            # Prefer explicit baseline; otherwise use cheapest run
            baseline_runs = [r for r in runs if r["technique"] == "baseline"]
            if baseline_runs:
                bl = baseline_runs[0]
            else:
                bl = min(runs, key=lambda r: r.get("total_cost_usd", float("inf")))
            baselines[key] = {
                "cost": bl.get("total_cost_usd", 0),
                "quality": bl.get("final_quality", 0),
                "calls": bl.get("num_llm_calls", 1),
            }

        # Step 2: Set normalized fields on each result
        for r in self.results:
            if "error" in r:
                continue
            bl = baselines.get(_bl_key(r), {"cost": 0, "quality": 0, "calls": 1})
            bl_cost = bl["cost"]
            bl_qual = bl["quality"]

            r["baseline_cost_usd"] = bl_cost
            r["baseline_quality"] = bl_qual

            r_cost = r.get("total_cost_usd", 0)
            r_calls = r.get("num_llm_calls", 0)

            # Cost overhead: technique_cost / baseline_cost.
            # Fallback to call-count ratio when cost data is missing
            # (e.g. Ollama doesn't always report usage tokens).
            if bl_cost > 0 and r_cost > 0:
                r["cost_overhead"] = r_cost / bl_cost
            elif r_calls > 0 and bl.get("calls", 0) > 0:
                r["cost_overhead"] = r_calls / bl["calls"]
            else:
                r["cost_overhead"] = 1.0

            # Quality gain per additional dollar
            quality_delta = r["final_quality"] - bl_qual
            cost_delta = r_cost - bl_cost
            if cost_delta > 0:
                r["quality_gain_per_cost"] = quality_delta / cost_delta
            else:
                r["quality_gain_per_cost"] = 0.0

            # Cost efficiency: quality per dollar
            if r_cost > 0:
                r["cost_efficiency"] = r["final_quality"] / r_cost
            else:
                r["cost_efficiency"] = 0.0

        logger.info(
            f"Normalized costs for {len(baselines)} (task, repeat) pairs "
            f"(baseline runs found: {sum(1 for r in self.results if r.get('technique') == 'baseline')})"
        )

    def _run_to_dict(self, run: ReliabilityRun) -> dict[str, Any]:
        """Convert a ReliabilityRun to a serializable dict."""
        d = {
            "id": run.id,
            "task_id": run.task_id,
            "task_category": run.task_category,
            "technique": run.technique,
            "config": run.config,
            "final_quality": run.final_quality,
            "best_individual_quality": run.best_individual_quality,
            "mean_individual_quality": run.mean_individual_quality,
            "diversity_gain": run.diversity_gain,
            "coding_gain": run.coding_gain,
            "num_llm_calls": run.num_llm_calls,
            "total_tokens": run.total_tokens,
            "total_cost_usd": run.total_cost_usd,
            "judge_cost_usd": run.judge_cost_usd,
            "total_latency_s": run.total_latency_s,
            "rounds": run.rounds,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            # Per-bucket breakdowns: every dollar/token in the cache is now
            # attributable to a specific model and a specific bucket, so the
            # cache can be repriced under new MODEL_COSTS without reruns.
            "individual_tokens": run.individual_tokens,
            "individual_cost_usd": run.individual_cost_usd,
            "individual_by_model": {m: {"tokens": t, "cost_usd": c}
                                    for m, (t, c) in run.individual_by_model.items()},
            "overhead_tokens": run.overhead_tokens,
            "overhead_cost_usd": run.overhead_cost_usd,
            "overhead_by_model": {m: {"tokens": t, "cost_usd": c}
                                  for m, (t, c) in run.overhead_by_model.items()},
            "judge_tokens": run.judge_tokens,
            "judge_by_model": {m: {"tokens": t, "cost_usd": c}
                               for m, (t, c) in run.judge_by_model.items()},
            # Per-output quality scores for convergence plots
            "individual_scores": [o.quality_score for o in run.individual_outputs],
            "individual_models": [o.model for o in run.individual_outputs],
        }
        return d

    def _save_results(self):
        """Save results to JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.output_path / f"benchmark_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(
                {
                    "config": {
                        "models": self.config.models,
                        "judge_model": self.config.judge_model,
                        "critic_model": self.config.critic_model,
                        "voter_model": getattr(self.config, "voter_model", "primary"),
                        "score_strategy": getattr(self.config, "score_strategy", "blended"),
                        "techniques": self.config.techniques,
                        "categories": self.config.categories,
                    },
                    "results": self.results,
                    "summary": self._compute_summary(),
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {filepath}")
        return filepath

    @staticmethod
    def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the mean."""
        import numpy as np
        if len(values) < 2:
            m = values[0] if values else 0
            return (m, m)
        arr = np.array(values)
        rng = np.random.default_rng(42)
        means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        return (float(np.percentile(means, 100 * alpha)),
                float(np.percentile(means, 100 * (1 - alpha))))

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics with bootstrap CIs and significance tests."""
        from collections import defaultdict

        by_technique: dict[str, list[dict]] = defaultdict(list)
        by_category: dict[str, list[dict]] = defaultdict(list)

        for r in self.results:
            if "error" not in r:
                by_technique[r["technique"]].append(r)
                by_category[r["task_category"]].append(r)

        # Build per-task baseline lookup for paired tests
        baseline_by_task: dict[str, float] = {}
        for r in by_technique.get("baseline", []):
            baseline_by_task[r["task_id"]] = r["final_quality"]

        technique_summary = {}
        for tech, runs in by_technique.items():
            quals = [r["final_quality"] for r in runs]
            costs = [r["total_cost_usd"] for r in runs]
            overheads = [r.get("cost_overhead", 0) for r in runs if r.get("cost_overhead") is not None]
            gains_per_cost = [r.get("quality_gain_per_cost", 0) for r in runs
                              if r.get("quality_gain_per_cost") is not None
                              and r.get("quality_gain_per_cost") != float("inf")]
            qual_ci = self._bootstrap_ci(quals)

            entry = {
                "mean_quality": sum(quals) / len(quals) if quals else 0,
                "quality_ci_95": list(qual_ci),
                "std_quality": float((sum((q - sum(quals)/len(quals))**2 for q in quals) / len(quals))**0.5) if quals else 0,
                "min_quality": min(quals) if quals else 0,
                "max_quality": max(quals) if quals else 0,
                "mean_cost": sum(costs) / len(costs) if costs else 0,
                "total_cost": sum(costs),
                "mean_cost_overhead": sum(overheads) / len(overheads) if overheads else 0,
                "mean_quality_gain_per_cost": (
                    sum(gains_per_cost) / len(gains_per_cost) if gains_per_cost else 0
                ),
                "num_runs": len(runs),
                "mean_diversity_gain": (
                    sum(r["diversity_gain"] for r in runs) / len(runs) if runs else 0
                ),
            }

            # Paired Wilcoxon signed-rank test vs baseline
            if tech != "baseline" and baseline_by_task:
                paired_tech = []
                paired_base = []
                for r in runs:
                    bl = baseline_by_task.get(r["task_id"])
                    if bl is not None:
                        paired_tech.append(r["final_quality"])
                        paired_base.append(bl)
                if len(paired_tech) >= 5:
                    try:
                        from scipy.stats import wilcoxon
                        _, p_value = wilcoxon(paired_tech, paired_base, alternative="greater")
                        entry["wilcoxon_p"] = float(p_value)
                        entry["wilcoxon_significant"] = bool(p_value < 0.05)
                    except Exception:
                        pass  # scipy not available

            technique_summary[tech] = entry

        category_summary = {}
        for cat, runs in by_category.items():
            quals = [r["final_quality"] for r in runs]
            qual_ci = self._bootstrap_ci(quals)
            category_summary[cat] = {
                "mean_quality": sum(quals) / len(quals) if quals else 0,
                "quality_ci_95": list(qual_ci),
                "num_runs": len(runs),
            }

        return {
            "by_technique": technique_summary,
            "by_category": category_summary,
            "total_runs": len(self.results),
            "total_cost": sum(r.get("total_cost_usd", 0) for r in self.results),
        }
