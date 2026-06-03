"""
Data models for the AgentCodec benchmark system.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .messages import ChatRequest, ToolCall


class CombiningStrategy(str, Enum):
    SC = "sc"       # Selection Combining — pick best
    MRC = "mrc"     # Maximal Ratio Combining — quality-weighted synthesis
    EGC = "egc"     # Equal Gain Combining — equal-weight consensus


class HARQMode(str, Enum):
    CC = "cc"   # Chase Combining — retry + combine equally
    IR = "ir"   # Incremental Redundancy — retry with critic feedback


class TaskCategory(str, Enum):
    QA = "qa"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CODE = "code"


@dataclass
class TaskItem:
    """A single benchmark task."""
    id: str
    category: TaskCategory
    prompt: str
    # Full provider-neutral request shape. When None, populated in
    # __post_init__ from `prompt` so legacy call sites (and the benchmark
    # runner) keep working unchanged. Techniques that need to mutate the
    # user-facing prompt while preserving system / history / tools call
    # `task.request.with_user(new_text)`.
    request: ChatRequest | None = None
    reference: str | None = None       # ground truth if available
    metadata: dict[str, Any] = field(default_factory=dict)
    # Objective checks: list of (substring_or_regex, weight) pairs.
    # If provided, these are used to VERIFY the judge score against ground truth.
    # Each check tests whether the output contains a required fact.
    # Score = weighted proportion of checks that pass.
    objective_checks: list[tuple[str, float]] | None = None
    # Deterministic scoring mode. When set, QualityScorer pairs the LLM judge
    # with an exact / tolerance / relaxed check on the structured reference.
    # None = legacy pure-judge behavior (15-criterion sigma-delta checklist,
    # continuous). See agentcodec/scoring.py for the supported modes and
    # agentcodec/channel.py:QualityScorer for how score_strategy combines them:
    #   "exact_letter"  — multi-choice: extract A-J, compare to reference
    #   "exact_match"   — case-insensitive normalized string equality
    #   "yes_no"        — yes/no extraction + match
    #   "numeric"       — parse number, exact match modulo formatting
    #   "relaxed"       — ChartQA-style: numeric within 5%, else string equality
    #   "judge"         — continuous LLM judge (same as None; explicit opt-in)
    score_mode: str | None = None

    def __post_init__(self) -> None:
        # Infer score_mode from metadata.source when not set explicitly.
        # Covers the footgun where a downloaded JSON cache predates the
        # score_mode field, or a user constructs a TaskItem with
        # metadata={"source": "mmlu", ...} but forgets to set score_mode.
        # An explicit value always wins; inference only fills the gap.
        if self.score_mode is None:
            from .scoring import infer_score_mode_from_metadata
            self.score_mode = infer_score_mode_from_metadata(self.metadata)
        # Auto-derive a ChatRequest from the plain prompt when one wasn't
        # supplied. This keeps every legacy TaskItem(prompt=...) construction
        # working, while exposing a uniform `task.request` to techniques
        # that need system / history / tool context.
        if self.request is None:
            from .messages import ChatRequest
            self.request = ChatRequest.from_prompt(self.prompt)

    def verify_objective(self, output: str) -> float | None:
        """
        Run objective checks against an output. Returns a score [0,1] based on
        what proportion of required facts are present, or None if no checks defined.
        """
        if not self.objective_checks:
            return None
        import re
        total_weight = sum(w for _, w in self.objective_checks)
        if total_weight == 0:
            return None
        earned = 0.0
        for pattern, weight in self.objective_checks:
            # Try as regex first, fall back to substring
            try:
                if re.search(pattern, output, re.IGNORECASE):
                    earned += weight
            except re.error:
                if pattern.lower() in output.lower():
                    earned += weight
        return earned / total_weight


@dataclass
class AgentOutput:
    """A single LLM agent output."""
    text: str
    model: str
    temperature: float
    prompt_variant: str = "default"
    quality_score: float = 0.0
    latency_s: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0
    # Soft-output fields: token-level log-probabilities from the LLM.
    # Populated when AgentChannel.transmit() is called with request_logprobs=True.
    # None means logprobs were not requested or not available (e.g. Anthropic).
    token_logprobs: list[float] | None = None
    mean_logprob: float | None = None
    # Per-position top-k alternatives, one dict per generated token mapping
    # the alternative token string to its logprob. Populated only when the
    # backend exposes top_logprobs (OpenAI-compat, Ollama-python). Required
    # for CISC's P(True) confidence extraction, which needs the logprobs
    # assigned to the literal tokens "0" and "1" at a specific position.
    top_logprobs_per_token: list[dict[str, float]] | None = None

    # --- Cost transparency (set by AgentChannel.transmit) ---
    # Tier indicating how cost_usd was derived. See agentcodec.cost.CostSource.
    # None on legacy outputs that pre-date the cost-source layer.
    cost_source: str | None = None
    # Resolved per-1M-token rates that produced cost_usd.
    rate_input_per_1m: float | None = None
    rate_output_per_1m: float | None = None
    # Human-readable list of what is NOT modeled in cost_usd
    # (e.g. prompt caching, batch discounts).
    cost_caveats: list[str] = field(default_factory=list)
    # Token breakdown — explicit so callers don't have to back it out of
    # token_count = input_tokens + output_tokens.
    input_tokens: int = 0
    output_tokens: int = 0

    # --- Thinking telemetry (populated when the backend exposes a separate
    # reasoning channel or when inline <think>...</think> tags are stripped) ---
    thinking_supported: bool = False    # the model family supports thinking
    thinking_enabled: bool = False      # the call asked for thinking
    thinking_emitted: bool = False      # the model actually produced thinking
    thinking_text: str | None = None    # raw thinking content (truncated by callers if needed)
    thinking_chars: int = 0             # char count of thinking text
    thinking_tokens: int = 0            # token attribution (exact when API exposes it)
    thinking_tokens_source: str | None = None  # see AgentChannel for set values
    thinking_cost_usd: float = 0.0      # share of cost_usd attributable to thinking
    answer_tokens: int = 0              # output_tokens - thinking_tokens
    answer_cost_usd: float = 0.0        # cost of the final answer portion
    finish_reason: str | None = None    # backend's reported finish reason
    backend_warnings: list[str] = field(default_factory=list)

    # --- Tool calling (set by transports that surface tool_use responses) ---
    # Provider-neutral tool-call list. None when the model produced no
    # tool calls or when tools weren't requested.
    tool_calls: tuple[ToolCall, ...] | None = None
    # Optional escape hatch: the raw provider response object as a dict.
    # Populated by the compat shims so callers can recover provider-specific
    # fields the neutral abstraction doesn't model. Deliberately NOT
    # serialized into telemetry to avoid PII leakage.
    raw_provider_response: dict[str, Any] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityRun:
    """Record of a single technique execution on a single task."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    task_category: str = ""
    technique: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    # Outputs
    individual_outputs: list[AgentOutput] = field(default_factory=list)
    overhead_outputs: list[AgentOutput] = field(default_factory=list)  # synthesis, critic, decode calls
    judge_outputs: list[AgentOutput] = field(default_factory=list)     # judge/scorer LLM calls
    combined_output: str = ""
    final_quality: float = 0.0

    # Metrics
    best_individual_quality: float = 0.0
    mean_individual_quality: float = 0.0
    diversity_gain: float = 0.0         # final - best_individual
    coding_gain: float = 0.0            # final - uncoded_baseline (set later)
    num_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    judge_cost_usd: float = 0.0        # judge calls cost (subset of total)
    rounds: int = 0                     # for iterative techniques
    # Per-bucket breakdowns so the cache is exactly repriceable later.
    # individual_* covers channel calls; overhead_* covers synthesis/critic/decode;
    # judge_* covers score() and score_comparative() transmits.
    individual_tokens: int = 0
    individual_cost_usd: float = 0.0
    overhead_tokens: int = 0
    overhead_cost_usd: float = 0.0
    judge_tokens: int = 0
    # Per-bucket-by-model breakdowns: model_name -> (tokens, cost_usd). Lets
    # downstream tools reprice without assuming one model per bucket.
    individual_by_model: dict[str, tuple[int, float]] = field(default_factory=dict)
    overhead_by_model: dict[str, tuple[int, float]] = field(default_factory=dict)
    judge_by_model: dict[str, tuple[int, float]] = field(default_factory=dict)

    # Normalized metrics (set after baseline is known)
    baseline_cost_usd: float = 0.0       # baseline cost for the same task
    baseline_quality: float = 0.0        # baseline quality for the same task
    cost_overhead: float = 0.0           # total_cost / baseline_cost (1.0 = same as baseline)
    quality_gain_per_cost: float = 0.0   # (quality - baseline_quality) / (cost - baseline_cost)
    cost_efficiency: float = 0.0         # final_quality / total_cost_usd

    # Timestamps
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def compute_metrics(self):
        """Compute derived metrics from individual outputs."""
        all_outputs = self.individual_outputs + self.overhead_outputs + self.judge_outputs
        if self.individual_outputs:
            scores = [o.quality_score for o in self.individual_outputs]
            self.best_individual_quality = max(scores)
            self.mean_individual_quality = sum(scores) / len(scores)

            # Safety floor: if synthesis/decoding catastrophically degraded quality,
            # fall back to the best individual output. This prevents the combiner
            # from destroying good answers — analogous to how real receivers fall
            # back to the strongest branch if combining fails.
            if self.final_quality < self.best_individual_quality - 0.15:
                best_output = max(self.individual_outputs, key=lambda o: o.quality_score)
                self.combined_output = best_output.text
                self.final_quality = self.best_individual_quality

            self.diversity_gain = self.final_quality - self.best_individual_quality
        if all_outputs:
            self.num_llm_calls = len(all_outputs)
            self.total_tokens = sum(o.token_count for o in all_outputs)
            self.total_cost_usd = sum(o.cost_usd for o in all_outputs)
            self.total_latency_s = sum(o.latency_s for o in all_outputs)
        # Per-bucket breakdowns for exact downstream repricing.
        def _aggregate(outs):
            by_model: dict[str, list[float]] = {}
            for o in outs:
                t, c = by_model.setdefault(o.model, [0, 0.0])
                by_model[o.model] = [t + o.token_count, c + o.cost_usd]
            tokens = sum(t for t, _ in by_model.values())
            cost = sum(c for _, c in by_model.values())
            return tokens, cost, {m: (int(t), float(c)) for m, (t, c) in by_model.items()}
        self.individual_tokens, self.individual_cost_usd, self.individual_by_model = _aggregate(self.individual_outputs)
        self.overhead_tokens, self.overhead_cost_usd, self.overhead_by_model = _aggregate(self.overhead_outputs)
        if self.judge_outputs:
            jt, jc, jbm = _aggregate(self.judge_outputs)
            self.judge_tokens = jt
            self.judge_cost_usd = jc
            self.judge_by_model = jbm
        # Cost efficiency (quality per dollar)
        if self.total_cost_usd > 0:
            self.cost_efficiency = self.final_quality / self.total_cost_usd
        self.finished_at = time.time()

    def set_baseline(self, baseline_cost: float, baseline_quality: float):
        """Set baseline reference for normalized cost metrics."""
        self.baseline_cost_usd = baseline_cost
        self.baseline_quality = baseline_quality
        if baseline_cost > 0:
            self.cost_overhead = self.total_cost_usd / baseline_cost
        quality_delta = self.final_quality - baseline_quality
        cost_delta = self.total_cost_usd - baseline_cost
        if cost_delta > 0:
            self.quality_gain_per_cost = quality_delta / cost_delta
        else:
            self.quality_gain_per_cost = 0.0
