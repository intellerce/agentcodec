"""
ReliabilityResult and the streaming event hierarchy.

`ReliabilityResult` is what `ReliabilityModule.run()` and `arun()` return.
Streaming methods (`stream`/`astream`) yield instances of `Event` —
`TokenEvent`, `ProgressEvent`, `WarningEvent`, and exactly one terminal
`FinalEvent` per call carrying the same `ReliabilityResult`.

Default consumption is *minimal*: callers see `text`, `cost_usd`,
`latency_s`, `technique_used`, `thinking_used`. The full per-call trace
(individual outputs, judge calls, router decision, cost-source breakdown)
is opt-in via `return_trace=True` or `result.verbose()`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Base streaming event. Subclasses are dataclasses with explicit fields."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)


@dataclass
class TokenEvent(Event):
    """Incremental text from an in-flight LLM call."""
    text: str = ""
    role: str = "answer"           # "answer" | "thinking" | "critique" | "synthesis" | "verification"
    model: str = ""
    call_id: str = ""              # links to trace["calls"][...]
    cumulative: bool = False       # True only when chunk_format="cumulative"


@dataclass
class ProgressEvent(Event):
    """Structured update about the technique's internal progress."""
    stage: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


@dataclass
class WarningEvent(Event):
    """Out-of-band warning, error, or notable backend behavior."""
    message: str = ""
    code: str = "unknown"          # stable identifier for log aggregation
    severity: str = "warn"         # "info" | "warn" | "error"


@dataclass
class FinalEvent(Event):
    """Terminal event of any stream — carries the final ReliabilityResult."""
    result: ReliabilityResult = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    """The full result of a `ReliabilityModule.run()` call.

    Minimal-mode access: ``.text``, ``.cost_usd``, ``.latency_s``,
    ``.technique_used``, ``.thinking_used``, ``.error``.

    Quality + retraining signals (always populated, independent of
    ``return_trace``):
        ``.final_quality``          — the judge's score for the final answer
        ``.best_individual_quality`` — best score across pre-combine branches
        ``.diversity_gain``         — final - best_individual (combining benefit)
        ``.input_tokens``, ``.output_tokens``, ``.thinking_tokens``,
        ``.judge_cost_usd``, ``.rounds``, ``.num_llm_calls``

    These are the fields the anonymous-telemetry payload reads to build
    the (predicted_quality, observed_quality) pair that drives SemKNN
    retraining — see ``agentcodec.telemetry.build_event_from_result``.

    Verbose access: ``.trace`` (full dict) or ``.verbose()`` (alias).
    """
    text: str = ""
    technique_used: str = ""
    cost_usd: float = 0.0
    cost_source: str = "exact_table_rate"   # worst tier across all calls
    cost_is_estimate: bool = True           # True unless every call was exact_user_rate
    latency_s: float = 0.0                  # wall-clock
    cumulative_latency_s: float = 0.0       # sum of per-call latencies
    thinking_used: bool = False             # any call emitted thinking
    thinking_text: str | None = None        # captured reasoning across all calls (None when no call emitted thinking)
    thinking_cost_usd: float = 0.0          # share of cost_usd attributable to thinking tokens
    error: str | None = None                # populated on fallback_baseline

    # --- Quality / retraining signals (always populated) ---
    final_quality: float | None = None          # judge score for combined output
    best_individual_quality: float | None = None
    diversity_gain: float | None = None          # final - best_individual

    # --- Token + cost rollups (always populated) ---
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    judge_cost_usd: float = 0.0
    rounds: int = 0
    num_llm_calls: int = 0

    # Full trace — populated when return_trace=True at .run().
    trace: dict[str, Any] = field(default_factory=dict)

    def verbose(self) -> dict[str, Any]:
        """Return the full trace dict. Equivalent to `.trace`."""
        return self.trace

    def cost_caveats(self) -> list[str]:
        """Distinct list of all caveats across all calls."""
        return list(self.trace.get("totals", {}).get("all_caveats_distinct", []))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole result (minimal + trace) to a JSON-friendly dict."""
        return {
            "text": self.text,
            "technique_used": self.technique_used,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "cost_is_estimate": self.cost_is_estimate,
            "latency_s": self.latency_s,
            "cumulative_latency_s": self.cumulative_latency_s,
            "thinking_used": self.thinking_used,
            "thinking_text": self.thinking_text,
            "thinking_cost_usd": self.thinking_cost_usd,
            "error": self.error,
            "final_quality": self.final_quality,
            "best_individual_quality": self.best_individual_quality,
            "diversity_gain": self.diversity_gain,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "judge_cost_usd": self.judge_cost_usd,
            "rounds": self.rounds,
            "num_llm_calls": self.num_llm_calls,
            "trace": self.trace,
        }


# ---------------------------------------------------------------------------
# Helpers — used by api.py to build a ReliabilityResult from a ReliabilityRun
# ---------------------------------------------------------------------------

def _per_call_dict(o, role: str) -> dict[str, Any]:
    """Convert an AgentOutput to a serializable per-call dict for the trace."""
    d: dict[str, Any] = {
        "role": role,
        "model": getattr(o, "model", None),
        "temperature": getattr(o, "temperature", None),
        "latency_s": getattr(o, "latency_s", None),
        "input_tokens": getattr(o, "input_tokens", 0) or 0,
        "output_tokens": getattr(o, "output_tokens", 0) or 0,
        "token_count": getattr(o, "token_count", 0) or 0,
        "cost_usd": getattr(o, "cost_usd", 0.0) or 0.0,
        "cost_source": getattr(o, "cost_source", None),
        "cost_caveats": list(getattr(o, "cost_caveats", []) or []),
        "rate_input_per_1m": getattr(o, "rate_input_per_1m", None),
        "rate_output_per_1m": getattr(o, "rate_output_per_1m", None),
        "finish_reason": getattr(o, "finish_reason", None),
        "mean_logprob": getattr(o, "mean_logprob", None),
        "quality_score": getattr(o, "quality_score", None),
        "prompt_variant": getattr(o, "prompt_variant", "default"),
        "thinking": {
            "supported": getattr(o, "thinking_supported", False),
            "enabled": getattr(o, "thinking_enabled", False),
            "emitted": getattr(o, "thinking_emitted", False),
            "tokens": getattr(o, "thinking_tokens", 0) or 0,
            "tokens_source": getattr(o, "thinking_tokens_source", None),
            "chars": getattr(o, "thinking_chars", 0) or 0,
            "cost_usd": getattr(o, "thinking_cost_usd", 0.0) or 0.0,
            "text": getattr(o, "thinking_text", None),
        },
        "answer_tokens": getattr(o, "answer_tokens", 0) or 0,
        "answer_cost_usd": getattr(o, "answer_cost_usd", 0.0) or 0.0,
        "warnings": list(getattr(o, "backend_warnings", []) or []),
    }
    # Judge calls carry the parsed yes/no checklist on their metadata; surface
    # it here so callers can render exactly which criteria drove the score.
    md = getattr(o, "metadata", None) or {}
    if "checklist" in md:
        d["checklist"] = md["checklist"]
    text = getattr(o, "text", "") or ""
    d["text_preview"] = (text[:200] + "...") if len(text) > 200 else text
    return d


def build_result_from_run(
    run,
    *,
    technique_used: str,
    wall_clock_s: float,
    return_trace: bool = False,
    error: str | None = None,
    routing_info: dict[str, Any] | None = None,
    category_info: dict[str, Any] | None = None,
    extra_warnings: list[dict[str, Any]] | None = None,
) -> ReliabilityResult:
    """Convert a `ReliabilityRun` into the library's `ReliabilityResult`.

    Aggregates cost-source tiers across all calls, computes the worst tier,
    and unions the caveats list. The minimal fields are always populated;
    `trace` is filled only when `return_trace=True`.
    """
    from .cost import CostSource

    text = getattr(run, "combined_output", "") or ""
    if not text and getattr(run, "individual_outputs", None):
        text = run.individual_outputs[0].text

    # Aggregate cost-source telemetry across all calls.
    all_outputs = list(getattr(run, "individual_outputs", []) or []) \
                + list(getattr(run, "overhead_outputs", []) or []) \
                + list(getattr(run, "judge_outputs", []) or [])

    worst_rank = -1
    worst_source = "exact_table_rate"
    breakdown: dict[str, float] = {}
    caveats_distinct: set[str] = set()
    cumulative_latency = 0.0
    thinking_used = False
    thinking_total_tokens = 0
    thinking_total_cost = 0.0

    thinking_texts: list[str] = []

    for o in all_outputs:
        cs = getattr(o, "cost_source", None) or "exact_table_rate"
        try:
            rank = CostSource(cs).rank
        except ValueError:
            rank = 99
        if rank > worst_rank:
            worst_rank = rank
            worst_source = cs
        breakdown[cs] = breakdown.get(cs, 0.0) + (getattr(o, "cost_usd", 0.0) or 0.0)
        for c in getattr(o, "cost_caveats", []) or []:
            caveats_distinct.add(c)
        cumulative_latency += getattr(o, "latency_s", 0.0) or 0.0
        if getattr(o, "thinking_emitted", False):
            thinking_used = True
        thinking_total_tokens += getattr(o, "thinking_tokens", 0) or 0
        thinking_total_cost += getattr(o, "thinking_cost_usd", 0.0) or 0.0
        tt = getattr(o, "thinking_text", None)
        if tt:
            thinking_texts.append(tt)

    aggregated_thinking_text = "\n\n---\n\n".join(thinking_texts) if thinking_texts else None

    cost_is_estimate = worst_source not in {CostSource.EXACT_USER_RATE.value}

    # Token rollup — same shape we'd populate in the trace, but lifted to
    # the top-level so telemetry can read them without inspecting trace.
    input_tokens = sum(
        getattr(o, "input_tokens", 0) or 0 for o in all_outputs
    )
    output_tokens = sum(
        getattr(o, "output_tokens", 0) or 0 for o in all_outputs
    )

    result = ReliabilityResult(
        text=text,
        technique_used=technique_used,
        cost_usd=getattr(run, "total_cost_usd", 0.0) or 0.0,
        cost_source=worst_source,
        cost_is_estimate=cost_is_estimate,
        latency_s=wall_clock_s,
        cumulative_latency_s=cumulative_latency,
        thinking_used=thinking_used,
        thinking_text=aggregated_thinking_text,
        thinking_cost_usd=thinking_total_cost,
        error=error,
        # Quality / retraining signals — these are the (predicted, observed)
        # pair half. SemKNN already gives the prediction at /route time;
        # `final_quality` is the user's judge's verdict after dispatch.
        final_quality=getattr(run, "final_quality", None),
        best_individual_quality=getattr(run, "best_individual_quality", None),
        diversity_gain=getattr(run, "diversity_gain", None),
        # Token / call rollups
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_total_tokens,
        judge_cost_usd=getattr(run, "judge_cost_usd", 0.0) or 0.0,
        rounds=getattr(run, "rounds", 0) or 0,
        num_llm_calls=getattr(run, "num_llm_calls", 0) or 0,
    )

    if return_trace:
        result.trace = {
            "technique_used": technique_used,
            "rounds": result.rounds,
            "router": routing_info or {},
            "category": category_info or {},
            "totals": {
                "wall_clock_s": wall_clock_s,
                "cumulative_latency_s": cumulative_latency,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "thinking_tokens": result.thinking_tokens,
                "cost_usd": result.cost_usd,
                "thinking_cost_usd": thinking_total_cost,
                "judge_cost_usd": result.judge_cost_usd,
                "num_llm_calls": result.num_llm_calls,
                "cost_source_breakdown": breakdown,
                "weakest_tier": worst_source,
                "all_caveats_distinct": sorted(caveats_distinct),
            },
            "calls": (
                [_per_call_dict(o, "channel") for o in (run.individual_outputs or [])]
                + [_per_call_dict(o, "overhead") for o in (run.overhead_outputs or [])]
                + [_per_call_dict(o, "judge") for o in (run.judge_outputs or [])]
            ),
            "config_snapshot": dict(getattr(run, "config", {}) or {}),
            "warnings": list(extra_warnings or []),
            "final_quality": result.final_quality,
            "best_individual_quality": result.best_individual_quality,
            "diversity_gain": result.diversity_gain,
        }

    return result
