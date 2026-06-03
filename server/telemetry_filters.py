"""
Server-side telemetry ingest filters — lightweight edition.

Why this file exists
--------------------
The client SDK at ``agentcodec/telemetry.py`` POSTs ``observed_quality``,
``embedding``, ``observed_cost_usd``, latency, token counts, etc. to
the collector. Everything past the wire is *client-supplied*: a hostile
sender can fabricate any of those fields while still passing the wire
schema. Without filtering, fabricated events poison the SemKNN
q-matrix.

This module runs as the first thing the ingest endpoint does after
JSON decoding. It returns ``(decision, weight, reasons)``:

  * ``decision == REJECT``  → drop the event, increment your abuse counter
    keyed by ``session_id``, and move on.
  * ``decision == ACCEPT``  → forward to the q-matrix updater, but
    multiply ``weight`` into the per-event influence so soft-suspicious
    events contribute less than fully trusted ones.

Designed for: one small process. All state is in-memory and bounded:
``RobustStats`` is capped at ``window`` samples per (technique, field)
cell; ``SessionRateLimiter`` is an LRU of at most ``max_sessions``
entries. Total memory overhead is a few MB even with thousands of
techniques. Per-event cost is in microseconds; no I/O.

Pipeline (run in order; first REJECT short-circuits)
----------------------------------------------------
  1. f_schema          required fields present, numeric ranges sane.
  2. f_embedding       BGE dim, unit norm, no degenerate vectors.
  3. f_rate            per-session token bucket (in-memory).
  4. f_throughput      output_tokens / latency_s within physical range.
  5. f_cost            cost vs token count within market envelope.
  6. f_outlier         robust z-score on (quality, cost, latency)
                       against rolling per-technique median/MAD.

What this DOES catch
--------------------
  * Schema-valid garbage (random floats in the right ranges).
  * Embeddings that aren't BGE outputs (wrong dim, not unit-norm,
    all-zero, all-constant).
  * Bursty single-source spam (rate limiter).
  * Physically impossible (tokens, cost, latency) combinations.
  * Statistical outliers vs the population of legitimate traffic
    for the same ``technique_used``.

What this does NOT catch, and the upgrade path
----------------------------------------------
  1. **Cross-process Sybil attacks**
     Today's ``SessionRateLimiter`` is per-process. An attacker who
     rotates ``session_id`` and spreads load across IPs slips past.
     Upgrade: replace ``SessionRateLimiter`` with a Redis-backed
     token bucket keyed by (session_id, src_ip), and add a separate
     limiter on (src_ip, num_distinct_sessions_per_minute) to catch
     rotation. Keep this module's signature; just swap the impl.

  2. **Canary tasks**
     The strongest single signal against quality fabrication is
     "you said 0.95 on a task whose ground truth is 0.20". This
     requires a curated embedding corpus + known quality ranges.
     Upgrade: add ``f_canary`` that scores cosine similarity of
     ``ev['embedding']`` against the canary index; if the top match
     exceeds ~0.92 and the reported quality is outside the known
     range, REJECT and flag the session_id.

  3. **Quarantine + offline review**
     Right now ACCEPT-with-low-weight just dilutes the update. On a
     larger server, route any event with ``weight < 0.5`` into a
     quarantine table; a nightly job promotes or burns batches based
     on canary agreement and downstream q-matrix delta.

  4. **HMAC-signed envelopes / per-install API keys**
     Cuts off spoofing at the wire. Add when the public collector
     starts attracting attention. Until then, plausibility filters
     + robust aggregation are the load-bearing defenses, not
     authentication — a real install can lie too.

  5. **Influence caps in the q-matrix updater**
     Filters live here; influence caps live in the aggregator. Even
     a fully ACCEPTed event must not be able to move a q-matrix cell
     by more than some ε (e.g. EMA with decay < 0.02). Without that
     cap, a whitelisted but compromised install still wins.

  6. **Cross-process distribution learning**
     ``RobustStats`` is per-process. Once you scale to multiple boxes,
     periodically dump (median, MAD) per cell to Redis and merge on
     load. Until then, each box bootstraps its own population — which
     is fine but means cold starts are unfiltered for the first ~32
     events per technique.

  7. **Technique-shape rules**
     Each AgentCodec technique (harq_ir, diversity_mrc, acm, fountain,
     turbo, soft, fec, baseline) has a structural signature: number
     of rounds, num_llm_calls, presence of judge_cost. Once you have
     enough traffic to know the real fingerprints, add ``f_shape``
     that rejects events whose declared technique doesn't match
     its declared shape.

Tuning knobs
------------
All thresholds are constants at the top of each filter; raise them if
you see legitimate-looking rejects in the logs, lower them if you see
poisoning getting through. The robust-outlier ``Z_HARD`` of 6 is
intentionally permissive — a one-tailed 6-sigma outlier is essentially
never legitimate, so false positives are rare; tighten to 5 if you
want to be more aggressive.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class FilterResult:
    decision: Decision
    reason: str = ""
    weight: float = 1.0


Filter = Callable[[dict, "PipelineState"], FilterResult]


# ---------------------------------------------------------------------------
# In-memory state (bounded)
# ---------------------------------------------------------------------------


class RobustStats:
    """Per-(technique, field) rolling median + MAD over a bounded window.

    Recompute is amortized: median/MAD is cached and only rebuilt every
    ``recompute_every`` observations. Worst-case O(n log n) where n is
    capped at ``window``; expected O(1) per ``observe`` call.
    """

    def __init__(self, window: int = 512, recompute_every: int = 32,
                 min_samples: int = 32) -> None:
        self.window = window
        self.recompute_every = recompute_every
        self.min_samples = min_samples
        self._buf: dict[tuple[str, str], deque[float]] = {}
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._dirty: dict[tuple[str, str], int] = {}

    def observe(self, technique: str, field_name: str, value: float) -> None:
        key = (technique, field_name)
        d = self._buf.get(key)
        if d is None:
            d = deque(maxlen=self.window)
            self._buf[key] = d
        d.append(value)
        self._dirty[key] = self._dirty.get(key, 0) + 1

    def median_mad(self, technique: str, field_name: str) -> tuple[float, float] | None:
        key = (technique, field_name)
        d = self._buf.get(key)
        if d is None or len(d) < self.min_samples:
            return None
        if self._dirty.get(key, 0) >= self.recompute_every or key not in self._cache:
            xs = sorted(d)
            n = len(xs)
            med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
            devs = sorted(abs(x - med) for x in xs)
            mad = devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
            self._cache[key] = (med, mad)
            self._dirty[key] = 0
        return self._cache[key]


class SessionRateLimiter:
    """Per-session token bucket. LRU-bounded so it cannot OOM.

    On a single-process server this is enough to bound the rate at
    which one ``session_id`` can poison the population. For multi-box
    deploys, swap for a Redis-backed bucket.
    """

    def __init__(self, max_sessions: int = 10_000,
                 capacity: float = 60.0, refill_per_s: float = 1.0) -> None:
        self.max_sessions = max_sessions
        self.capacity = capacity
        self.refill = refill_per_s
        # session_id -> (tokens_left, last_seen_monotonic)
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, session_id: str, now: float) -> bool:
        b = self._buckets.get(session_id)
        if b is None:
            self._buckets[session_id] = (self.capacity - 1.0, now)
            if len(self._buckets) > self.max_sessions:
                self._buckets.popitem(last=False)
            return True
        tokens, last = b
        tokens = min(self.capacity, tokens + (now - last) * self.refill)
        if tokens < 1.0:
            self._buckets[session_id] = (tokens, now)
            self._buckets.move_to_end(session_id)
            return False
        self._buckets[session_id] = (tokens - 1.0, now)
        self._buckets.move_to_end(session_id)
        return True


@dataclass
class PipelineState:
    rolling: RobustStats = field(default_factory=RobustStats)
    rate: SessionRateLimiter = field(default_factory=SessionRateLimiter)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


_REQUIRED = (
    "schema_version", "session_id", "ts_iso", "technique_used",
    "latency_s", "embedding", "observed_quality",
)
_NUMERIC = (
    "latency_s", "wall_clock_s", "cumulative_latency_s",
    "observed_cost_usd", "judge_cost_usd",
    "input_tokens", "output_tokens", "thinking_tokens",
    "rounds", "num_llm_calls",
)


def f_schema(ev: dict, _state: PipelineState) -> FilterResult:
    for k in _REQUIRED:
        if ev.get(k) is None:
            return FilterResult(Decision.REJECT, f"missing:{k}")
    q = ev["observed_quality"]
    if not (isinstance(q, (int, float)) and 0.0 <= q <= 1.0):
        return FilterResult(Decision.REJECT, "quality_oor")
    for k in _NUMERIC:
        v = ev.get(k)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or v < 0 or not math.isfinite(v):
            return FilterResult(Decision.REJECT, f"{k}_invalid")
    if ev["latency_s"] > 3600:
        return FilterResult(Decision.REJECT, "latency_too_high")
    if (ev.get("input_tokens") or 0) + (ev.get("output_tokens") or 0) > 10_000_000:
        return FilterResult(Decision.REJECT, "tokens_too_high")
    return FilterResult(Decision.ACCEPT)


_EMBED_DIMS = frozenset({384, 768, 1024})


def f_embedding(ev: dict, _state: PipelineState) -> FilterResult:
    e = ev["embedding"]
    if not isinstance(e, list) or len(e) not in _EMBED_DIMS:
        return FilterResult(Decision.REJECT, "embedding_dim")
    s = 0.0
    total = 0.0
    for x in e:
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return FilterResult(Decision.REJECT, "embedding_nonfinite")
        s += x * x
        total += x
    norm = math.sqrt(s)
    if not (0.9 <= norm <= 1.1):
        return FilterResult(Decision.REJECT, f"embedding_norm:{norm:.3f}")
    mean = total / len(e)
    var = sum((x - mean) ** 2 for x in e) / len(e)
    # Healthy BGE vectors have per-coord variance ~ 1/d ≈ 1e-3.
    # All-zero or all-constant attacks land at var ≈ 0.
    if var < 1e-6:
        return FilterResult(Decision.REJECT, "embedding_degenerate")
    return FilterResult(Decision.ACCEPT)


def f_rate(ev: dict, state: PipelineState) -> FilterResult:
    if not state.rate.allow(ev["session_id"], time.monotonic()):
        return FilterResult(Decision.REJECT, "session_rate")
    return FilterResult(Decision.ACCEPT)


# Throughput envelope: a single LLM call can't produce >10k output tok/s
# (above 5k tok/s is already top-of-line speculative-decoding territory),
# and won't produce <0.5 tok/s without something else being wrong.
_TPS_MIN = 0.5
_TPS_MAX = 10_000.0


def f_throughput(ev: dict, _state: PipelineState) -> FilterResult:
    lat = ev.get("latency_s") or 0.0
    out = ev.get("output_tokens") or 0
    if lat <= 0 or out <= 0:
        return FilterResult(Decision.ACCEPT)
    tps = out / lat
    if tps < _TPS_MIN or tps > _TPS_MAX:
        # Soft signal — could be a weird local model. Don't reject;
        # discount instead so the aggregator weighs it less.
        return FilterResult(Decision.ACCEPT, f"tps:{tps:.1f}", weight=0.3)
    return FilterResult(Decision.ACCEPT)


# Cost envelope: cheapest known per-token price floor, priciest ceiling,
# each widened 10x to absorb future model launches and judge surcharges
# without code changes. Outside these bounds → almost certainly fabricated.
_COST_PER_TOK_FLOOR = 5e-9      # 10x below $0.05/Mtok
_COST_PER_TOK_CEIL = 7.5e-4     # 10x above $75/Mtok


def f_cost(ev: dict, _state: PipelineState) -> FilterResult:
    cost = ev.get("observed_cost_usd")
    in_t = ev.get("input_tokens") or 0
    out_t = ev.get("output_tokens") or 0
    if cost is None or in_t + out_t == 0:
        return FilterResult(Decision.ACCEPT)
    total = in_t + out_t
    if cost < total * _COST_PER_TOK_FLOOR or cost > total * _COST_PER_TOK_CEIL:
        return FilterResult(Decision.ACCEPT, f"cost_envelope:{cost:.6f}", weight=0.3)
    # Judge is typically a single cheap call; if it's ≥5x the main run,
    # someone's numbers are upside-down.
    jc = ev.get("judge_cost_usd") or 0.0
    if jc > cost * 5:
        return FilterResult(Decision.ACCEPT, "judge_cost_inverted", weight=0.3)
    return FilterResult(Decision.ACCEPT)


# Robust-outlier thresholds. z is scaled by 1.4826*MAD ≈ σ for Gaussian.
_Z_HARD = 6.0   # reject
_Z_SOFT = 3.5   # accept with reduced weight


def f_outlier(ev: dict, state: PipelineState) -> FilterResult:
    tech = ev["technique_used"]
    worst = 0.0
    worst_field = ""
    # Update first, then check — bootstrapping period is unfiltered, which
    # matches the documented behavior.
    for k in ("observed_quality", "observed_cost_usd", "latency_s"):
        v = ev.get(k)
        if v is None:
            continue
        stats = state.rolling.median_mad(tech, k)
        state.rolling.observe(tech, k, float(v))
        if stats is None:
            continue
        med, mad = stats
        if mad <= 0:
            continue
        z = abs(v - med) / (1.4826 * mad)
        if z > worst:
            worst = z
            worst_field = k
    if worst >= _Z_HARD:
        return FilterResult(Decision.REJECT, f"z:{worst_field}={worst:.1f}")
    if worst >= _Z_SOFT:
        return FilterResult(
            Decision.ACCEPT, f"z:{worst_field}={worst:.1f}",
            weight=max(0.1, 1.0 - worst / _Z_HARD),
        )
    return FilterResult(Decision.ACCEPT)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


DEFAULT_PIPELINE: tuple[Filter, ...] = (
    f_schema, f_embedding, f_rate, f_throughput, f_cost, f_outlier,
)


def run_pipeline(
    ev: dict[str, Any],
    state: PipelineState,
    pipeline: tuple[Filter, ...] = DEFAULT_PIPELINE,
) -> tuple[Decision, float, list[str]]:
    """Run filters in order, REJECT short-circuits.

    Returns:
        decision: ACCEPT or REJECT.
        weight:   product of per-filter weights in (0, 1]; multiply into
                  the per-event influence in the q-matrix updater.
        reasons:  human-readable trail, useful for dashboards & logs.
    """
    weight = 1.0
    reasons: list[str] = []
    for f in pipeline:
        r = f(ev, state)
        if r.reason:
            reasons.append(f"{f.__name__[2:]}:{r.reason}")
        weight *= r.weight
        if r.decision == Decision.REJECT:
            return Decision.REJECT, 0.0, reasons
    return Decision.ACCEPT, weight, reasons
