"""
Cost transparency layer.

Every dollar amount the library reports is labeled with a `CostSource` tier
indicating how it was computed. Hosts can log the tier alongside the cost
to know how much trust to place in any given number — and the library will
loudly warn at construction time when a model would fall back to the
default ($2/$8) stub, which is almost certainly wrong.

Tiers (best to worst):
    EXACT_USER_RATE        token counts from API + rate from user config
    EXACT_TABLE_RATE       token counts from API + rate from MODEL_COSTS table
    INFERRED_TABLE_RATE    token counts from API + rate inferred from model name
    DEFAULT_FALLBACK       token counts from API + $2/$8 default (LOUD WARN)
    TOKENS_ESTIMATED       token counts estimated from char heuristics
    THINKING_CHARS_EST     thinking tokens attributed by inline-tag char share

The function `compute_cost()` returns a CostBreakdown that pins down which
tier applied and an explicit list of caveats (prompt caching ignored,
batch discount ignored, etc.) so the host always knows what is NOT modeled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CostSource(str, Enum):
    """How a particular cost number was derived. Lower = tighter accounting."""

    EXACT_USER_RATE = "exact_user_rate"
    OPENROUTER_RATE = "openrouter_rate"
    OPENROUTER_FUZZY_RATE = "openrouter_fuzzy_rate"
    EXACT_TABLE_RATE = "exact_table_rate"
    INFERRED_TABLE_RATE = "inferred_table_rate"
    DEFAULT_FALLBACK = "default_fallback"
    TOKENS_ESTIMATED = "tokens_estimated_from_chars"
    THINKING_CHARS_EST = "tokens_estimated_with_thinking_chars"

    @property
    def is_exact(self) -> bool:
        return self in (
            CostSource.EXACT_USER_RATE,
            CostSource.OPENROUTER_RATE,
            CostSource.EXACT_TABLE_RATE,
        )

    @property
    def is_estimate(self) -> bool:
        return not self.is_exact

    @property
    def rank(self) -> int:
        order = [
            CostSource.EXACT_USER_RATE,
            CostSource.OPENROUTER_RATE,
            CostSource.OPENROUTER_FUZZY_RATE,
            CostSource.EXACT_TABLE_RATE,
            CostSource.INFERRED_TABLE_RATE,
            CostSource.DEFAULT_FALLBACK,
            CostSource.TOKENS_ESTIMATED,
            CostSource.THINKING_CHARS_EST,
        ]
        return order.index(self)


@dataclass
class CostBreakdown:
    """Result of pricing a single LLM call."""
    cost_usd: float
    source: CostSource
    rate_input_per_1m: float
    rate_output_per_1m: float
    caveats: list[str] = field(default_factory=list)

    def merge_worst(self, other: CostBreakdown) -> CostBreakdown:
        """Aggregate with another breakdown, keeping the worst tier and union of caveats."""
        worst = self.source if self.source.rank >= other.source.rank else other.source
        return CostBreakdown(
            cost_usd=self.cost_usd + other.cost_usd,
            source=worst,
            rate_input_per_1m=self.rate_input_per_1m,
            rate_output_per_1m=self.rate_output_per_1m,
            caveats=list({*self.caveats, *other.caveats}),
        )


# Caveats applied to all cloud-API costs by default. Hosts that have
# negotiated discounts or use prompt caching should subtract these in their
# own accounting layer; the library doesn't model them.
_DEFAULT_CAVEATS = (
    "Prompt caching discounts not modeled.",
    "Batch / volume / tier discounts not modeled.",
)


def resolve_rate(
    model: str,
    *,
    user_override: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], CostSource]:
    """Resolve a (input_per_1M, output_per_1M) rate and the tier it came from.

    Resolution order:
      1. user-supplied `cost_per_1m` from config (EXACT_USER_RATE)
      2. OpenRouter live catalog (OPENROUTER_RATE / OPENROUTER_FUZZY_RATE)
      3. MODEL_COSTS table (EXACT_TABLE_RATE)
      4. parameter-count inference from the model name (INFERRED_TABLE_RATE)
      5. default $2/$8 fallback (DEFAULT_FALLBACK)

    Set ``AGENTCODEC_DISABLE_OPENROUTER=1`` to skip step 2 entirely (useful
    in offline / locked-down environments).
    """
    import os

    # 1. User override wins.
    if user_override is not None:
        return user_override, CostSource.EXACT_USER_RATE

    # 2. OpenRouter (disk-cached). Network failures fall through silently.
    if os.environ.get("AGENTCODEC_DISABLE_OPENROUTER") not in ("1", "true", "True"):
        try:
            from . import pricing
            result = pricing.lookup(model)
            if result is not None:
                in_per_m, out_per_m, src = result
                tier = (
                    CostSource.OPENROUTER_RATE
                    if src == "openrouter"
                    else CostSource.OPENROUTER_FUZZY_RATE
                )
                return (in_per_m, out_per_m), tier
        except Exception:  # pragma: no cover -- defensive
            pass

    # 3/4/5 — defer to the channel module's existing logic.
    from .channel import MODEL_COSTS, _infer_cost_from_name

    if model in MODEL_COSTS and model != "default":
        return MODEL_COSTS[model], CostSource.EXACT_TABLE_RATE

    inferred = _infer_cost_from_name(model)
    if inferred is not None:
        return inferred, CostSource.INFERRED_TABLE_RATE

    return MODEL_COSTS["default"], CostSource.DEFAULT_FALLBACK


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    user_override: tuple[float, float] | None = None,
    tokens_estimated: bool = False,
    thinking_chars_estimated: bool = False,
    extra_caveats: list[str] | None = None,
) -> CostBreakdown:
    """Compute the cost for a single LLM call and pin down its provenance.

    Args:
        model:                 the model identifier
        input_tokens:          prompt tokens
        output_tokens:         completion tokens (including any thinking tokens)
        user_override:         optional (input_per_1M, output_per_1M) from config
        tokens_estimated:      True if token counts came from a char heuristic,
                               not the API's usage block
        thinking_chars_estimated: True if thinking attribution was estimated
                               from inline `<think>` tags rather than an API field
        extra_caveats:         additional caveats specific to this call
    """
    rate, source = resolve_rate(model, user_override=user_override)
    cost = (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000.0

    # If token counts are themselves estimates, downgrade the tier accordingly.
    # Thinking-chars estimation is the loosest tier (estimated tokens + estimated
    # attribution share), so it wins.
    if thinking_chars_estimated:
        source = CostSource.THINKING_CHARS_EST
    elif tokens_estimated:
        source = CostSource.TOKENS_ESTIMATED

    caveats = list(_DEFAULT_CAVEATS)
    if source == CostSource.DEFAULT_FALLBACK:
        caveats.append(
            f"Model {model!r} not in MODEL_COSTS and not name-inferable; "
            f"using $2/$8 default — set `cost_per_1m` in the model config."
        )
    if source == CostSource.INFERRED_TABLE_RATE:
        caveats.append(
            f"Rate for {model!r} inferred from parameter count in the name; "
            f"set `cost_per_1m` to make this exact."
        )
    if source == CostSource.OPENROUTER_FUZZY_RATE:
        caveats.append(
            f"Rate for {model!r} resolved via OpenRouter fuzzy token-overlap "
            f"match — verify it points at the intended model, or set "
            f"`cost_per_1m` to pin the rate."
        )
    if tokens_estimated:
        caveats.append(
            "Token counts estimated from character heuristics (~4 chars/token); "
            "actual usage block was unavailable from the backend."
        )
    if thinking_chars_estimated:
        caveats.append(
            "Thinking tokens attributed via inline-tag char share — "
            "the underlying model didn't expose a separate thinking token count."
        )
    if extra_caveats:
        caveats.extend(extra_caveats)

    return CostBreakdown(
        cost_usd=cost,
        source=source,
        rate_input_per_1m=rate[0],
        rate_output_per_1m=rate[1],
        caveats=caveats,
    )


def summarize_pricing(
    models: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a per-model pricing summary and a list of warnings.

    Used at construction time to print the cost-tier table and surface any
    model that would fall back to the default stub. Returns (rows, warnings).
    """
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for m in models:
        name = m.get("model")
        override = m.get("cost_per_1m")
        if override is not None and not isinstance(override, tuple):
            override = (float(override["input"]), float(override["output"]))
        rate, source = resolve_rate(name, user_override=override)
        rows.append({
            "model": name,
            "rate_input_per_1m": rate[0],
            "rate_output_per_1m": rate[1],
            "tier": source.value,
        })
        if source == CostSource.DEFAULT_FALLBACK:
            warnings.append(
                f"Model {name!r} uses DEFAULT_FALLBACK pricing ($2/$8). "
                f"Reported costs will be wrong. "
                f"Add `cost_per_1m: {{input: ..., output: ...}}` to its config."
            )
        elif source == CostSource.INFERRED_TABLE_RATE:
            warnings.append(
                f"Model {name!r} priced by parameter-count inference "
                f"(${rate[0]}/${rate[1]} per 1M). "
                f"Set `cost_per_1m` in config to make pricing exact."
            )
    return rows, warnings
