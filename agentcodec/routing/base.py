"""
Common router protocol + decision dataclass + auto-category classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import TaskCategory, TaskItem


@dataclass
class RouterDecision:
    """What a router decided for a given task.

    Always non-empty. `chosen` is the technique name that should be passed
    to `dispatch.dispatch()`. The rest is telemetry for the trace.
    """
    chosen: str
    confidence: float = 0.0
    router_type: str = ""
    candidates_score: dict[str, float] | None = None
    predicted_quality: float | None = None
    predicted_cost_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.router_type,
            "chosen": self.chosen,
            "confidence": self.confidence,
        }
        if self.candidates_score is not None:
            d["candidates_score"] = self.candidates_score
        if self.predicted_quality is not None:
            d["predicted_quality"] = self.predicted_quality
        if self.predicted_cost_usd is not None:
            d["predicted_cost_usd"] = self.predicted_cost_usd
        if self.extra:
            # Strip large / impl-detail keys from user-facing traces.
            d["extra"] = {
                k: v for k, v in self.extra.items()
                if k not in {"embedding"}
            }
        return d


class Router(Protocol):
    """Every router implements this single method."""
    def choose(self, task: TaskItem) -> RouterDecision: ...


# ---------------------------------------------------------------------------
# Auto-category classifier
# ---------------------------------------------------------------------------

# Cheap, deterministic content-feature signals used both by the linear router
# and by AutoCategoryClassifier. Must stay in lockstep with the upstream
# ACM router trainer (not shipped in the open-source release) and with
# techniques/acm_learned.py.
_CODE_HINTS = (" def ", " class ", "function ", "import ", "return ", "->",
               "()", "; ", "{}")
_MATH_HINTS = (" integral", " derivative", " equation", " prove", "theorem",
               " lemma", " matrix", " vector", " probability", " variance")
_MATH_CHARS = "$=^∫∑∏√≤≥≠"
_CREATIVE_HINTS = (" story", " poem", " write a", " imagine", " describe",
                   " creative", " fictional", " character")


class AutoCategoryClassifier:
    """Lightweight rule-based prompt → TaskCategory classifier.

    Used when `defaults.category: "auto"` in the config and the host doesn't
    pass an explicit category to `run()`. Deterministic — no LLM call.

    Heuristic order (first match wins):
      - code:      contains code fences or code-like keywords
      - reasoning: contains math/proof/theorem hints, OR digits + question mark
      - creative:  contains story/poem/write-a-... hints
      - qa:        the catch-all default

    This is intentionally simple. SemKNN does its own routing on the prompt
    embedding, so it doesn't actually need the category. The classifier is
    here only to give linear/ACM-table routers something sensible.
    """

    def classify(self, prompt: str) -> TaskCategory:
        p = prompt or ""
        p_lower = p.lower()
        if "```" in p or any(kw in p_lower for kw in _CODE_HINTS):
            return TaskCategory.CODE
        has_math = any(c in p for c in _MATH_CHARS) \
                   or any(kw in p_lower for kw in _MATH_HINTS)
        if has_math:
            return TaskCategory.REASONING
        if any(kw in p_lower for kw in _CREATIVE_HINTS):
            return TaskCategory.CREATIVE
        # If the prompt has digits AND a question mark, lean reasoning (math QA).
        if "?" in p and any(ch.isdigit() for ch in p):
            return TaskCategory.REASONING
        return TaskCategory.QA
