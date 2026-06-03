"""
LinearRouter — multinomial logit / per-technique ridge regression.

Loads the JSON artifact produced by the upstream ACM router trainer
(not shipped in the open-source release) when ``router_type`` is
``"linear"`` (ridge or logit). The forward pass is identical for both
training targets — argmax over (X @ W).

This router uses (category, difficulty, optional content features) to pick
a technique. Difficulty is set to 0.5 when no pilot probe is run; when
running through ACMLearnedRouter it's filled from a logprob probe.
"""

from __future__ import annotations

from pathlib import Path

from ..models import TaskItem
from ..techniques.acm_learned import RouterWeights
from .base import RouterDecision


class LinearRouter:
    """Linear logistic / ridge router."""

    def __init__(self, cache_path: str | Path) -> None:
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Linear router weights not found at {cache_path!r}")
        self._weights = RouterWeights.from_path(path)
        self.cache_path = str(path)
        self.technique_classes = list(self._weights.technique_classes)
        self.metadata = dict(self._weights.metadata)

    def choose(self, task: TaskItem) -> RouterDecision:
        category = task.category.value if hasattr(task.category, "value") else str(task.category)
        # No pilot probe in library mode by default — use 0.5 as the prior.
        # Hosts that want a real difficulty estimate should compose with the
        # ACMLearnedRouter technique through the dispatcher (it does the probe).
        difficulty = 0.5
        chosen, confidence = self._weights.predict(category, difficulty, prompt=task.prompt)
        probs = self._weights.predict_proba(category, difficulty, prompt=task.prompt)
        scores = {t: float(p) for t, p in zip(self._weights.technique_classes, probs, strict=False)}
        return RouterDecision(
            chosen=chosen,
            confidence=float(confidence),
            router_type="acm_linear",
            candidates_score=scores,
            extra={
                "trained_on": self.metadata.get("trained_on"),
                "cv_mean_q": self.metadata.get("cv_mean_q"),
                "feature_order": self._weights.feature_order,
            },
        )
