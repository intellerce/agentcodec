"""
FixedRouter — always returns the same technique.

Used when ``strategy.type == "fixed"`` in the library config. Trivial
implementation; exists so the library facade can treat fixed and routed
strategies uniformly.
"""

from __future__ import annotations

from ..models import TaskItem
from .base import RouterDecision


class FixedRouter:
    """Always picks the same technique."""

    def __init__(self, technique: str) -> None:
        self.technique = technique

    def choose(self, task: TaskItem) -> RouterDecision:
        return RouterDecision(
            chosen=self.technique,
            confidence=1.0,
            router_type="fixed",
        )
