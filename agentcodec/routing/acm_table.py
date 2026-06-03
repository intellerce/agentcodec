"""
ACMTableRouter — wraps the hand-coded ACM difficulty-bin table.

This is the original ACM (Adaptive Coding & Modulation) routing logic from
the paper, implemented in :class:`agentcodec.techniques.acm.ACMRouter`.

The library facade typically dispatches the entire ACMRouter as a single
technique (technique="acm") so the difficulty probe runs inside it. This
router class exists for symmetry with the other routers — it doesn't pick a
technique at the library facade level (that's done inside ACMRouter), it
just signals "hand off to the acm technique."
"""

from __future__ import annotations

from typing import Any

from ..models import TaskItem
from .base import RouterDecision


class ACMTableRouter:
    """Sentinel router that signals the dispatcher to use the `acm` technique.

    The actual difficulty probe + bin lookup happens inside
    `agentcodec.techniques.acm.ACMRouter`. This wrapper just makes the
    routing-strategy interface uniform.
    """

    def __init__(
        self,
        table: list[dict[str, Any]] | None = None,
        category_tables: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.table = table
        self.category_tables = category_tables

    def choose(self, task: TaskItem) -> RouterDecision:
        return RouterDecision(
            chosen="acm",
            confidence=1.0,
            router_type="acm_table",
            extra={
                "note": "ACM technique runs its own difficulty probe + bin lookup",
                "has_inline_table": self.table is not None,
                "has_category_tables": self.category_tables is not None,
            },
        )
