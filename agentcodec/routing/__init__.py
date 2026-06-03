"""
Routing strategies for ReliabilityModule.

A `Router` decides which technique to dispatch for a given task. Every
router exposes the same interface:

    router.choose(task) -> RouterDecision

Built-in routers:
    - FixedRouter          : always returns the same technique
    - RemoteSemKNNRouter   : cost-aware semantic-KNN over a remote service
    - LinearRouter         : linear logit / ridge over hand-engineered features
    - ACMTableRouter       : the original hand-coded difficulty-bin table

All routers share the same `RouterDecision` shape so the library facade
doesn't have to special-case them.
"""

from .acm_table import ACMTableRouter
from .base import AutoCategoryClassifier, Router, RouterDecision
from .factory import build_router
from .fixed import FixedRouter
from .linear import LinearRouter
from .remote import (
    RemoteSemKNNRouter,
    canonical_family,
    parse_params_b,
    parse_quant,
)

__all__ = [
    "ACMTableRouter",
    "AutoCategoryClassifier",
    "FixedRouter",
    "LinearRouter",
    "RemoteSemKNNRouter",
    "Router",
    "RouterDecision",
    "build_router",
    "canonical_family",
    "parse_params_b",
    "parse_quant",
]
