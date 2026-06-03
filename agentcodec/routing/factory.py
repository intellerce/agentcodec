"""
Factory for building a Router from a LibraryConfig.

Takes the full LibraryConfig (not just the strategy block) so the
RemoteSemKNNRouter can derive a user_config fingerprint from the
configured channels and critic.

Environment overrides (intended for local development; redirect production
deployments via YAML instead):

  ``AGENTCODEC_SEMKNN_SERVER_URL``
      If set, **replaces** ``router.server_url`` for SemKNN routes. Lets
      you swap between the public backend and a ``./backend/start_dev.sh``
      instance without editing the YAML. Empty string is treated as
      "unset". The override is logged at INFO level on apply.

  ``AGENTCODEC_TELEMETRY_ENDPOINT``
      Separate from the above — this one only redirects /telemetry POSTs.
      See agentcodec.telemetry.
"""

from __future__ import annotations

import logging
import os

from ..config import FixedStrategy, LibraryConfig, RoutedStrategy
from .acm_table import ACMTableRouter
from .base import Router
from .fixed import FixedRouter
from .linear import LinearRouter
from .remote import RemoteSemKNNRouter, _derive_user_config

logger = logging.getLogger(__name__)

ENV_SEMKNN_SERVER_URL = "AGENTCODEC_SEMKNN_SERVER_URL"


def _build_fallback(rcfg, config: LibraryConfig) -> Router | None:
    """Construct the offline-fallback router named in router.fallback, or None."""
    kind = getattr(rcfg, "fallback", "none")
    if kind == "none":
        return None
    if kind == "linear":
        if not rcfg.fallback_cache:
            raise ValueError("router.fallback=linear requires fallback_cache")
        return LinearRouter(cache_path=rcfg.fallback_cache)
    if kind == "acm_table":
        return ACMTableRouter(
            table=rcfg.table, category_tables=rcfg.category_tables,
        )
    raise ValueError(f"Unknown router.fallback: {kind!r}")


def build_router(config: LibraryConfig) -> Router:
    """Construct the right Router for the given LibraryConfig.

    Widened from the older signature (which took only the strategy block)
    so the remote SemKNN router can derive a `user_config` fingerprint from
    the surrounding `models` + `critic` blocks.
    """
    strategy = config.strategy

    if isinstance(strategy, FixedStrategy):
        return FixedRouter(technique=strategy.technique)

    if isinstance(strategy, RoutedStrategy):
        r = strategy.router
        if r.type == "semknn":
            # Allow a dev-time env override of the SemKNN base URL so the
            # same YAML config can target the public backend in CI and a
            # local ./backend/start_dev.sh in dev without edits.
            server_url = r.server_url
            override = os.environ.get(ENV_SEMKNN_SERVER_URL, "").strip()
            if override:
                logger.info(
                    "%s set; redirecting SemKNN /route from %r to %r",
                    ENV_SEMKNN_SERVER_URL, server_url, override,
                )
                server_url = override
            return RemoteSemKNNRouter(
                server_url=server_url,
                lambda_=r.lambda_,
                k=r.knn_k_override,
                api_key=r.api_key,
                timeout_s=r.timeout_s,
                strict_match=r.strict_match,
                user_config=_derive_user_config(config),
                fallback=_build_fallback(r, config),
            )
        if r.type == "acm_linear":
            return LinearRouter(cache_path=r.cache)
        if r.type == "acm_table":
            return ACMTableRouter(
                table=r.table, category_tables=r.category_tables,
            )
        raise ValueError(f"Unknown router type: {r.type!r}")

    raise TypeError(f"Unknown strategy type: {type(strategy).__name__}")
