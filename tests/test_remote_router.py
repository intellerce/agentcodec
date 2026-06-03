"""Tests for RemoteSemKNNRouter against an in-process httpx mock backend."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agentcodec.models import TaskCategory, TaskItem
from agentcodec.routing import LinearRouter, RouterDecision  # noqa: F401
from agentcodec.routing.remote import (
    DEFAULT_BGE_MODEL,
    RemoteSemKNNRouter,
    _StrictMismatch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeEncoder:
    """Stand-in for the :class:`_UnifiedEncoder` API: ``encode(text) ->
    list[float]`` returning a deterministic unit-norm vector at the
    configured dimension. The vector is unit-norm so the server's norm
    check passes; the exact value is irrelevant for these tests."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)
        # Deterministic but distinct across calls — pick the basis vector
        # whose index varies with the call count.
        idx = (len(self.calls) - 1) % self.dim
        v = [0.0] * self.dim
        v[idx] = 1.0
        return v


def _task(prompt: str = "What's the capital of France?") -> TaskItem:
    return TaskItem(id="t1", category=TaskCategory.QA, prompt=prompt)


def _mock_transport(
    meta: dict[str, Any] | None = None,
    route_response: dict[str, Any] | None = None,
    route_status: int = 200,
) -> httpx.MockTransport:
    meta = meta or {
        "bge_model": DEFAULT_BGE_MODEL,
        "dim": 1024,
        "default_profile": "p0",
        "fuzzy_match_policy": "warn",
        "expose_scores": False,
        "max_lambda": 100.0,
        "version": "0.3.0",
    }
    route_response = route_response or {
        "chosen": "harq_ir",
        "confidence": 0.71,
        "profile_used": "p0",
        "match_quality": "exact",
        "match_similarity": 1.0,
        "estimate": False,
        "masked_techniques": [],
        "warnings": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/meta":
            return httpx.Response(200, json=meta)
        if req.url.path == "/route":
            return httpx.Response(route_status, json=route_response)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _make_router(transport: httpx.MockTransport, **kwargs) -> RemoteSemKNNRouter:
    r = RemoteSemKNNRouter(
        server_url="https://semknn.example.com",
        lambda_=kwargs.pop("lambda_", 5.0),
        encoder=_FakeEncoder(dim=1024),
        **kwargs,
    )
    r._client = httpx.Client(transport=transport)
    return r


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_decision() -> None:
    r = _make_router(_mock_transport())
    d = r.choose(_task())
    assert isinstance(d, RouterDecision)
    assert d.chosen == "harq_ir"
    assert d.confidence == pytest.approx(0.71)
    assert d.router_type == "semknn_remote"
    assert d.extra["match_quality"] == "exact"
    assert d.extra["estimate"] is False
    assert d.extra["lambda"] == 5.0


def test_meta_called_once() -> None:
    """The /meta request should only happen on the first .choose() call."""
    calls = {"meta": 0, "route": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/meta":
            calls["meta"] += 1
            return httpx.Response(200, json={
                "bge_model": DEFAULT_BGE_MODEL, "dim": 1024,
                "default_profile": "p0", "fuzzy_match_policy": "warn",
                "expose_scores": False, "max_lambda": 100.0, "version": "0.3.0",
            })
        if req.url.path == "/route":
            calls["route"] += 1
            return httpx.Response(200, json={
                "chosen": "baseline", "confidence": 0.5,
                "profile_used": "p0", "match_quality": "exact",
                "match_similarity": 1.0, "estimate": False,
                "masked_techniques": [], "warnings": [],
            })
        return httpx.Response(404)

    r = _make_router(httpx.MockTransport(handler))
    r.choose(_task("a"))
    r.choose(_task("b"))
    r.choose(_task("c"))
    assert calls["meta"] == 1
    assert calls["route"] == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_bge_mismatch_raises_at_first_call() -> None:
    transport = _mock_transport(meta={
        "bge_model": "BAAI/bge-base-en-v1.5",  # mismatch
        "dim": 768,
        "default_profile": "p0",
        "fuzzy_match_policy": "warn",
        "expose_scores": False,
        "max_lambda": 100.0,
        "version": "0.3.0",
    })
    r = _make_router(transport)
    with pytest.raises(RuntimeError, match="BGE-model mismatch"):
        r.choose(_task())


# ---------------------------------------------------------------------------
# Estimate / warnings
# ---------------------------------------------------------------------------


def test_estimate_response_emits_one_time_warning() -> None:
    transport = _mock_transport(route_response={
        "chosen": "self_consistency",
        "confidence": 0.62,
        "profile_used": "p0",
        "match_quality": "fallback",
        "match_similarity": 0.5,
        "estimate": True,
        "masked_techniques": [],
        "warnings": ["Profile model_families don't match your channel pool."],
    })
    r = _make_router(transport)
    d1 = r.choose(_task("a"))
    assert d1.extra["estimate"] is True
    assert "one_time_warning" in d1.extra
    # Same (profile, match_quality) → no duplicate warning on the next call.
    d2 = r.choose(_task("b"))
    assert d2.extra["estimate"] is True
    assert "one_time_warning" not in d2.extra


# ---------------------------------------------------------------------------
# 409 strict-match handling
# ---------------------------------------------------------------------------


def test_strict_match_409_is_not_silently_fallbacked() -> None:
    transport = _mock_transport(
        route_response={"detail": "strict_match_no_exact_match"},
        route_status=409,
    )
    fallback_called = {"v": False}

    class _FallbackRouter:
        def choose(self, task):
            fallback_called["v"] = True
            return RouterDecision(chosen="baseline", router_type="fallback")

    r = _make_router(transport, fallback=_FallbackRouter())
    with pytest.raises(_StrictMismatch):
        r.choose(_task())
    assert fallback_called["v"] is False, (
        "strict-match 409 must NOT delegate to fallback — that would mask the "
        "user's explicit request for an exact match."
    )


# ---------------------------------------------------------------------------
# Network errors + fallback
# ---------------------------------------------------------------------------


def test_network_error_without_fallback_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")
    r = _make_router(httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="unreachable"):
        r.choose(_task())


def test_network_error_with_fallback_delegates() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    class _Fallback:
        def choose(self, task):
            return RouterDecision(
                chosen="baseline", confidence=0.9, router_type="fallback",
            )

    r = _make_router(httpx.MockTransport(handler), fallback=_Fallback())
    d = r.choose(_task())
    assert d.chosen == "baseline"
    assert d.router_type == "fallback"
