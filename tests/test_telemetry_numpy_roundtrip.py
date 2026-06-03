"""Telemetry must serialize cleanly even when callers hand it raw numpy
arrays (BGE encoders return np.ndarray by default).

The library already converts to `list[float]` in `_encode_for_telemetry`
(`agentcodec/api.py`), but a regression here would crash silently on
network flush. This test pins the contract.
"""

from __future__ import annotations

import json
import time

import httpx
import numpy as np
import pytest

from agentcodec.telemetry import (
    Telemetry,
    TelemetryConfig,
    _scrub,
)


def _capture_client(captured: list[httpx.Request]) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"accepted": 1})
    return httpx.Client(transport=httpx.MockTransport(handler))


def _wait_for(t: Telemetry, target: int, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while t.stats["sent"] < target and time.monotonic() < deadline:
        time.sleep(0.02)


def test_scrub_passes_numeric_lists_through() -> None:
    """A short list of floats (the shape we send embeddings as) is kept."""
    out = _scrub({"embedding": [0.1, 0.2, 0.3]})
    assert out == {"embedding": [0.1, 0.2, 0.3]}


def test_record_with_list_embedding_serializes(monkeypatch) -> None:
    """End-to-end with the real list-of-floats shape — should reach the
    HTTP transport without raising."""
    monkeypatch.delenv("AGENTCODEC_TELEMETRY", raising=False)
    monkeypatch.setenv("AGENTCODEC_TELEMETRY_QUIET", "1")

    captured: list[httpx.Request] = []
    cfg = TelemetryConfig(
        endpoint="https://t.example.com/telemetry",
        enabled=True, flush_interval_s=0.05, batch_max=4, timeout_s=2.0,
    )
    t = Telemetry(cfg, client_version="0.0.0", http_client=_capture_client(captured))
    t.record({"embedding": [float(x) for x in range(8)], "k": 20, "lambda": 5.0})
    _wait_for(t, target=1)
    t.shutdown()
    body = captured[0].read().decode()
    blob = json.loads(body)
    assert blob["events"][0]["embedding"] == list(range(8))
    assert blob["events"][0]["lambda"] == 5.0


def test_record_with_numpy_embedding_serializes_or_drops_cleanly(monkeypatch) -> None:
    """If a caller hands in a numpy array directly (regression of the
    `_encode_for_telemetry` conversion), telemetry must not crash the
    background worker.

    The defensible behaviors are:
        a) silently drop the offending event (current `_scrub` does this
           by virtue of not handling ndarray; the worker then can't
           serialize it and the batch is discarded), OR
        b) coerce to list[float] in `record()`.

    Either way: no exceptions surface to the caller, no events leak with
    raw ndarray.
    """
    monkeypatch.delenv("AGENTCODEC_TELEMETRY", raising=False)
    monkeypatch.setenv("AGENTCODEC_TELEMETRY_QUIET", "1")

    captured: list[httpx.Request] = []
    cfg = TelemetryConfig(
        endpoint="https://t.example.com/telemetry",
        enabled=True, flush_interval_s=0.05, batch_max=4, timeout_s=2.0,
    )
    t = Telemetry(cfg, client_version="0.0.0", http_client=_capture_client(captured))

    # This must not raise from the caller's perspective.
    t.record({"embedding": np.zeros(8, dtype=np.float32), "technique_used": "harq_ir"})
    time.sleep(0.3)
    t.shutdown()

    # If something *was* sent, it must be JSON-serializable (no ndarray).
    if captured:
        body = captured[0].read().decode()
        try:
            blob = json.loads(body)
        except json.JSONDecodeError as e:  # pragma: no cover
            pytest.fail(f"telemetry sent invalid JSON: {e}")
        emb = blob["events"][0].get("embedding")
        if emb is not None:
            assert isinstance(emb, list)
            assert all(isinstance(x, (int, float)) for x in emb)