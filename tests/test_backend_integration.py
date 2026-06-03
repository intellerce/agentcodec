"""End-to-end checks against a real SemKNN backend.

These tests auto-probe a small list of likely local URLs and run when
one of them answers. They **do not** require any env var to be set —
just start the backend (`cd backend && ./start_dev.sh`) and run pytest.

Auto-probe order:
  1. ``AGENTCODEC_TEST_BACKEND_URL`` (env, if set — explicit override)
  2. ``AGENTCODEC_SEMKNN_SERVER_URL`` (env, if set — reuses the
      production-style override so one variable points everything local)
  3. ``http://127.0.0.1:18765`` (the start_dev.sh default port)
  4. ``http://127.0.0.1:8000``  (the Docker default port)

If none answers we skip cleanly so CI doesn't fail on machines with no
backend at all.

What they cover:
  * /healthz, /meta, /profiles are wired correctly.
  * /route returns a ``chosen`` technique for both a matched lineup
    (estimate=False) and a mismatched lineup (estimate=True).
  * /route validates the embedding dim against /meta.
  * /telemetry accepts a scrubbed event and acknowledges the count.

The integration tests synthesize a unit-norm vector at the dim reported
by ``/meta``, so they don't need the BGE encoder installed — fastembed
or sentence-transformers downloads are out of the critical path.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
logger = logging.getLogger(__name__)


def _candidate_urls() -> list[str]:
    """Build the auto-probe list. Explicit env vars first, then defaults."""
    seen: set[str] = set()
    out: list[str] = []
    for src in (
        os.environ.get("AGENTCODEC_TEST_BACKEND_URL"),
        os.environ.get("AGENTCODEC_SEMKNN_SERVER_URL"),
        "http://127.0.0.1:18765",
        "http://127.0.0.1:8000",
    ):
        if not src:
            continue
        u = src.rstrip("/")
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    candidates = _candidate_urls()
    last_err: Exception | None = None
    for url in candidates:
        try:
            c = httpx.Client(base_url=url, timeout=2.0)
            r = c.get("/healthz")
            r.raise_for_status()
            logger.info("backend integration: using %s", url)
            try:
                yield c
            finally:
                c.close()
            return
        except Exception as e:
            last_err = e
            try:
                c.close()
            except Exception:
                pass
            continue
    pytest.skip(
        "No SemKNN backend reachable on any of the probed URLs: "
        f"{candidates!r}. Start one with "
        f"`cd backend && ./start_dev.sh` (default port 18765), or set "
        f"AGENTCODEC_TEST_BACKEND_URL=<your-url>. Last error: {last_err!r}"
    )


@pytest.fixture(scope="module")
def meta(client: httpx.Client) -> dict:
    return client.get("/meta").json()


def _unit_vec(dim: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


def test_healthz(client: httpx.Client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_meta_has_required_fields(meta: dict) -> None:
    for k in ("version", "bge_model", "dim", "default_profile",
              "fuzzy_match_policy", "expose_scores", "max_lambda",
              "telemetry_enabled", "telemetry_max_batch"):
        assert k in meta, f"/meta missing {k!r}"
    assert isinstance(meta["dim"], int) and meta["dim"] > 0
    assert isinstance(meta["telemetry_enabled"], bool)


def test_profiles(client: httpx.Client) -> None:
    r = client.get("/profiles")
    assert r.status_code == 200
    body = r.json()
    assert "default" in body
    assert isinstance(body["profiles"], list)
    assert body["profiles"], "no profiles loaded — start_dev.sh should ship one"
    profile_ids = {p["id"] for p in body["profiles"]}
    assert body["default"] in profile_ids


# ---------------------------------------------------------------------------
# /route — happy + estimate + validation
# ---------------------------------------------------------------------------


def test_route_exact_match(client: httpx.Client, meta: dict) -> None:
    """The shipped dev profile's GENERATOR pool is nemotron:30b + devstral:24b
    (GLM-5.1 is the judge, not a channel). Sending that lineup with matching
    channel_specs → exact match, estimate=False."""
    payload = {
        "embedding": _unit_vec(meta["dim"], seed=1),
        "lambda": 5.0,
        "user_config": {
            "model_families": ["nemotron", "devstral"],
            "channel_specs": [
                {"family": "nemotron", "params_b": 30, "quant": None},
                {"family": "devstral", "params_b": 24, "quant": None},
            ],
            "n_distinct_channels": 2,
            "primary_temperature": 0.7,
            "category_temperatures": {},
            "has_separate_critic": True,
        },
    }
    r = client.post("/route", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_quality"] == "exact"
    assert body["estimate"] is False
    assert body["chosen"]                              # non-empty technique
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["predicted_quality_for_chosen"] >= 0.0
    assert body["predicted_cost_for_chosen"] >= 0.0


def test_route_estimate_on_mismatch(client: httpx.Client, meta: dict) -> None:
    """A lineup unknown to the server triggers fallback to the default
    profile with estimate=True and a warning the client should surface."""
    payload = {
        "embedding": _unit_vec(meta["dim"], seed=2),
        "lambda": 1.0,
        "user_config": {
            "model_families": ["qwen-2.5", "llama-3"],
            "n_distinct_channels": 2,
            "primary_temperature": 0.7,
            "category_temperatures": {},
            "has_separate_critic": False,
        },
    }
    r = client.post("/route", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["estimate"] is True
    assert body["match_quality"] in {"default", "partial", "fallback"}
    assert body["warnings"], "mismatched lineup must produce a client-visible warning"


def test_route_rejects_wrong_dim(client: httpx.Client, meta: dict) -> None:
    bad_dim = meta["dim"] + 17
    payload = {"embedding": _unit_vec(bad_dim, seed=3), "lambda": 1.0}
    r = client.post("/route", json=payload)
    assert r.status_code == 422
    assert "expected" in r.text


def test_route_rejects_non_unit_norm(client: httpx.Client, meta: dict) -> None:
    """norm=2 should be rejected. (Tests the privacy/sanity floor.)"""
    bad = [2.0] + [0.0] * (meta["dim"] - 1)
    r = client.post("/route", json={"embedding": bad, "lambda": 1.0})
    assert r.status_code == 422
    assert "L2-norm" in r.text


def test_route_rejects_lambda_above_max(client: httpx.Client, meta: dict) -> None:
    payload = {
        "embedding": _unit_vec(meta["dim"], seed=4),
        "lambda": meta["max_lambda"] + 1.0,
    }
    r = client.post("/route", json=payload)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /telemetry — accepted, scrubbed, persisted
# ---------------------------------------------------------------------------


def test_telemetry_accepts_clean_event(client: httpx.Client, meta: dict) -> None:
    if not meta.get("telemetry_enabled"):
        pytest.skip("telemetry disabled on this backend")
    payload = {
        "events": [{
            "schema_version": 1,
            "session_id": "test-session",
            "client_version": "0.0.0-test",
            "ts_iso": "2026-05-27T00:00:00Z",
            "router_type": "fixed",
            "technique_used": "harq_ir",
            "task_category": "qa",
            "latency_s": 1.23,
            "input_tokens": 100,
            "output_tokens": 50,
            "embedding": _unit_vec(meta["dim"], seed=5),
            "embedding_bge_model": meta["bge_model"],
            "user_config": {"model_families": ["nemotron"]},
        }],
    }
    r = client.post("/telemetry", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == 1
    assert body["dropped_invalid"] == 0


def test_telemetry_scrubs_forbidden_keys(client: httpx.Client, meta: dict) -> None:
    """Server-side belt to the client-side suspenders: events containing
    forbidden keys (prompt, api_key, etc.) are scrubbed before write."""
    if not meta.get("telemetry_enabled"):
        pytest.skip("telemetry disabled on this backend")
    poisoned = {
        "schema_version": 1,
        "technique_used": "baseline",
        "prompt": "this should not be persisted",
        "api_key": "sk-also-no",
        "task_id": "user-12345",
    }
    r = client.post("/telemetry", json={"events": [poisoned]})
    assert r.status_code == 200, r.text
    body = r.json()
    # The event is "accepted" (scrubbed shape was non-empty) — the fields
    # themselves get dropped. The unit-level test
    # backend/tests/test_telemetry.py asserts the on-disk JSONL doesn't
    # contain those values; here we just confirm the wire contract holds.
    assert body["accepted"] == 1


def test_telemetry_rejects_oversize_batch(client: httpx.Client, meta: dict) -> None:
    if not meta.get("telemetry_enabled"):
        pytest.skip("telemetry disabled on this backend")
    limit = meta["telemetry_max_batch"]
    too_many = [{"schema_version": 1} for _ in range(limit + 1)]
    r = client.post("/telemetry", json={"events": too_many})
    assert r.status_code == 413
