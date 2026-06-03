"""Unit tests for the anonymous-telemetry client.

We exercise the queue, env-var disable, one-time notice, scrubbing, the
fail-silent behavior on network errors, and the never-block contract.
"""

from __future__ import annotations

import time

import httpx
import pytest

from agentcodec.telemetry import (
    SCHEMA_VERSION,
    Telemetry,
    TelemetryConfig,
    _scrub,
    build_event_from_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_client(
    captured: list[httpx.Request],
    status: int = 200,
    raise_exc: Exception | None = None,
) -> httpx.Client:
    """An httpx.Client that records each POST and returns `status`."""
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(status, json={"accepted": 1})
    return httpx.Client(transport=httpx.MockTransport(handler))


def _wait_for_send(t: Telemetry, target: int, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while t.stats["sent"] < target and time.monotonic() < deadline:
        time.sleep(0.02)


def _cfg(endpoint: str = "https://t.example.com/telemetry", **kwargs) -> TelemetryConfig:
    return TelemetryConfig(
        endpoint=endpoint, flush_interval_s=0.1, batch_max=8,
        timeout_s=2.0, queue_max=64, **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the master switch is unset for each test, and silence the notice
    # so it doesn't pollute pytest output.
    monkeypatch.delenv("AGENTCODEC_TELEMETRY", raising=False)
    monkeypatch.setenv("AGENTCODEC_TELEMETRY_QUIET", "1")


# ---------------------------------------------------------------------------
# enable/disable
# ---------------------------------------------------------------------------


def test_disabled_yaml_short_circuits() -> None:
    t = Telemetry(_cfg(enabled=False), client_version="0.0.0")
    assert t.enabled is False
    t.record({"hello": "world"})
    assert t.stats["sent"] == 0
    assert t.stats["dropped"] == 0
    assert t._thread is None  # never even spun up


@pytest.mark.parametrize("val", ["0", "false", "FALSE", "off", "no", "disabled"])
def test_env_var_disables(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("AGENTCODEC_TELEMETRY", val)
    t = Telemetry(_cfg(enabled=True), client_version="0.0.0")
    assert t.enabled is False


def test_env_var_overrides_yaml_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var is the master switch: even if YAML says disabled, setting
    the env to a positive value re-enables."""
    monkeypatch.setenv("AGENTCODEC_TELEMETRY", "1")
    t = Telemetry(_cfg(enabled=False), client_version="0.0.0")
    assert t.enabled is True


def test_no_endpoint_disables_silently() -> None:
    t = Telemetry(_cfg(endpoint=None), client_version="0.0.0")
    assert t.enabled is False


# ---------------------------------------------------------------------------
# send / batch / shutdown
# ---------------------------------------------------------------------------


def test_record_sends_to_endpoint() -> None:
    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.3.0",
                  http_client=_capture_client(captured))
    t.record({"technique_used": "harq_ir", "observed_quality": 0.81})
    _wait_for_send(t, target=1)
    t.shutdown()
    assert len(captured) >= 1
    # Body shape is {events: [...]}
    body = captured[0].read().decode()
    assert '"events"' in body
    assert '"harq_ir"' in body
    # Envelope fields were added by the client.
    assert '"client_version":"0.3.0"' in body
    assert f'"schema_version":{SCHEMA_VERSION}' in body
    assert '"session_id"' in body


def test_session_id_consistent_within_process() -> None:
    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(captured))
    t.record({"i": 0})
    t.record({"i": 1})
    _wait_for_send(t, target=2)
    t.shutdown()
    body = b"".join(r.read() for r in captured).decode()
    import re
    sids = set(re.findall(r'"session_id":"([0-9a-f-]+)"', body))
    assert len(sids) == 1, f"expected 1 session id, saw {sids}"


def test_drops_when_queue_full() -> None:
    """Worker starts but the network call never returns until shutdown —
    fill the queue and assert events are dropped, not blocking."""

    def slow_handler(req: httpx.Request) -> httpx.Response:
        # Block forever; we only care that the producer doesn't.
        import time as _t
        _t.sleep(5)
        return httpx.Response(200, json={"accepted": 0})

    client = httpx.Client(transport=httpx.MockTransport(slow_handler))
    t = Telemetry(
        TelemetryConfig(
            endpoint="https://t.example.com/telemetry",
            queue_max=4, batch_max=2, flush_interval_s=0.05, timeout_s=2.0,
        ),
        client_version="0.0.0",
        http_client=client,
    )
    # Send way more than the queue can hold; record() must not raise/block.
    t0 = time.monotonic()
    for i in range(200):
        t.record({"i": i})
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"record() blocked: {elapsed:.2f}s"
    assert t.stats["dropped"] > 0
    t.shutdown(timeout_s=0.5)


# ---------------------------------------------------------------------------
# Network failure → silent
# ---------------------------------------------------------------------------


def test_network_error_does_not_raise() -> None:
    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(
                      captured, raise_exc=httpx.ConnectError("nope"),
                  ))
    t.record({"x": 1})
    time.sleep(0.3)
    t.shutdown()
    assert t.stats["sent"] == 0
    assert t.stats["failed_batches"] >= 1


def test_server_5xx_drops_batch_silently() -> None:
    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(captured, status=500))
    t.record({"x": 1})
    time.sleep(0.3)
    t.shutdown()
    assert t.stats["sent"] == 0
    assert t.stats["failed_batches"] >= 1


# ---------------------------------------------------------------------------
# Scrubbing — the privacy fence
# ---------------------------------------------------------------------------


def test_scrub_drops_forbidden_keys() -> None:
    out = _scrub({
        "prompt": "How does QUIC compare to TCP?",        # forbidden
        "output_text": "QUIC is built on UDP and ...",     # forbidden
        "api_key": "sk-deadbeef",                           # forbidden
        "task_id": "user-12345-private",                    # forbidden
        "metadata": {"user_id": "abc"},                     # forbidden
        "technique_used": "harq_ir",                        # kept
        "observed_quality": 0.81,                           # kept
        "nested": {"prompt": "drop me", "keep": 1},         # forbidden inside nested
    })
    assert "prompt" not in out
    assert "output_text" not in out
    assert "api_key" not in out
    assert "task_id" not in out
    assert "metadata" not in out
    assert out["technique_used"] == "harq_ir"
    assert out["nested"] == {"keep": 1}


def test_scrub_drops_long_strings() -> None:
    out = _scrub({"comment": "x" * 2000})
    assert "comment" not in out


def test_record_scrubs_payload() -> None:
    """End-to-end: a poisoned payload arrives at the server without secrets."""
    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(captured))
    t.record({
        "technique_used": "harq_ir",
        "prompt": "should never be sent",
        "api_key": "sk-also-no",
    })
    _wait_for_send(t, target=1)
    t.shutdown()
    body = captured[0].read().decode()
    assert "should never be sent" not in body
    assert "sk-also-no" not in body
    assert "harq_ir" in body


# ---------------------------------------------------------------------------
# build_event_from_result — what gets shipped
# ---------------------------------------------------------------------------


class _FakeResult:
    technique_used = "harq_ir"
    latency_s = 2.34
    wall_clock_s = 2.45
    cumulative_latency_s = 2.45
    input_tokens = 1234
    output_tokens = 567
    thinking_tokens = 0
    rounds = 3
    num_llm_calls = 4
    thinking_used = False
    # `final_quality` is what build_event_from_result reads as the
    # observed_quality retraining signal. quality_score is the per-call
    # value, not the run-level one.
    final_quality = 0.81
    best_individual_quality = 0.78
    diversity_gain = 0.03
    judge_cost_usd = 0.001
    cost_usd = 0.0123
    cost_source = "estimated"
    # Things that MUST NOT appear in the payload, even if accidentally read.
    text = "PROMPT OUTPUT — should never be sent"
    reference = "ground-truth answer"
    task_id = "private-task-id-with-pii"


def test_record_telemetry_fires_for_fixed_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry now fires for every router, not just SemKNN. For a
    FixedRouter run we lazy-encode the prompt with BGE-small (stubbed in
    this test to avoid loading the model) and the event must reach the
    HTTP transport with an embedding attached but no prompt text."""
    from types import SimpleNamespace

    # Stub the encoder helper so the test doesn't hit sentence-transformers.
    import agentcodec.api as api_mod
    monkeypatch.setattr(
        api_mod, "_encode_for_telemetry",
        lambda prompt: ([0.1, 0.2, 0.3], "BAAI/bge-small-en-v1.5"),
    )

    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(captured))

    # Build a minimal ReliabilityModule-shaped stub that exercises the real
    # _record_telemetry method without instantiating the full module.
    from agentcodec.api import ReliabilityModule
    from agentcodec.routing import FixedRouter

    mod = SimpleNamespace(
        router=FixedRouter(technique="harq_ir"),
        config=SimpleNamespace(
            models=[SimpleNamespace(
                model="qwen2.5:7b", temperature=0.7,
                category_temperatures=None,
            )],
            critic=None,
        ),
        telemetry=t,
    )

    decision = SimpleNamespace(extra={}, router_type="fixed")
    task = SimpleNamespace(
        category=SimpleNamespace(value="qa"),
        prompt="What's the capital of France?",
    )
    ReliabilityModule._record_telemetry(
        mod, result=_FakeResult(), decision=decision, task=task,
    )
    _wait_for_send(t, target=1)
    t.shutdown()

    assert t.stats["sent"] == 1
    assert len(captured) == 1
    body = captured[0].read()
    # Prompt text never appears.
    assert b"What's the capital of France?" not in body
    # Embedding does.
    assert b"embedding" in body
    assert b"BAAI/bge-small-en-v1.5" in body


def test_record_telemetry_skipped_when_encoder_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sentence-transformers isn't installed (helper returns None), the
    non-SemKNN path silently skips the event rather than crashing."""
    from types import SimpleNamespace

    import agentcodec.api as api_mod
    monkeypatch.setattr(api_mod, "_encode_for_telemetry", lambda prompt: (None, None))

    captured: list[httpx.Request] = []
    t = Telemetry(_cfg(), client_version="0.0.0",
                  http_client=_capture_client(captured))

    from agentcodec.api import ReliabilityModule
    from agentcodec.routing import FixedRouter

    mod = SimpleNamespace(
        router=FixedRouter(technique="harq_ir"),
        config=SimpleNamespace(
            models=[SimpleNamespace(
                model="qwen2.5:7b", temperature=0.7,
                category_temperatures=None,
            )],
            critic=None,
        ),
        telemetry=t,
    )
    decision = SimpleNamespace(extra={}, router_type="fixed")
    task = SimpleNamespace(category=SimpleNamespace(value="qa"), prompt="x")
    ReliabilityModule._record_telemetry(
        mod, result=_FakeResult(), decision=decision, task=task,
    )
    t.shutdown()
    assert t.stats["sent"] == 0
    assert captured == []


def test_derive_user_config_canonicalizes_lineup() -> None:
    """The channel-pool fingerprint is what the server uses for
    Jaccard-matching. Concrete identifiers must be canonicalized via the
    shipped alias table so two users running 'the same models with
    different version strings' look identical to the server."""
    from types import SimpleNamespace

    from agentcodec.routing.remote import _derive_user_config

    cfg = SimpleNamespace(
        models=[
            SimpleNamespace(
                model="nvidia/Llama-3.1-Nemotron-70B-Instruct",
                temperature=0.7, category_temperatures=None,
            ),
            SimpleNamespace(
                model="mistralai/Devstral-Small-2505",
                temperature=0.5, category_temperatures=None,
            ),
        ],
        critic=SimpleNamespace(same=False),
    )
    uc = _derive_user_config(cfg)
    assert uc["model_families"] == ["nemotron", "devstral"]
    assert uc["n_distinct_channels"] == 2
    assert uc["primary_temperature"] == 0.7
    assert uc["has_separate_critic"] is True


def test_build_event_omits_text_reference_taskid() -> None:
    payload = build_event_from_result(
        result=_FakeResult(),
        routing_extra={
            "profile_used": "p", "match_quality": "exact",
            "predicted_quality_for_chosen": 0.78,
            "predicted_cost_for_chosen": 0.011,
            "k": 20,
        },
        router_type="semknn_remote",
        user_config={"model_families": ["nemotron", "devstral", "glm-5.1"]},
        lambda_=5.0,
        embedding=[0.0] * 384,
        bge_model="BAAI/bge-small-en-v1.5",
        task_category="qa",
    )
    assert "text" not in payload and "reference" not in payload
    assert "task_id" not in payload
    assert payload["observed_quality"] == 0.81
    assert payload["predicted_quality"] == 0.78
    assert payload["technique_used"] == "harq_ir"
    assert payload["lambda"] == 5.0
    assert payload["user_config"]["model_families"] == [
        "nemotron", "devstral", "glm-5.1",
    ]
    assert payload["embedding"] is not None
    assert payload["embedding_bge_model"] == "BAAI/bge-small-en-v1.5"
