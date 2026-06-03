"""When the primary technique raises and `on_error: fallback_baseline`
is configured, ReliabilityModule must transparently downgrade to a
baseline call rather than propagating the exception.

We exercise the `run()` and `stream()` paths separately because they
build the same fallback through different code paths and either could
regress in isolation.
"""

from __future__ import annotations

import pytest

from agentcodec.dispatch import DispatchContext, dispatch
from agentcodec.results import FinalEvent, WarningEvent


def _build_module(monkeypatch, on_error: str, mock_scorer, mock_channel):
    """Construct a minimal ReliabilityModule whose dispatcher is stubbed.

    We don't instantiate via from_dict() because that builds real
    AgentChannels (which would import openai/anthropic). Instead we
    monkey-patch `dispatch` to make the primary technique blow up, then
    construct the module by hand.
    """
    from agentcodec.api import ReliabilityModule
    from agentcodec.config import (
        Defaults,
        FixedStrategy,
        JudgeConfig,
        LibraryConfig,
        ModelConfig,
    )
    cfg = LibraryConfig(
        models=[ModelConfig(model="mock/gpt", api_key="mock")],
        judge=JudgeConfig(model="mock/judge", api_key="mock"),
        strategy=FixedStrategy(technique="harq_ir", params={"max_rounds": 2}),
        defaults=Defaults(on_error=on_error),
    )

    # ReliabilityModule.__init__ builds AgentChannels which import openai.
    # Sidestep it by bypassing __init__ and wiring fields directly.
    mod = ReliabilityModule.__new__(ReliabilityModule)
    mod.config = cfg
    mod._startup_warnings = []
    mod.channels = {"mock/gpt": mock_channel}
    mod.scorer = mock_scorer
    mod.critic_channel = None

    from agentcodec.routing import AutoCategoryClassifier, FixedRouter
    mod.router = FixedRouter(technique="harq_ir")
    mod._classifier = AutoCategoryClassifier()
    mod._ctx = DispatchContext(channels=mod.channels, scorer=mod.scorer)

    # Telemetry off (also enforced by conftest env var) — give the module
    # a no-op stub so .record() never tries to start a worker thread.
    class _StubTelemetry:
        enabled = False
        def record(self, *a, **kw): pass
        def shutdown(self): pass
        def flush(self, *a, **kw): return True
    mod.telemetry = _StubTelemetry()

    return mod


def test_fallback_baseline_on_run(monkeypatch, mock_channel, mock_scorer) -> None:
    """A failing primary technique falls back to baseline; the result
    carries `error` and the baseline's text."""
    mod = _build_module(monkeypatch, "fallback_baseline", mock_scorer, mock_channel)

    # Make harq_ir blow up, leave baseline working.
    real_dispatch = dispatch

    def flaky_dispatch(technique, task, ctx):
        if technique == "harq_ir":
            raise RuntimeError("simulated primary failure")
        return real_dispatch(technique, task, ctx)

    monkeypatch.setattr("agentcodec.api.dispatch", flaky_dispatch)

    result = mod.run("Hello world")
    assert result.error is not None
    assert "simulated primary failure" in result.error
    assert result.technique_used == "baseline"
    assert result.text  # the baseline call still produced output


def test_raise_propagates(monkeypatch, mock_channel, mock_scorer) -> None:
    """With `on_error: raise`, technique failures must surface to the caller."""
    mod = _build_module(monkeypatch, "raise", mock_scorer, mock_channel)

    def always_fails(technique, task, ctx):
        raise RuntimeError("primary failure that should propagate")

    monkeypatch.setattr("agentcodec.api.dispatch", always_fails)

    with pytest.raises(RuntimeError, match="primary failure that should propagate"):
        mod.run("Hello world")


def test_fallback_baseline_streaming_emits_warning(
    monkeypatch, mock_channel, mock_scorer,
) -> None:
    """`stream()` must emit a WarningEvent(code='fallback_to_baseline')
    before the baseline run kicks in, so consumers see the downgrade."""
    mod = _build_module(monkeypatch, "fallback_baseline", mock_scorer, mock_channel)

    real_dispatch = dispatch

    def flaky_dispatch(technique, task, ctx):
        if technique == "harq_ir":
            raise RuntimeError("boom")
        return real_dispatch(technique, task, ctx)

    monkeypatch.setattr("agentcodec.api.dispatch", flaky_dispatch)

    events = list(mod.stream("Hello"))
    warnings = [e for e in events if isinstance(e, WarningEvent)]
    finals = [e for e in events if isinstance(e, FinalEvent)]
    assert len(finals) == 1
    fallback_warns = [w for w in warnings if w.code == "fallback_to_baseline"]
    assert fallback_warns, (
        "expected a WarningEvent(code='fallback_to_baseline') before the "
        "stream completed; got: " + str([(w.code, w.message) for w in warnings])
    )
    assert finals[0].result.technique_used == "baseline"
    assert finals[0].result.error is not None
