"""
End-to-end tests for the Anthropic compat shim. Same two-path matrix as
the OpenAI shim tests:

* Passthrough: native ``anthropic.Anthropic`` instantiated lazily, calls
  proxied unchanged.
* Reliability: builds a ReliabilityModule lazily on first call and
  returns an Anthropic-shaped ``Message`` object.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcodec.anthropic import Anthropic, AsyncAnthropic


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = types.ModuleType("anthropic")
    fake.Anthropic = MagicMock(name="anthropic.Anthropic")  # type: ignore[attr-defined]
    fake.AsyncAnthropic = MagicMock(name="anthropic.AsyncAnthropic")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


def test_passthrough_lazy(monkeypatch):
    fake = _install_fake_anthropic(monkeypatch)
    client = Anthropic(api_key="sk-test")
    fake.Anthropic.assert_not_called()
    client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=128,
    )
    fake.Anthropic.assert_called_once()


def test_passthrough_forwards_kwargs(monkeypatch):
    fake = _install_fake_anthropic(monkeypatch)
    client = Anthropic(api_key="sk-test")
    client.messages.create(
        model="claude-sonnet-4-6",
        system="Be terse.",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=128,
        temperature=0.3,
        tools=[{"name": "weather"}],
    )
    create = fake.Anthropic.return_value.messages.create
    kwargs = create.call_args.kwargs
    assert kwargs["system"] == "Be terse."
    assert kwargs["temperature"] == 0.3
    assert kwargs["tools"] == [{"name": "weather"}]


# ---------------------------------------------------------------------------
# Reliability path
# ---------------------------------------------------------------------------


def test_reliability_returns_anthropic_shape(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    mock_channel.responses[:] = [("Paris.", 0.99)]
    client = Anthropic(api_key="sk-test", reliability="baseline")
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        system="Be brief.",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=128,
    )
    # Native Anthropic shape: content blocks, stop_reason, usage.input_tokens.
    assert resp.role == "assistant"
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "Paris."
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    # System reached the channel (hoisted from the top-level kwarg).
    assert mock_channel.calls[0]["request"].system == "Be brief."
    # Reliability escape hatch.
    assert resp.reliability.technique_used == "baseline"


def test_reliability_per_call_bypass(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    fake = _install_fake_anthropic(monkeypatch)
    client = Anthropic(api_key="sk-test", reliability="baseline")
    client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        reliability=False,
    )
    fake.Anthropic.return_value.messages.create.assert_called_once()


def test_streaming_yields_anthropic_event_dicts(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    mock_channel.responses[:] = [("Streamed.", 0.9)]
    client = Anthropic(api_key="sk-test", reliability="baseline")
    events = list(client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=128,
        stream=True,
    ))
    types_seen = [getattr(e, "type", None) for e in events]
    assert "message_start" in types_seen
    assert "content_block_start" in types_seen
    assert any(t == "content_block_delta" for t in types_seen)
    assert types_seen[-1] == "message_stop"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


async def test_async_passthrough(monkeypatch):
    fake = _install_fake_anthropic(monkeypatch)

    async def _async_create(**kw: Any) -> Any:
        return {"echo": kw}

    fake.AsyncAnthropic.return_value.messages.create = _async_create
    client = AsyncAnthropic(api_key="sk-test")
    out = await client.messages.create(
        model="claude-sonnet-4-6", messages=[], max_tokens=64,
    )
    assert out["echo"]["model"] == "claude-sonnet-4-6"
