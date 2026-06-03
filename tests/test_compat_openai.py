"""
End-to-end tests for the OpenAI compat shim. Two distinct paths are
exercised:

* **Passthrough** (``reliability=None``) — the shim must instantiate the
  native ``openai.OpenAI`` lazily and proxy ``client.chat.completions.create``
  through unchanged. We patch the openai SDK so no network call happens.
* **Reliability** (``reliability="baseline"``, etc.) — the shim builds a
  ReliabilityModule lazily on first call and adapts the response back to
  OpenAI's shape. We patch ``agentcodec.api.AgentChannel`` so the
  reliability pipeline runs against the MockChannel.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcodec.openai import AsyncOpenAI, OpenAI


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``openai`` module so the passthrough path can construct
    a client without the real SDK being installed."""
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock(name="openai.OpenAI")  # type: ignore[attr-defined]
    fake_openai.AsyncOpenAI = MagicMock(name="openai.AsyncOpenAI")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return fake_openai


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


def test_passthrough_lazy_instantiates_native_only_on_use(monkeypatch):
    """No `reliability=` → no native client constructed until first call."""
    fake = _install_fake_openai(monkeypatch)
    client = OpenAI(api_key="sk-test")
    # Constructor doesn't touch the native SDK.
    fake.OpenAI.assert_not_called()
    # First call constructs it.
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert fake.OpenAI.call_count == 1
    fake.OpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini", messages=[], stream=False,
    )


def test_passthrough_forwards_arbitrary_kwargs(monkeypatch):
    fake = _install_fake_openai(monkeypatch)
    client = OpenAI(api_key="sk-test")
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}], temperature=0.2, custom_kwarg="x",
    )
    create = fake.OpenAI.return_value.chat.completions.create
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["tools"] == [{"type": "function"}]
    assert kwargs["temperature"] == 0.2
    assert kwargs["custom_kwarg"] == "x"


def test_passthrough_proxies_unknown_namespaces(monkeypatch):
    """`client.embeddings.create(...)` falls through to the native client."""
    fake = _install_fake_openai(monkeypatch)
    client = OpenAI(api_key="sk-test")
    _ = client.embeddings  # triggers __getattr__
    fake.OpenAI.assert_called_once()


# ---------------------------------------------------------------------------
# Reliability path
# ---------------------------------------------------------------------------


def test_reliability_preset_returns_openai_shaped_response(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    client = OpenAI(api_key="sk-test", reliability="baseline")
    mock_channel.responses[:] = [("Paris.", 0.99)]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Capital of France?"},
        ],
    )
    # Native OpenAI shape preserved.
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Paris."
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    # Reliability metadata exposed via custom attribute.
    assert resp.reliability.technique_used == "baseline"
    # System message reached the underlying channel.
    assert mock_channel.calls[0]["request"].system == "Be brief."


def test_reliability_per_call_overrides_constructor(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    fake_openai = _install_fake_openai(monkeypatch)
    client = OpenAI(api_key="sk-test", reliability="baseline")
    # Per-call reliability=False forces passthrough for this call only.
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        reliability=False,
    )
    assert fake_openai.OpenAI.return_value.chat.completions.create.call_count == 1


def test_tools_thread_through_to_reliability_module(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    client = OpenAI(api_key="sk-test", reliability="baseline")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Lookup weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Weather in SF?"}],
        tools=tools, tool_choice="auto",
    )
    # The ChatRequest the channel saw must carry the tools.
    request = mock_channel.calls[0]["request"]
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "get_weather"
    assert request.tool_choice == "auto"


def test_streaming_produces_openai_shaped_chunks(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    client = OpenAI(api_key="sk-test", reliability="baseline")
    mock_channel.responses[:] = [("Streamed reply.", 0.95)]
    chunks = list(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    ))
    # First chunk carries the role + first content delta.
    assert any(
        getattr(c.choices[0].delta, "content", None) == "Streamed reply."
        for c in chunks
    )
    # Last chunk has finish_reason set.
    assert chunks[-1].choices[0].finish_reason == "stop"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


async def test_async_passthrough(monkeypatch):
    fake = _install_fake_openai(monkeypatch)

    async def _async_create(**kw: Any) -> Any:
        return {"echo": kw}

    fake.AsyncOpenAI.return_value.chat.completions.create = _async_create
    client = AsyncOpenAI(api_key="sk-test")
    out = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[],
    )
    assert out == {"echo": {"model": "gpt-4o-mini", "messages": [], "stream": False}}


async def test_async_reliability(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    client = AsyncOpenAI(api_key="sk-test", reliability="baseline")
    mock_channel.responses[:] = [("Async OK.", 0.9)]
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
    )
    assert resp.choices[0].message.content == "Async OK."
