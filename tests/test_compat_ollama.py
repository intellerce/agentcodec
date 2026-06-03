"""
End-to-end tests for the Ollama compat shim.

Ollama's responses are plain dicts (no SDK types), so the reliability
adapter produces ``{"message": {"role":..., "content":...}, "done":...}``
shape with a ``"reliability"`` key carrying technique/cost/latency.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentcodec.ollama import AsyncClient, Client


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = types.ModuleType("ollama")
    fake.Client = MagicMock(name="ollama.Client")  # type: ignore[attr-defined]
    fake.AsyncClient = MagicMock(name="ollama.AsyncClient")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)
    return fake


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


def test_passthrough_lazy(monkeypatch):
    fake = _install_fake_ollama(monkeypatch)
    client = Client(host="http://localhost:11434")
    fake.Client.assert_not_called()
    client.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": "Hi"}])
    fake.Client.assert_called_once()


# ---------------------------------------------------------------------------
# Reliability path
# ---------------------------------------------------------------------------


def test_reliability_returns_ollama_dict_shape(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    mock_channel.responses[:] = [("Paris.", 0.99)]
    client = Client(host="http://localhost:11434", reliability="baseline")
    resp = client.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Capital of France?"},
        ],
    )
    # Native ollama-library dict shape preserved.
    assert resp["message"]["role"] == "assistant"
    assert resp["message"]["content"] == "Paris."
    assert resp["done"] is True
    assert resp["done_reason"] == "stop"
    assert resp["eval_count"] > 0
    # System reached the channel.
    assert mock_channel.calls[0]["request"].system == "Be brief."
    # Reliability summary.
    assert resp["reliability"]["technique_used"] == "baseline"


def test_options_thread_through_to_reliability(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    mock_channel.responses[:] = [("ok", 0.9)]
    client = Client(host="http://localhost:11434", reliability="baseline")
    client.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Hi"}],
        options={"temperature": 0.4, "seed": 42, "top_p": 0.9, "num_predict": 128},
    )
    request = mock_channel.calls[0]["request"]
    assert request.seed == 42
    assert request.top_p == 0.9
    assert request.max_tokens == 128


def test_streaming_yields_ollama_chunk_dicts(
    monkeypatch, mock_channel, mock_scorer,
):
    monkeypatch.setattr("agentcodec.api.AgentChannel", lambda **kw: mock_channel)
    monkeypatch.setattr(
        "agentcodec.api.ReliabilityModule._build_scorer",
        lambda self, jcfg: mock_scorer,
    )
    mock_channel.responses[:] = [("Streamed.", 0.95)]
    client = Client(host="http://localhost:11434", reliability="baseline")
    chunks = list(client.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    ))
    # Intermediate chunks: done=False, content delta.
    assert any(
        not c["done"] and c["message"]["content"] == "Streamed." for c in chunks
    )
    # Final chunk: done=True with reliability summary.
    assert chunks[-1]["done"] is True
    assert "reliability" in chunks[-1]


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


async def test_async_passthrough(monkeypatch):
    fake = _install_fake_ollama(monkeypatch)

    async def _async_chat(**kw: Any) -> Any:
        return {"echo": kw}

    fake.AsyncClient.return_value.chat = _async_chat
    client = AsyncClient(host="http://localhost:11434")
    out = await client.chat(model="qwen2.5:7b", messages=[])
    assert out["echo"]["model"] == "qwen2.5:7b"
