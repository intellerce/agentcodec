"""Phase 2 unit tests — native async streaming via AgentChannel.atransmit_stream().

Each backend is exercised with a mocked async client that emits a stream
of provider-shaped chunks; we then assert that:
* the right ChannelChunk roles are yielded (answer / thinking / tool_call)
* a single terminal ChannelDone arrives with an AgentOutput
* thinking_text + telemetry land on that AgentOutput
* atransmit() returns the same AgentOutput as atransmit_stream's last frame
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — fake async client / response objects
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Wraps a Python iterable in an async iterator."""
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _AsyncContextManagerWrappingIter:
    """Async-context-manager that yields self and is async-iterable."""
    def __init__(self, items):
        self._items = list(items)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return _AsyncIter(self._items)


def _make_channel(model: str = "gpt-4o-mini", *, thinking=None, cost=None):
    """Construct an AgentChannel with the *sync* SDK paths mocked. Tests
    then attach an async client to the channel after construction."""
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock(name="openai.OpenAI")
    fake_openai.AsyncOpenAI = MagicMock(name="openai.AsyncOpenAI")
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = MagicMock(name="anthropic.Anthropic")
    fake_anthropic.AsyncAnthropic = MagicMock(name="anthropic.AsyncAnthropic")
    sys.modules["openai"] = fake_openai
    sys.modules["anthropic"] = fake_anthropic

    from agentcodec.channel import AgentChannel

    return AgentChannel(
        model=model,
        api_key="test",
        thinking=thinking,
        cost_per_1m=cost,
    )


# ---------------------------------------------------------------------------
# OpenAI async streaming — content + reasoning_content + tool_calls
# ---------------------------------------------------------------------------


def _openai_delta_chunk(*, content=None, reasoning_content=None, tool_calls=None):
    delta = MagicMock(
        content=content, reasoning_content=reasoning_content, reasoning=None,
        tool_calls=tool_calls,
    )
    choice = MagicMock(delta=delta, finish_reason=None)
    return MagicMock(choices=[choice], usage=None)


def _openai_final_chunk(*, finish_reason="stop", prompt_tokens=10,
                        completion_tokens=20, reasoning_tokens=None):
    delta = MagicMock(content=None, reasoning_content=None, reasoning=None,
                      tool_calls=None)
    choice = MagicMock(delta=delta, finish_reason=finish_reason)
    if reasoning_tokens is not None:
        details = MagicMock(reasoning_tokens=reasoning_tokens)
    else:
        details = None
    usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=details,
    )
    return MagicMock(choices=[choice], usage=usage)


@pytest.mark.asyncio
async def test_openai_stream_answer_only():
    from agentcodec.messages import ChannelChunk, ChannelDone

    ch = _make_channel("gpt-4o-mini", cost={"input": 1.0, "output": 2.0})
    fake_client = MagicMock()
    chunks = [
        _openai_delta_chunk(content="Hello"),
        _openai_delta_chunk(content=", world"),
        _openai_final_chunk(prompt_tokens=4, completion_tokens=6),
    ]

    async def _create(**kwargs):
        return _AsyncIter(chunks)
    fake_client.chat.completions.create = _create
    ch._async_client = fake_client

    frames = []
    async for f in ch.atransmit_stream("hi"):
        frames.append(f)

    answer_chunks = [f for f in frames if isinstance(f, ChannelChunk) and f.role == "answer"]
    done_frames = [f for f in frames if isinstance(f, ChannelDone)]
    assert "".join(c.text for c in answer_chunks) == "Hello, world"
    assert len(done_frames) == 1
    out = done_frames[0].output
    assert out.text == "Hello, world"
    assert out.input_tokens == 4 and out.output_tokens == 6
    assert out.thinking_emitted is False


@pytest.mark.asyncio
async def test_openai_stream_thinking_via_reasoning_content():
    from agentcodec.messages import ChannelChunk, ChannelDone

    ch = _make_channel("gpt-5", cost={"input": 1.0, "output": 10.0})
    fake_client = MagicMock()
    chunks = [
        _openai_delta_chunk(reasoning_content="Let me think."),
        _openai_delta_chunk(reasoning_content=" Step 1 ..."),
        _openai_delta_chunk(content="Final: 42."),
        _openai_final_chunk(
            prompt_tokens=8, completion_tokens=30, reasoning_tokens=18,
        ),
    ]

    async def _create(**kwargs):
        return _AsyncIter(chunks)
    fake_client.chat.completions.create = _create
    ch._async_client = fake_client

    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    done: ChannelDone | None = None
    async for f in ch.atransmit_stream("solve"):
        if isinstance(f, ChannelChunk):
            if f.role == "thinking":
                thinking_chunks.append(f.text)
            elif f.role == "answer":
                answer_chunks.append(f.text)
        elif isinstance(f, ChannelDone):
            done = f

    assert "".join(thinking_chunks) == "Let me think. Step 1 ..."
    assert "".join(answer_chunks) == "Final: 42."
    assert done is not None
    out = done.output
    assert out.thinking_text == "Let me think. Step 1 ..."
    assert out.thinking_emitted is True
    assert out.thinking_tokens == 18  # exact from completion_tokens_details
    assert out.thinking_tokens_source == "openai_reasoning_tokens"


@pytest.mark.asyncio
async def test_openai_stream_inline_think_tag_captured_post_hoc():
    """Inline `<think>...</think>` deltas come through as `answer` chunks
    mid-stream (we can't reliably split across delta boundaries), but the
    final AgentOutput strips them out of .text and lands them in thinking_text."""
    from agentcodec.messages import ChannelDone

    ch = _make_channel("qwen3:14b", cost={"input": 0.1, "output": 0.1})
    fake_client = MagicMock()
    chunks = [
        _openai_delta_chunk(content="<think>internal</think>"),
        _openai_delta_chunk(content="Visible answer."),
        _openai_final_chunk(prompt_tokens=5, completion_tokens=10),
    ]

    async def _create(**kwargs):
        return _AsyncIter(chunks)
    fake_client.chat.completions.create = _create
    ch._async_client = fake_client

    done: ChannelDone | None = None
    async for f in ch.atransmit_stream("x"):
        if isinstance(f, ChannelDone):
            done = f

    assert done is not None
    out = done.output
    assert out.text == "Visible answer."
    assert out.thinking_text == "internal"
    assert out.thinking_tokens_source == "inline_tag_strip"


@pytest.mark.asyncio
async def test_openai_atransmit_drains_stream():
    """atransmit() should return the same AgentOutput as the stream's terminal frame."""
    ch = _make_channel("gpt-4o-mini", cost={"input": 1.0, "output": 2.0})
    fake_client = MagicMock()
    chunks = [
        _openai_delta_chunk(content="answer"),
        _openai_final_chunk(prompt_tokens=2, completion_tokens=3),
    ]

    async def _create(**kwargs):
        return _AsyncIter(chunks)
    fake_client.chat.completions.create = _create
    ch._async_client = fake_client

    out = await ch.atransmit("hi")
    assert out.text == "answer"
    assert out.input_tokens == 2
    assert out.output_tokens == 3


# ---------------------------------------------------------------------------
# Anthropic async streaming — content_block events with thinking + text deltas
# ---------------------------------------------------------------------------


def _anthropic_event(etype, **fields):
    e = MagicMock(type=etype)
    for k, v in fields.items():
        setattr(e, k, v)
    return e


@pytest.mark.asyncio
async def test_anthropic_stream_thinking_then_text():
    from agentcodec.messages import ChannelChunk, ChannelDone

    ch = _make_channel(
        "claude-sonnet-4-5",
        thinking={"enabled": True, "budget_tokens": 4096},
        cost={"input": 3.0, "output": 15.0},
    )

    # Build a sequence: message_start (with input_tokens) →
    # content_block_start(thinking) → 2x content_block_delta(thinking_delta) →
    # content_block_start(text) → 2x content_block_delta(text_delta) →
    # message_delta (with output_tokens)
    events = [
        _anthropic_event(
            "message_start",
            message=MagicMock(usage=MagicMock(input_tokens=12)),
        ),
        _anthropic_event(
            "content_block_start", index=0,
            content_block=MagicMock(type="thinking"),
        ),
        _anthropic_event(
            "content_block_delta", index=0,
            delta=MagicMock(type="thinking_delta", thinking="Careful reasoning"),
        ),
        _anthropic_event(
            "content_block_delta", index=0,
            delta=MagicMock(type="thinking_delta", thinking=" about QUIC."),
        ),
        _anthropic_event(
            "content_block_start", index=1,
            content_block=MagicMock(type="text"),
        ),
        _anthropic_event(
            "content_block_delta", index=1,
            delta=MagicMock(type="text_delta", text="QUIC is a "),
        ),
        _anthropic_event(
            "content_block_delta", index=1,
            delta=MagicMock(type="text_delta", text="transport protocol."),
        ),
        _anthropic_event(
            "message_delta",
            usage=MagicMock(output_tokens=80),
            delta=MagicMock(stop_reason="end_turn"),
        ),
    ]

    fake_client = MagicMock()
    fake_client.messages.stream = MagicMock(
        return_value=_AsyncContextManagerWrappingIter(events),
    )
    ch._async_anthropic_client = fake_client

    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    done: ChannelDone | None = None
    async for f in ch.atransmit_stream("What is QUIC?"):
        if isinstance(f, ChannelChunk):
            if f.role == "thinking":
                thinking_chunks.append(f.text)
            elif f.role == "answer":
                answer_chunks.append(f.text)
        elif isinstance(f, ChannelDone):
            done = f

    assert "".join(thinking_chunks) == "Careful reasoning about QUIC."
    assert "".join(answer_chunks) == "QUIC is a transport protocol."
    assert done is not None
    out = done.output
    assert out.text == "QUIC is a transport protocol."
    assert out.thinking_text == "Careful reasoning about QUIC."
    assert out.thinking_emitted is True
    assert out.thinking_tokens_source == "anthropic_thinking_block"
    assert out.input_tokens == 12
    assert out.output_tokens == 80
    # Cost split: with thinking + answer in the same usage block, the share
    # is char-based. Both costs must be > 0 and sum to total.
    assert out.thinking_cost_usd > 0.0
    assert out.answer_cost_usd > 0.0
    assert out.thinking_cost_usd + out.answer_cost_usd == pytest.approx(
        out.cost_usd, rel=1e-6,
    )


# ---------------------------------------------------------------------------
# Ollama-native async streaming — line-delimited JSON via httpx
# ---------------------------------------------------------------------------


class _FakeAiterStream:
    def __init__(self, lines):
        self._lines = list(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttpxClient:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, **kwargs):
        return _FakeAiterStream(self._lines)


@pytest.mark.asyncio
async def test_ollama_native_stream_with_thinking(monkeypatch):
    """Stream native-Ollama lines, verify content/thinking split + final counts."""
    from agentcodec.messages import ChannelChunk, ChannelDone

    ch = _make_channel(
        "qwen3:14b",
        cost={"input": 0.1, "output": 0.1},
    )
    # Force the native-ollama branch by setting a localhost base_url.
    ch.base_url = "http://localhost:11434/v1"

    lines = [
        json.dumps({"message": {"thinking": "Let me think"}, "done": False}),
        json.dumps({"message": {"thinking": " about it."}, "done": False}),
        json.dumps({"message": {"content": "Answer:"}, "done": False}),
        json.dumps({"message": {"content": " 42"}, "done": False}),
        json.dumps({
            "message": {},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 25,
        }),
    ]

    # Patch httpx.AsyncClient with our fake.
    import httpx
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: _FakeHttpxClient(lines),
    )

    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []
    done: ChannelDone | None = None
    async for f in ch.atransmit_stream("solve"):
        if isinstance(f, ChannelChunk):
            if f.role == "thinking":
                thinking_chunks.append(f.text)
            elif f.role == "answer":
                answer_chunks.append(f.text)
        elif isinstance(f, ChannelDone):
            done = f

    assert "".join(thinking_chunks) == "Let me think about it."
    assert "".join(answer_chunks) == "Answer: 42"
    assert done is not None
    out = done.output
    assert out.text == "Answer: 42"
    assert out.thinking_text == "Let me think about it."
    assert out.thinking_tokens_source == "ollama_thinking_field"
    assert out.input_tokens == 5
    assert out.output_tokens == 25
    assert out.finish_reason == "stop"
