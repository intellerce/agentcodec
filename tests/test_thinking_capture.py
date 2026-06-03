"""Phase 1 unit tests — thinking_text + telemetry capture across backends.

Each backend's transmit path is exercised with a mocked SDK response that
emits thinking content via that backend's native channel (Anthropic
ThinkingBlock, OpenAI reasoning_content, Ollama msg.thinking) and via
inline `<think>...</think>` tags. We assert that AgentOutput's thinking_*
fields land correctly and that ReliabilityResult / trace surface them.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_split_inline_thinking_closed_block():
    from agentcodec.channel import _split_inline_thinking

    clean, thought = _split_inline_thinking(
        "intro <think>internal reasoning here</think> answer text"
    )
    assert clean == "intro  answer text"
    assert thought == "internal reasoning here"


def test_split_inline_thinking_multiple_blocks():
    from agentcodec.channel import _split_inline_thinking

    clean, thought = _split_inline_thinking(
        "a<think>one</think>b<thinking>two</thinking>c"
    )
    assert clean == "abc"
    # Order preserved, joined with newlines
    assert "one" in thought and "two" in thought


def test_split_inline_thinking_unclosed_block():
    """Model hit max_tokens mid-thinking — the unclosed remainder is captured."""
    from agentcodec.channel import _split_inline_thinking

    clean, thought = _split_inline_thinking(
        "preamble <think>I was reasoning when I ran out"
    )
    assert clean == "preamble"
    assert thought == "I was reasoning when I ran out"


def test_split_inline_thinking_no_tags():
    from agentcodec.channel import _split_inline_thinking

    assert _split_inline_thinking("just an answer") == ("just an answer", "")
    assert _split_inline_thinking("") == ("", "")


def test_split_inline_thinking_mirrors_strip():
    """Whatever _split removes from the clean side must equal what
    QualityScorer._strip_thinking would produce — paired behavior."""
    from agentcodec.channel import QualityScorer, _split_inline_thinking

    samples = [
        "head <think>secret</think> tail",
        "<thinking>just thinking</thinking>",
        "no tags",
        "open <think>unclosed forever",
        "<analysis>x</analysis> y <reflection>z</reflection>",
    ]
    for s in samples:
        stripped = QualityScorer._strip_thinking(s)
        clean, _ = _split_inline_thinking(s)
        assert clean == stripped, f"mismatch on {s!r}: split={clean!r} strip={stripped!r}"


# ---------------------------------------------------------------------------
# _build_thinking_kwargs unit tests (decoupled from SDK mocking)
# ---------------------------------------------------------------------------


def _make_channel(model: str = "gpt-4o-mini", *, thinking=None, cost=None):
    """Construct an AgentChannel with the SDKs mocked out."""
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = MagicMock(name="openai.OpenAI")
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = MagicMock(name="anthropic.Anthropic")
    sys.modules["openai"] = fake_openai
    sys.modules["anthropic"] = fake_anthropic

    from agentcodec.channel import AgentChannel

    return AgentChannel(
        model=model,
        api_key="test",
        thinking=thinking,
        cost_per_1m=cost,
    )


def test_build_thinking_kwargs_no_thinking():
    ch = _make_channel("gpt-4o-mini", cost={"input": 1.0, "output": 2.0})
    kw = ch._build_thinking_kwargs(
        thinking_text="", answer_text="hello",
        input_tokens=10, output_tokens=20,
    )
    assert kw["thinking_emitted"] is False
    assert kw["thinking_text"] is None
    assert kw["thinking_tokens"] == 0
    assert kw["thinking_tokens_source"] is None
    assert kw["thinking_cost_usd"] == 0.0
    assert kw["answer_tokens"] == 20
    # Full output billed as answer.
    assert kw["answer_cost_usd"] == pytest.approx(10 * 1.0 / 1e6 + 20 * 2.0 / 1e6)


def test_build_thinking_kwargs_exact_tokens():
    ch = _make_channel("gpt-5", cost={"input": 1.0, "output": 10.0})
    kw = ch._build_thinking_kwargs(
        thinking_text="hidden",
        answer_text="answer",
        input_tokens=100,
        output_tokens=200,
        thinking_tokens_exact=150,
        thinking_tokens_source="openai_reasoning_tokens",
    )
    assert kw["thinking_emitted"] is True
    assert kw["thinking_text"] == "hidden"
    assert kw["thinking_tokens"] == 150
    assert kw["thinking_tokens_source"] == "openai_reasoning_tokens"
    # Cost split: 75% of output cost is thinking.
    assert kw["thinking_cost_usd"] == pytest.approx(200 * 10.0 / 1e6 * 0.75)
    assert kw["answer_tokens"] == 50


def test_build_thinking_kwargs_char_share_estimate():
    ch = _make_channel("gpt-4o-mini", cost={"input": 1.0, "output": 2.0})
    # thinking 70 chars, answer 30 chars → thinking ≈ 70% of 100 output tokens = 70.
    kw = ch._build_thinking_kwargs(
        thinking_text="x" * 70,
        answer_text="y" * 30,
        input_tokens=10,
        output_tokens=100,
    )
    assert kw["thinking_emitted"] is True
    assert kw["thinking_tokens"] == 70
    assert kw["thinking_tokens_source"] == "char_share_estimate"
    assert kw["answer_tokens"] == 30


# ---------------------------------------------------------------------------
# Anthropic transmit path — ThinkingBlock capture
# ---------------------------------------------------------------------------


class _FakeAnthropicBlock:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


def _make_anthropic_response(*, thinking_text="", answer_text="", input_tokens=5, output_tokens=15):
    blocks: list[Any] = []
    if thinking_text:
        blocks.append(_FakeAnthropicBlock(
            type="thinking", text=thinking_text, thinking=thinking_text,
        ))
    blocks.append(_FakeAnthropicBlock(type="text", text=answer_text))
    return MagicMock(
        content=blocks,
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def test_anthropic_thinking_block_captured():
    ch = _make_channel(
        model="claude-sonnet-4-5",
        thinking={"enabled": True, "budget_tokens": 4096},
        cost={"input": 3.0, "output": 15.0},
    )
    ch.anthropic_client.messages.create.return_value = _make_anthropic_response(
        thinking_text="careful reasoning about QUIC",
        answer_text="QUIC is a transport protocol.",
        input_tokens=12,
        output_tokens=80,
    )
    out = ch.transmit("What is QUIC?")
    assert out.text == "QUIC is a transport protocol."
    assert out.thinking_emitted is True
    assert out.thinking_text == "careful reasoning about QUIC"
    assert out.thinking_chars == len("careful reasoning about QUIC")
    assert out.thinking_supported is True
    assert out.thinking_enabled is True
    assert out.thinking_tokens > 0
    assert out.thinking_tokens_source == "anthropic_thinking_block"
    assert out.thinking_cost_usd > 0.0
    assert out.answer_tokens > 0
    assert out.answer_cost_usd > 0.0
    # Sanity: split sums to total output cost.
    assert out.thinking_cost_usd + out.answer_cost_usd == pytest.approx(
        out.cost_usd, rel=1e-6,
    )


def test_anthropic_no_thinking_block():
    ch = _make_channel(model="claude-sonnet-4-5")
    ch.anthropic_client.messages.create.return_value = _make_anthropic_response(
        thinking_text="", answer_text="plain answer", input_tokens=5, output_tokens=10,
    )
    out = ch.transmit("hi")
    assert out.text == "plain answer"
    assert out.thinking_emitted is False
    assert out.thinking_text is None
    assert out.thinking_tokens == 0


# ---------------------------------------------------------------------------
# OpenAI transmit path — reasoning_content + inline <think> tag
# ---------------------------------------------------------------------------


def _make_openai_response(
    *,
    content: str = "",
    reasoning_content: str | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    reasoning_tokens: int | None = None,
    finish_reason: str = "stop",
):
    # Pin BOTH reasoning attrs explicitly — leaving them implicit makes
    # MagicMock fabricate truthy children on access, which then poisons the
    # channel's `getattr(..., "reasoning", None) or ""` fallback.
    message = MagicMock(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=None,
    )
    choice = MagicMock(
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
    )
    if reasoning_tokens is not None:
        usage_details = MagicMock(reasoning_tokens=reasoning_tokens)
    else:
        usage_details = None
    usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=usage_details,
    )
    return MagicMock(choices=[choice], usage=usage)


def test_openai_reasoning_content_captured():
    ch = _make_channel("gpt-5", cost={"input": 1.0, "output": 10.0})
    ch.client.chat.completions.create.return_value = _make_openai_response(
        content="The capital of France is Paris.",
        reasoning_content="Let me think: France's capital is Paris.",
        prompt_tokens=20,
        completion_tokens=100,
        reasoning_tokens=60,
    )
    out = ch.transmit("Capital of France?")
    assert "Paris" in out.text
    assert out.thinking_emitted is True
    assert out.thinking_text and "France" in out.thinking_text
    assert out.thinking_tokens == 60
    assert out.thinking_tokens_source == "openai_reasoning_tokens"
    # Cost split should yield exact 60% thinking.
    expected_thinking_cost = 100 * 10.0 / 1e6 * 0.6
    assert out.thinking_cost_usd == pytest.approx(expected_thinking_cost)


def test_openai_inline_think_tag_captured():
    """Older OpenAI-compat backends (vLLM serving qwen3, Ollama) emit
    `<think>...</think>` inline in `content`. The split helper must capture
    it AND strip it from the answer."""
    ch = _make_channel("qwen3:14b", cost={"input": 0.1, "output": 0.1})
    ch.client.chat.completions.create.return_value = _make_openai_response(
        content="<think>step 1: x. step 2: y.</think>Final answer: 42.",
        prompt_tokens=8,
        completion_tokens=30,
    )
    out = ch.transmit("solve it")
    assert "step 1" not in out.text and "Final answer: 42." in out.text
    assert out.thinking_emitted is True
    assert out.thinking_text and "step 1" in out.thinking_text
    assert out.thinking_tokens_source == "inline_tag_strip"


# ---------------------------------------------------------------------------
# Trace + ReliabilityResult surfacing
# ---------------------------------------------------------------------------


def test_reliability_result_surfaces_thinking_text():
    from agentcodec.models import AgentOutput, ReliabilityRun
    from agentcodec.results import build_result_from_run

    out = AgentOutput(
        text="answer",
        model="claude-opus-4",
        temperature=0.7,
        thinking_emitted=True,
        thinking_text="this is the captured reasoning",
        thinking_tokens=42,
        thinking_chars=len("this is the captured reasoning"),
        thinking_cost_usd=0.0050,
        answer_tokens=10,
        answer_cost_usd=0.0010,
        cost_usd=0.0060,
        input_tokens=5,
        output_tokens=52,
    )
    run = ReliabilityRun(
        technique="baseline",
        individual_outputs=[out],
        combined_output="answer",
        total_cost_usd=0.0060,
        num_llm_calls=1,
    )
    result = build_result_from_run(
        run, technique_used="baseline", wall_clock_s=0.5, return_trace=True,
    )
    assert result.thinking_used is True
    assert result.thinking_text == "this is the captured reasoning"
    assert result.thinking_tokens == 42
    assert result.thinking_cost_usd == pytest.approx(0.0050)
    # Trace exposes per-call text too.
    assert result.trace["calls"][0]["thinking"]["text"] == "this is the captured reasoning"
    assert result.trace["totals"]["thinking_cost_usd"] == pytest.approx(0.0050)
    # to_dict round-trip.
    d = result.to_dict()
    assert d["thinking_text"] == "this is the captured reasoning"
    assert d["thinking_cost_usd"] == pytest.approx(0.0050)


def test_reliability_result_aggregates_thinking_across_calls():
    """Two outputs that both emitted thinking → result.thinking_text joins
    them with a separator."""
    from agentcodec.models import AgentOutput, ReliabilityRun
    from agentcodec.results import build_result_from_run

    o1 = AgentOutput(
        text="a", model="m1", temperature=0.7,
        thinking_emitted=True, thinking_text="reasoning from branch 1",
        thinking_tokens=5, thinking_cost_usd=0.001, output_tokens=10,
    )
    o2 = AgentOutput(
        text="b", model="m2", temperature=0.7,
        thinking_emitted=True, thinking_text="reasoning from branch 2",
        thinking_tokens=7, thinking_cost_usd=0.002, output_tokens=14,
    )
    run = ReliabilityRun(
        technique="diversity_mrc",
        individual_outputs=[o1, o2],
        combined_output="combined",
        num_llm_calls=2,
    )
    result = build_result_from_run(run, technique_used="diversity_mrc", wall_clock_s=1.0)
    assert "branch 1" in result.thinking_text
    assert "branch 2" in result.thinking_text
    assert result.thinking_tokens == 12
    assert result.thinking_cost_usd == pytest.approx(0.003)
