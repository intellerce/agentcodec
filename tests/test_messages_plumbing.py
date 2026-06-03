"""
Smoke tests for the `messages=` / `system=` / `tools=` plumbing on
:meth:`ReliabilityModule.run` and the back-compat string path. These tests
do not call any real LLM — they wire the dispatch context with the same
MockChannel fixtures the technique smoke tests use, then assert about
what the channel saw.

The fixtures live in tests/conftest.py.
"""
from __future__ import annotations

import pytest

from agentcodec.dispatch import dispatch
from agentcodec.messages import ChatRequest
from agentcodec.models import TaskCategory, TaskItem

# ---------------------------------------------------------------------------
# Direct ChatRequest threading: technique receives the request via TaskItem
# ---------------------------------------------------------------------------


def test_taskitem_auto_builds_request_from_prompt():
    """The back-compat path: bare prompt yields a ChatRequest with a single
    user message and no system."""
    t = TaskItem(id="t1", category=TaskCategory.QA, prompt="hello")
    assert t.request is not None
    assert t.request.system is None
    assert t.request.last_user_text == "hello"


def test_taskitem_preserves_explicit_request():
    """An explicit request= kwarg wins over the auto-build path."""
    req = ChatRequest.from_prompt("ignored", system="be terse")
    t = TaskItem(
        id="t2", category=TaskCategory.QA, prompt="hello", request=req,
    )
    assert t.request is req
    assert t.request.system == "be terse"


# ---------------------------------------------------------------------------
# End-to-end via the dispatcher: MockChannel sees a ChatRequest with system
# ---------------------------------------------------------------------------


def test_baseline_dispatch_passes_request_to_channel(dispatch_ctx, mock_channel_pair):
    """The baseline technique calls channel.transmit() with `task.prompt`
    (a string), which AgentChannel.transmit normalizes to a ChatRequest.
    Through the MockChannel, we observe the recorded call shape."""
    req = ChatRequest.from_prompt(
        "What is 2+2?", system="Always answer in one number.",
    )
    task = TaskItem(
        id="task_a", category=TaskCategory.QA,
        prompt=req.last_user_text, request=req,
    )
    run = dispatch("baseline", task, dispatch_ctx)
    assert run.individual_outputs, "baseline produced no output"
    # The MockChannel records each transmit call; assert it saw the prompt.
    seen_prompts = [
        c.get("prompt") for c in mock_channel_pair[0].calls
    ]
    assert any("What is 2+2?" in (p or "") for p in seen_prompts), seen_prompts


# ---------------------------------------------------------------------------
# from_openai_messages adapter
# ---------------------------------------------------------------------------


def test_from_openai_messages_with_system_and_history():
    msgs = [
        {"role": "system", "content": "You are a librarian."},
        {"role": "user", "content": "Recommend a book."},
        {"role": "assistant", "content": "The Stranger."},
        {"role": "user", "content": "Why?"},
    ]
    req = ChatRequest.from_openai_messages(msgs)
    assert req.system == "You are a librarian."
    assert req.last_user_text == "Why?"
    assert len(req.history) == 3   # system + user + assistant


def test_with_user_preserves_system_and_history():
    """The keystone invariant: mutating the user turn preserves everything else."""
    req = ChatRequest.from_openai_messages([
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Define entropy."},
    ])
    mutated = req.with_user("Critique your previous answer.")
    assert mutated.system == "Be precise."
    assert mutated.last_user_text == "Critique your previous answer."
    # history (everything except final user) keeps the system message
    assert len(mutated.history) == 1
    assert mutated.history[0].role == "system"


# ---------------------------------------------------------------------------
# Tools / stop / seed / top_p threading
# ---------------------------------------------------------------------------


def test_tools_and_stop_thread_through_to_request():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]
    req = ChatRequest.from_openai_messages(
        [{"role": "user", "content": "Weather in SF?"}],
        tools=tools, tool_choice="auto",
        stop=["</done>"], seed=42, top_p=0.9, response_format={"type": "json_object"},
    )
    assert req.tools is not None and len(req.tools) == 1
    assert req.tools[0]["function"]["name"] == "get_weather"
    assert req.tool_choice == "auto"
    assert req.stop == ("</done>",)
    assert req.seed == 42
    assert req.top_p == 0.9
    assert req.response_format == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Anthropic / Ollama adapters round-trip
# ---------------------------------------------------------------------------


def test_anthropic_round_trip():
    """Build from Anthropic's shape (system + messages), serialize back, get a
    payload Anthropic can consume."""
    req = ChatRequest.from_anthropic(
        system="You are concise.",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=256,
    )
    sys_text, msgs = req.to_anthropic_payload()
    assert sys_text == "You are concise."
    assert msgs == [{"role": "user", "content": "Hello"}]


def test_ollama_round_trip():
    req = ChatRequest.from_ollama_messages(
        [{"role": "user", "content": "ping"}],
        options={"temperature": 0.3, "seed": 7},
    )
    assert req.temperature == 0.3
    assert req.seed == 7
    out = req.to_ollama_messages()
    assert out == [{"role": "user", "content": "ping"}]


# ---------------------------------------------------------------------------
# `prompt=` and `messages=` are mutually exclusive in mod.run
# ---------------------------------------------------------------------------


def test_run_rejects_both_prompt_and_messages(monkeypatch, mock_channel, mock_scorer):
    """Smoke-check the validation path on the public API."""
    from agentcodec.api import ReliabilityModule
    from agentcodec.config import (
        FixedStrategy,
        JudgeConfig,
        LibraryConfig,
        ModelConfig,
    )

    # Bypass real-channel construction.
    cfg = LibraryConfig(
        models=[ModelConfig(model="gpt-4o-mini", api_key="dummy")],
        judge=JudgeConfig(model="gpt-4o-mini", api_key="dummy"),
        strategy=FixedStrategy(type="fixed", technique="baseline"),
    )
    monkeypatch.setattr(
        ReliabilityModule, "_build_scorer", lambda self, jcfg: mock_scorer,
    )
    monkeypatch.setattr(
        "agentcodec.api.AgentChannel", lambda **kw: mock_channel,
    )
    mod = ReliabilityModule(cfg)
    with pytest.raises(ValueError, match="exactly one of prompt= or messages="):
        mod.run()  # neither
    with pytest.raises(ValueError, match="exactly one of prompt= or messages="):
        mod.run("hi", messages=[{"role": "user", "content": "hi"}])
