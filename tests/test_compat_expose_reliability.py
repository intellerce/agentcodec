"""Phase 4 unit tests — ``expose_reliability_stream=`` flag on compat shims.

Without the flag (default), only ``role="answer"`` / ``"synthesis"`` /
``"thinking"`` TokenEvents flow through the native stream. Internal
roles (``"draft"``, ``"critique"``, ``"verification"``, ``"candidate"``)
get dropped. With the flag, internal roles flow through with sentinel
``agentcodec_role`` and ``agentcodec_call_id`` attributes for power users.

Tests directly exercise the adapters (``stream_from_event_iter`` /
``stream_chat_dicts``) with synthetic Event streams so no real ReliabilityModule
build is needed.
"""
from __future__ import annotations

from agentcodec.results import FinalEvent, ProgressEvent, ReliabilityResult, TokenEvent


def _events_with_all_roles() -> list:
    """A representative event stream that exercises every TokenEvent role."""
    return [
        ProgressEvent(stage="dispatch_start", detail={"technique": "harq_ir"}),
        TokenEvent(text="reasoning ", role="thinking", model="m", call_id="harq_ir:round1:gen"),
        TokenEvent(text="Initial draft answer", role="answer", model="m", call_id="harq_ir:round1:gen"),
        TokenEvent(text="critic notes...", role="critique", model="m", call_id="harq_ir:round2:critic"),
        TokenEvent(text="Refined draft", role="draft", model="m", call_id="harq_ir:round2:gen"),
        FinalEvent(result=ReliabilityResult(
            text="Refined draft", technique_used="harq_ir",
            input_tokens=10, output_tokens=20,
        )),
    ]


# ---------------------------------------------------------------------------
# OpenAI compat shim
# ---------------------------------------------------------------------------


def test_openai_default_drops_internal_roles():
    """Default: thinking → reasoning_content; answer → content; drafts + critiques dropped."""
    from agentcodec.openai._responses import stream_from_event_iter

    chunks = list(stream_from_event_iter(iter(_events_with_all_roles()), model="gpt-4o"))
    # We expect: 1 thinking chunk, 1 answer chunk, 1 terminal chunk = 3.
    assert len(chunks) == 3
    # Chunk 0: thinking via reasoning_content (no content set)
    delta0 = chunks[0].choices[0].delta
    assert getattr(delta0, "reasoning_content", None) == "reasoning "
    assert getattr(delta0, "content", None) is None
    # Chunk 1: answer via content
    delta1 = chunks[1].choices[0].delta
    assert getattr(delta1, "content", None) == "Initial draft answer"
    assert getattr(delta1, "reasoning_content", None) is None
    # Drafts + critiques dropped — they're not in the chunk list.
    # Terminal finish chunk.
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_openai_expose_surfaces_all_roles_with_sentinels():
    """expose_reliability_stream=True → all roles emit with agentcodec_role marker."""
    from agentcodec.openai._responses import stream_from_event_iter

    chunks = list(stream_from_event_iter(
        iter(_events_with_all_roles()), model="gpt-4o",
        expose_reliability_stream=True,
    ))
    # 4 token chunks + 1 terminal = 5
    assert len(chunks) == 5
    roles_seen = []
    for c in chunks[:-1]:
        delta = c.choices[0].delta
        roles_seen.append(getattr(delta, "agentcodec_role", None))
        # All should carry the sentinel call_id too
        assert getattr(delta, "agentcodec_call_id", None) is not None
    assert roles_seen == ["thinking", "answer", "critique", "draft"]
    # critique + draft must end up as content (not reasoning_content)
    for c in chunks[2:4]:
        delta = c.choices[0].delta
        assert getattr(delta, "content", None) is not None
        assert getattr(delta, "reasoning_content", None) is None


def test_openai_synthesis_routes_to_content():
    """role=synthesis should map to delta.content (same as answer)."""
    from agentcodec.openai._responses import stream_from_event_iter

    events = [
        TokenEvent(text="merged ", role="synthesis", model="m", call_id="diversity_mrc:synth"),
        FinalEvent(result=ReliabilityResult(text="merged", technique_used="diversity_mrc")),
    ]
    chunks = list(stream_from_event_iter(iter(events), model="gpt-4o"))
    assert len(chunks) == 2
    delta = chunks[0].choices[0].delta
    assert getattr(delta, "content", None) == "merged "


# ---------------------------------------------------------------------------
# Anthropic compat shim
# ---------------------------------------------------------------------------


def test_anthropic_default_routes_thinking_to_separate_block():
    """Default: thinking goes via thinking_delta on a side block; answer on block 0."""
    from agentcodec.anthropic._responses import stream_from_event_iter

    events = list(stream_from_event_iter(
        iter(_events_with_all_roles()), model="claude-sonnet-4-5",
    ))
    # Expected sequence:
    #   message_start, content_block_start(0=text),
    #   content_block_start(1=thinking), content_block_delta(1, thinking_delta),
    #   content_block_delta(0, text_delta) [answer],
    #   content_block_stop(1), content_block_stop(0),
    #   message_delta, message_stop
    [getattr(e, "type", None) for e in events]
    # Drafts and critiques must NOT appear as text_deltas.
    text_deltas = [
        e for e in events
        if e.type == "content_block_delta"
        and getattr(e.delta, "type", None) == "text_delta"
    ]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta.text == "Initial draft answer"
    # Thinking delta lands on a separate block index.
    thinking_deltas = [
        e for e in events
        if e.type == "content_block_delta"
        and getattr(e.delta, "type", None) == "thinking_delta"
    ]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0].delta.thinking == "reasoning "
    assert thinking_deltas[0].index == 1


def test_anthropic_expose_surfaces_all_roles_with_sentinels():
    """expose=True: drafts + critiques surface as text_delta with sentinel attrs."""
    from agentcodec.anthropic._responses import stream_from_event_iter

    events = list(stream_from_event_iter(
        iter(_events_with_all_roles()), model="claude-sonnet-4-5",
        expose_reliability_stream=True,
    ))
    text_deltas = [
        e for e in events
        if e.type == "content_block_delta"
        and getattr(e.delta, "type", None) == "text_delta"
    ]
    # 3 text_deltas (answer + critique + draft) — thinking still goes to its own block.
    assert len(text_deltas) == 3
    roles = [getattr(d.delta, "agentcodec_role", None) for d in text_deltas]
    assert roles == ["answer", "critique", "draft"]


# ---------------------------------------------------------------------------
# Ollama compat shim
# ---------------------------------------------------------------------------


def test_ollama_default_routes_thinking_to_thinking_field():
    """Default: thinking → message.thinking; answer → message.content; rest dropped."""
    from agentcodec.ollama._responses import stream_chat_dicts

    chunks = list(stream_chat_dicts(iter(_events_with_all_roles()), model="qwen3:14b"))
    # 1 thinking + 1 answer + 1 terminal = 3
    assert len(chunks) == 3
    assert chunks[0]["message"].get("thinking") == "reasoning "
    assert chunks[0]["message"].get("content") in (None, "")
    assert chunks[1]["message"]["content"] == "Initial draft answer"
    assert chunks[-1]["done"] is True


def test_ollama_expose_surfaces_all_roles():
    """expose=True: drafts + critiques surface as content with agentcodec_role."""
    from agentcodec.ollama._responses import stream_chat_dicts

    chunks = list(stream_chat_dicts(
        iter(_events_with_all_roles()), model="qwen3:14b",
        expose_reliability_stream=True,
    ))
    # 4 token chunks + 1 terminal
    assert len(chunks) == 5
    roles = [c["message"].get("agentcodec_role") for c in chunks[:-1]]
    assert roles == ["thinking", "answer", "critique", "draft"]
    # All carry the call_id sentinel too.
    for c in chunks[:-1]:
        assert c["message"].get("agentcodec_call_id") is not None
