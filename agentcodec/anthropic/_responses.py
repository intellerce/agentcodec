"""
Build Anthropic-shaped response objects from a :class:`ReliabilityResult`,
and translate our :class:`Event` stream into Anthropic-shaped chunks.

Anthropic's response is *block-structured* — ``response.content`` is a
list of ``TextBlock`` / ``ToolUseBlock`` instances — and the SDK uses
``stop_reason``/``input_tokens``/``output_tokens`` rather than OpenAI's
``finish_reason``/``prompt_tokens``/``completion_tokens``. The adapters
preserve that shape so downstream attribute access (``resp.content[0].text``,
``resp.stop_reason``, ``resp.usage.input_tokens``) keeps working.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from ..results import Event, FinalEvent, ProgressEvent, ReliabilityResult, TokenEvent


def message_from_result(
    result: ReliabilityResult,
    *,
    model: str,
) -> SimpleNamespace:
    """Build a ``Message``-shaped object (Anthropic ``messages.create`` return)."""
    content_blocks: list[SimpleNamespace] = []
    text = result.text or ""
    if text:
        content_blocks.append(SimpleNamespace(type="text", text=text))
    for call in _collect_tool_calls(result):
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            args = {"_raw": call.arguments}
        content_blocks.append(SimpleNamespace(
            type="tool_use",
            id=call.id,
            name=call.name,
            input=args,
        ))
    usage = SimpleNamespace(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(
        id=f"msg_agentcodec_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        model=model,
        content=content_blocks,
        stop_reason=_anthropic_stop_reason(result, content_blocks),
        stop_sequence=None,
        usage=usage,
        reliability=result,
    )


# See OpenAI shim for the rationale on which roles surface in the native
# stream. Anthropic emits ``thinking`` as a separate content block via
# ``thinking_delta``, so we maintain a side block index when thinking
# fires; ``answer``/``synthesis`` go on the canonical text block.
_ANTHROPIC_ANSWER_ROLES: frozenset[str] = frozenset({"answer", "synthesis"})


def stream_from_event_iter(
    events: Iterator[Event],
    *,
    model: str,
    expose_reliability_stream: bool = False,
) -> Iterator[SimpleNamespace]:
    """Adapt our reliability event stream into Anthropic-shaped events.

    Emits ``message_start``, ``content_block_start``, ``content_block_delta``
    (per ``TokenEvent``), ``content_block_stop``, ``message_delta``, and
    ``message_stop`` events — the same sequence the real Anthropic SDK
    yields.

    Default routing (``expose_reliability_stream=False``):
      * ``role="answer"`` or ``"synthesis"`` → ``text_delta`` on the
        answer block (index 0)
      * ``role="thinking"`` → ``thinking_delta`` on a separate thinking
        block (index 1, lazily opened on first thinking delta)
      * Other roles (``"draft"``, ``"critique"``, ``"candidate"``,
        ``"verification"``) are **dropped** — they're internal to the
        technique and exposing them in the answer block would jumble it.

    Opt-in (``expose_reliability_stream=True``):
      * All non-thinking roles surface as ``text_delta`` on the answer
        block AND additionally set ``delta.agentcodec_role`` and
        ``delta.agentcodec_call_id`` as sentinels. Existing consumers
        ignore unknown attributes; reliability-aware ones branch on them.
      * ``role="thinking"`` still routes to its own thinking block.
    """
    message_id = f"msg_agentcodec_{uuid.uuid4().hex[:24]}"
    final_result: ReliabilityResult | None = None

    yield SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            id=message_id, type="message", role="assistant", model=model,
            content=[],
            stop_reason=None, stop_sequence=None,
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        ),
    )
    # Answer block is always index 0. Thinking block is lazily opened at
    # index 1 the first time thinking content arrives.
    answer_block_idx = 0
    thinking_block_idx: int | None = None
    yield SimpleNamespace(
        type="content_block_start", index=answer_block_idx,
        content_block=SimpleNamespace(type="text", text=""),
    )
    for ev in events:
        if isinstance(ev, TokenEvent):
            role = ev.role
            if role == "thinking":
                if thinking_block_idx is None:
                    thinking_block_idx = 1
                    yield SimpleNamespace(
                        type="content_block_start", index=thinking_block_idx,
                        content_block=SimpleNamespace(type="thinking", thinking=""),
                    )
                delta_kwargs: dict[str, Any] = {
                    "type": "thinking_delta", "thinking": ev.text,
                }
                if expose_reliability_stream:
                    delta_kwargs["agentcodec_role"] = "thinking"
                    delta_kwargs["agentcodec_call_id"] = ev.call_id
                yield SimpleNamespace(
                    type="content_block_delta", index=thinking_block_idx,
                    delta=SimpleNamespace(**delta_kwargs),
                )
            elif role in _ANTHROPIC_ANSWER_ROLES or expose_reliability_stream:
                delta_kwargs = {"type": "text_delta", "text": ev.text}
                if expose_reliability_stream:
                    delta_kwargs["agentcodec_role"] = role
                    delta_kwargs["agentcodec_call_id"] = ev.call_id
                yield SimpleNamespace(
                    type="content_block_delta", index=answer_block_idx,
                    delta=SimpleNamespace(**delta_kwargs),
                )
            # else: internal role, default-hidden — drop the chunk.
        elif isinstance(ev, FinalEvent):
            final_result = ev.result
        elif isinstance(ev, ProgressEvent):
            continue
    if thinking_block_idx is not None:
        yield SimpleNamespace(type="content_block_stop", index=thinking_block_idx)
    yield SimpleNamespace(type="content_block_stop", index=answer_block_idx)
    stop_reason = "end_turn"
    output_tokens = 0
    if final_result is not None:
        stop_reason = "tool_use" if _collect_tool_calls(final_result) else "end_turn"
        output_tokens = final_result.output_tokens
    yield SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason, stop_sequence=None),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )
    yield SimpleNamespace(type="message_stop", reliability=final_result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_tool_calls(result: ReliabilityResult) -> list[Any]:
    trace = getattr(result, "trace", None) or {}
    outputs = trace.get("individual_outputs") or []
    for out in reversed(outputs):
        tc = out.get("tool_calls") if isinstance(out, dict) else None
        if tc:
            return list(tc)
    return []


def _anthropic_stop_reason(
    result: ReliabilityResult,
    content_blocks: list[Any],
) -> str:
    if any(getattr(b, "type", None) == "tool_use" for b in content_blocks):
        return "tool_use"
    fr = getattr(result, "finish_reason", None)
    if fr in ("length", "max_tokens"):
        return "max_tokens"
    return "end_turn"
