"""
Adapters that build OpenAI-shaped response objects from a
:class:`ReliabilityResult`, and translate our :class:`Event` stream into
OpenAI-shaped ``ChatCompletionChunk``-style dicts.

These are intentionally **duck-typed** rather than instances of
``openai.types.chat.ChatCompletion``. Importing the real types pulls
the whole openai SDK; the shim's value is that users can replace one
import without paying that cost on the passthrough path. The shapes
exposed (``.choices[0].message.content``, ``.usage.prompt_tokens``,
``.model``, etc.) match the OpenAI SDK exactly, so downstream attribute
access keeps working.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from ..results import Event, FinalEvent, ProgressEvent, ReliabilityResult, TokenEvent


def completion_from_result(
    result: ReliabilityResult,
    *,
    model: str,
) -> SimpleNamespace:
    """Build a ChatCompletion-shaped object from a ReliabilityResult.

    The returned object also carries a ``reliability`` attribute pointing
    back at the original :class:`ReliabilityResult` for users who want the
    technique trace, cost breakdown, or router decision.
    """
    tool_calls_payload: list[SimpleNamespace] = []
    # The reliability layer aggregates many internal LLM calls; we surface
    # tool_calls from the final answer-producing output when present.
    tc = _collect_tool_calls(result)
    if tc:
        for call in tc:
            tool_calls_payload.append(SimpleNamespace(
                id=call.id,
                type="function",
                function=SimpleNamespace(name=call.name, arguments=call.arguments),
            ))
    message = SimpleNamespace(
        role="assistant",
        content=result.text or "",
        tool_calls=tool_calls_payload or None,
        refusal=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=_finish_reason(result, tool_calls_payload),
        logprobs=None,
    )
    usage = SimpleNamespace(
        prompt_tokens=result.input_tokens,
        completion_tokens=result.output_tokens,
        total_tokens=result.input_tokens + result.output_tokens,
    )
    return SimpleNamespace(
        id=f"agentcodec-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
        system_fingerprint=None,
        # Custom escape hatch: original ReliabilityResult for power users.
        reliability=result,
    )


# Roles whose TokenEvent text is the user-facing answer in the native SDK
# shape: ``delta.content`` on OpenAI chunks. Other roles either map to
# provider-side reasoning channels (thinking) or are intentionally hidden
# from the native stream (drafts/critiques/etc. — internal to the technique).
_OPENAI_ANSWER_ROLES: frozenset[str] = frozenset({"answer", "synthesis"})


def stream_from_event_iter(
    events: Iterator[Event],
    *,
    model: str,
    expose_reliability_stream: bool = False,
) -> Iterator[SimpleNamespace]:
    """Adapt our reliability ``Event`` stream into OpenAI-shaped delta chunks.

    Each ``TokenEvent`` becomes a chunk with ``choices[0].delta.content``
    (for user-facing answer text) or ``choices[0].delta.reasoning_content``
    (for thinking deltas — matches OpenAI's o-series shape). The terminal
    ``FinalEvent`` becomes a finish chunk with ``choices[0].finish_reason``.
    ``ProgressEvent`` records are dropped — consumers that want them should
    iterate the native :meth:`ReliabilityModule.astream` instead.

    Default routing (``expose_reliability_stream=False``):
      * ``role="answer"`` or ``"synthesis"`` → ``delta.content``
      * ``role="thinking"`` → ``delta.reasoning_content``
      * Other roles (``"draft"``, ``"critique"``, ``"candidate"``,
        ``"verification"``) are **dropped** — they're internal to the
        technique and exposing them would jumble the answer stream.

    Opt-in (``expose_reliability_stream=True``):
      * All non-thinking roles become ``delta.content`` AND additionally
        set ``delta.agentcodec_role`` and ``delta.agentcodec_call_id`` as
        sentinel attributes. Existing OpenAI consumers ignore unknown
        ``delta`` attributes; reliability-aware consumers branch on them.
      * ``role="thinking"`` still routes to ``delta.reasoning_content``
        and additionally sets ``delta.agentcodec_role = "thinking"``.
    """
    completion_id = f"agentcodec-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    sent_role = False
    final_result: ReliabilityResult | None = None
    for ev in events:
        if isinstance(ev, TokenEvent):
            chunk = _build_openai_chunk(
                ev, completion_id, created_at, model,
                sent_role=sent_role,
                expose_reliability_stream=expose_reliability_stream,
            )
            if chunk is None:
                continue
            sent_role = True
            yield chunk
        elif isinstance(ev, FinalEvent):
            final_result = ev.result
        elif isinstance(ev, ProgressEvent):
            # Ignored on the OpenAI compat path — users who want them
            # should iterate the native stream instead.
            continue
    # Emit the terminal "done" chunk.
    finish_reason = "stop"
    usage = None
    if final_result is not None:
        tc = _collect_tool_calls(final_result)
        finish_reason = "tool_calls" if tc else "stop"
        usage = SimpleNamespace(
            prompt_tokens=final_result.input_tokens,
            completion_tokens=final_result.output_tokens,
            total_tokens=final_result.input_tokens + final_result.output_tokens,
        )
    yield SimpleNamespace(
        id=completion_id,
        object="chat.completion.chunk",
        created=created_at,
        model=model,
        choices=[SimpleNamespace(
            index=0,
            delta=SimpleNamespace(),
            finish_reason=finish_reason,
            logprobs=None,
        )],
        usage=usage,
        reliability=final_result,
    )


def _build_openai_chunk(
    ev: TokenEvent,
    completion_id: str,
    created_at: int,
    model: str,
    *,
    sent_role: bool,
    expose_reliability_stream: bool,
) -> SimpleNamespace | None:
    """Map one TokenEvent to an OpenAI-shaped chunk (or None to drop it)."""
    role = ev.role
    delta_kwargs: dict[str, Any] = {}
    if not sent_role:
        delta_kwargs["role"] = "assistant"

    if role == "thinking":
        delta_kwargs["reasoning_content"] = ev.text
        if expose_reliability_stream:
            delta_kwargs["agentcodec_role"] = "thinking"
            delta_kwargs["agentcodec_call_id"] = ev.call_id
    elif role in _OPENAI_ANSWER_ROLES:
        delta_kwargs["content"] = ev.text
        if expose_reliability_stream:
            delta_kwargs["agentcodec_role"] = role
            delta_kwargs["agentcodec_call_id"] = ev.call_id
    elif expose_reliability_stream:
        # Internal roles surfaced under opt-in: surface as content with a
        # sentinel so power users can demux without breaking default consumers.
        delta_kwargs["content"] = ev.text
        delta_kwargs["agentcodec_role"] = role
        delta_kwargs["agentcodec_call_id"] = ev.call_id
    else:
        # Default: hide internal-role chunks from the native stream.
        return None

    return SimpleNamespace(
        id=completion_id,
        object="chat.completion.chunk",
        created=created_at,
        model=model,
        choices=[SimpleNamespace(
            index=0,
            delta=SimpleNamespace(**delta_kwargs),
            finish_reason=None,
            logprobs=None,
        )],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_tool_calls(result: ReliabilityResult) -> list[Any]:
    """Look for tool_calls on the final answer-producing AgentOutput.

    The reliability layer flattens its trace into a single combined output;
    we walk the individual_outputs in trace order so the last technique
    output (whose text usually matches the final answer) is preferred.
    """
    trace = getattr(result, "trace", None) or {}
    outputs = trace.get("individual_outputs") or []
    if not outputs:
        return []
    for out in reversed(outputs):
        tc = out.get("tool_calls") if isinstance(out, dict) else None
        if tc:
            return list(tc)
    return []


def _finish_reason(
    result: ReliabilityResult,
    tool_calls_payload: list[Any],
) -> str:
    if tool_calls_payload:
        return "tool_calls"
    return getattr(result, "finish_reason", None) or "stop"
