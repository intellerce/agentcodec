"""
Build Ollama-shaped response dicts from a :class:`ReliabilityResult`, and
translate our :class:`Event` stream into the chunked dict shape the
``ollama`` library yields when ``stream=True``.

Ollama's responses are plain Python dicts — there's no SDK type to mimic.
We include a ``"reliability"`` key carrying a compact summary of the
:class:`ReliabilityResult` so power users can introspect.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ..results import Event, FinalEvent, ProgressEvent, ReliabilityResult, TokenEvent


def chat_dict_from_result(
    result: ReliabilityResult,
    *,
    model: str,
) -> dict[str, Any]:
    """Build the dict shape returned by ``ollama.Client.chat(stream=False)``."""
    tool_calls = _collect_tool_calls(result)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.text or "",
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "function": {
                    "name": tc.name,
                    "arguments": _maybe_json(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
    done_reason = "tool_calls" if tool_calls else (
        getattr(result, "finish_reason", None) or "stop"
    )
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "total_duration": int((result.latency_s or 0) * 1e9),  # nanoseconds
        "prompt_eval_count": result.input_tokens,
        "eval_count": result.output_tokens,
        "reliability": _summarize_reliability(result),
    }


# See agentcodec.openai._responses for the rationale on role gating.
_OLLAMA_ANSWER_ROLES: frozenset[str] = frozenset({"answer", "synthesis"})


def stream_chat_dicts(
    events: Iterator[Event],
    *,
    model: str,
    expose_reliability_stream: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield dicts mimicking ``ollama.Client.chat(stream=True)``.

    Default (``expose_reliability_stream=False``):
      * ``role="answer"`` or ``"synthesis"`` → ``message.content``
      * ``role="thinking"`` → ``message.thinking`` (Ollama's native field)
      * Other reliability-layer roles (``"draft"``, ``"critique"``,
        ``"candidate"``, ``"verification"``) are **dropped** — they're
        internal to the technique.

    Opt-in (``expose_reliability_stream=True``):
      * Same as above, plus internal roles surface in ``message.content``
        with extra ``message.agentcodec_role`` and
        ``message.agentcodec_call_id`` fields so power users can demux.
    """
    created_at = _now_iso()
    final_result: ReliabilityResult | None = None
    for ev in events:
        if isinstance(ev, TokenEvent):
            chunk = _build_ollama_chunk(
                ev, model, created_at,
                expose_reliability_stream=expose_reliability_stream,
            )
            if chunk is None:
                continue
            yield chunk
        elif isinstance(ev, FinalEvent):
            final_result = ev.result
        elif isinstance(ev, ProgressEvent):
            continue
    # Terminal chunk.
    final_dict: dict[str, Any] = {
        "model": model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
    }
    if final_result is not None:
        tc = _collect_tool_calls(final_result)
        final_dict["done_reason"] = "tool_calls" if tc else (
            getattr(final_result, "finish_reason", None) or "stop"
        )
        final_dict["prompt_eval_count"] = final_result.input_tokens
        final_dict["eval_count"] = final_result.output_tokens
        final_dict["total_duration"] = int((final_result.latency_s or 0) * 1e9)
        final_dict["reliability"] = _summarize_reliability(final_result)
    yield final_dict


def _build_ollama_chunk(
    ev: TokenEvent,
    model: str,
    created_at: str,
    *,
    expose_reliability_stream: bool,
) -> dict[str, Any] | None:
    """Map one TokenEvent to an Ollama-shaped dict (or None to drop it)."""
    role = ev.role
    if role == "thinking":
        message: dict[str, Any] = {"role": "assistant", "thinking": ev.text}
        if expose_reliability_stream:
            message["agentcodec_role"] = "thinking"
            message["agentcodec_call_id"] = ev.call_id
        return {
            "model": model, "created_at": created_at,
            "message": message, "done": False,
        }
    if role in _OLLAMA_ANSWER_ROLES:
        message = {"role": "assistant", "content": ev.text}
        if expose_reliability_stream:
            message["agentcodec_role"] = role
            message["agentcodec_call_id"] = ev.call_id
        return {
            "model": model, "created_at": created_at,
            "message": message, "done": False,
        }
    if expose_reliability_stream:
        message = {
            "role": "assistant", "content": ev.text,
            "agentcodec_role": role,
            "agentcodec_call_id": ev.call_id,
        }
        return {
            "model": model, "created_at": created_at,
            "message": message, "done": False,
        }
    return None


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


def _maybe_json(arguments: str) -> Any:
    """Ollama expects the function arguments as a dict, not a JSON string."""
    import json
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def _summarize_reliability(result: ReliabilityResult) -> dict[str, Any]:
    return {
        "technique_used": result.technique_used,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "final_quality": result.final_quality,
        "best_individual_quality": result.best_individual_quality,
        "diversity_gain": result.diversity_gain,
        # Populated when the wrapper requests `return_trace=True` (which it
        # now does by default). Empty dict otherwise. Lets callers inspect
        # per-branch quality and judge checklists without a second dispatch.
        "trace": result.trace,
    }
