"""
``anthropic.Anthropic``-shaped client with optional reliability layer.

Passthrough mode (``reliability=None``) constructs the native
``anthropic.Anthropic`` lazily and proxies every call. Reliability mode
funnels ``client.messages.create(...)`` through a
:class:`~agentcodec.ReliabilityModule`, returning an Anthropic-shaped
``Message`` (with ``content`` blocks, ``stop_reason``, ``usage.input_tokens``,
…). Embeddings and other namespaces always pass through.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from .._compat import LazyReliabilityResolver, resolve_reliability
from ..api import ReliabilityModule
from ..messages import ChatRequest
from ._responses import message_from_result, stream_from_event_iter

# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class Anthropic:
    """Drop-in for ``anthropic.Anthropic``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=api_key, base_url=base_url,
        )
        # See agentcodec.openai.client.OpenAI for the semantics — same flag.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None
        self.messages = _MessagesNamespace(self)

    def _get_native(self) -> Any:
        if self._native is None:
            from anthropic import Anthropic as _Native
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = _Native(**kw)
        return self._native

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_native(), name)


class _MessagesNamespace:
    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int = 1024,
        stream: bool = False,
        reliability: ReliabilityModule | str | Mapping[str, Any] | bool | None = None,
        expose_reliability_stream: bool | None = None,
        **kw: Any,
    ) -> Any:
        effective = _select_reliability(
            constructor=self._client._reliability, per_call=reliability,
            model=model,
            api_key=self._client._native_kwargs.get("api_key"),
            base_url=self._client._native_kwargs.get("base_url"),
        )
        if effective is None:
            return self._client._get_native().messages.create(
                model=model, messages=list(messages),
                system=system, max_tokens=max_tokens, stream=stream, **kw,
            )
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._client._expose_reliability_stream
        )
        return _run_reliability(
            effective, model=model, messages=messages, system=system,
            max_tokens=max_tokens, stream=stream, kw=kw,
            expose_reliability_stream=expose,
        )


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncAnthropic:
    """Async counterpart to :class:`Anthropic`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=api_key, base_url=base_url,
        )
        # See agentcodec.openai.client.OpenAI for the semantics — same flag.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None
        self.messages = _AsyncMessagesNamespace(self)

    def _get_native(self) -> Any:
        if self._native is None:
            from anthropic import AsyncAnthropic as _Native
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = _Native(**kw)
        return self._native

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_native(), name)


class _AsyncMessagesNamespace:
    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int = 1024,
        stream: bool = False,
        reliability: ReliabilityModule | str | Mapping[str, Any] | bool | None = None,
        expose_reliability_stream: bool | None = None,
        **kw: Any,
    ) -> Any:
        effective = _select_reliability(
            constructor=self._client._reliability, per_call=reliability,
            model=model,
            api_key=self._client._native_kwargs.get("api_key"),
            base_url=self._client._native_kwargs.get("base_url"),
        )
        if effective is None:
            return await self._client._get_native().messages.create(
                model=model, messages=list(messages),
                system=system, max_tokens=max_tokens, stream=stream, **kw,
            )
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._client._expose_reliability_stream
        )
        loop = asyncio.get_running_loop()
        if stream:
            sync_iter = await loop.run_in_executor(
                None,
                lambda: _run_reliability(
                    effective, model=model, messages=messages, system=system,
                    max_tokens=max_tokens, stream=True, kw=kw,
                    expose_reliability_stream=expose,
                ),
            )
            return _aiter_sync(sync_iter)
        return await loop.run_in_executor(
            None,
            lambda: _run_reliability(
                effective, model=model, messages=messages, system=system,
                max_tokens=max_tokens, stream=False, kw=kw,
                expose_reliability_stream=expose,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_reliability(
    *,
    constructor: LazyReliabilityResolver,
    per_call: ReliabilityModule | str | Mapping[str, Any] | bool | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> ReliabilityModule | None:
    if per_call is False:
        return None
    if per_call is None:
        return constructor.for_model(model)
    if isinstance(per_call, ReliabilityModule):
        return per_call
    return resolve_reliability(
        per_call, model=model, api_key=api_key, base_url=base_url,
    )


def _run_reliability(
    mod: ReliabilityModule,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | Sequence[Mapping[str, Any]] | None,
    max_tokens: int,
    stream: bool,
    kw: Mapping[str, Any],
    expose_reliability_stream: bool = False,
) -> Any:
    """Translate Anthropic ``messages.create`` kwargs to ReliabilityModule kwargs."""
    # ChatRequest.from_anthropic hoists `system` into a leading system
    # message so the rest of the pipeline sees a uniform shape.
    pre_built = ChatRequest.from_anthropic(
        messages=list(messages),
        system=system,
        tools=kw.get("tools"),
        tool_choice=kw.get("tool_choice"),
        stop_sequences=kw.get("stop_sequences"),
        temperature=kw.get("temperature"),
        max_tokens=max_tokens,
        top_p=kw.get("top_p"),
    )
    # Convert the Anthropic-shaped messages into OpenAI's shape for the
    # public mod.run(messages=...) entry point, which speaks OpenAI's
    # message dict shape. ``with_system`` ensures the system text Anthropic
    # carried as a kwarg becomes a real system message.
    openai_messages = pre_built.to_openai_messages()
    run_kwargs = _translate_anthropic_kwargs(kw, max_tokens=max_tokens)
    if stream:
        events = mod.stream(messages=openai_messages, **run_kwargs)
        return stream_from_event_iter(
            events, model=model,
            expose_reliability_stream=expose_reliability_stream,
        )
    # Same rationale as the OpenAI wrapper: always populate the trace so
    # callers can read per-branch quality / judge checklists from
    # `resp.reliability.trace` without a second dispatch.
    result = mod.run(messages=openai_messages, return_trace=True, **run_kwargs)
    return message_from_result(result, model=model)


def _translate_anthropic_kwargs(
    kw: Mapping[str, Any],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Pluck recognized kwargs; map Anthropic field names to our run() names."""
    out: dict[str, Any] = {"max_tokens": max_tokens}
    for src, dst in _ANTHROPIC_TO_RELIABILITY.items():
        if src in kw and kw[src] is not None:
            out[dst] = kw[src]
    return out


_ANTHROPIC_TO_RELIABILITY: dict[str, str] = {
    "tools": "tools",
    "tool_choice": "tool_choice",
    "stop_sequences": "stop",
    "temperature": "temperature",
    "top_p": "top_p",
    "metadata": "metadata",
}


async def _aiter_sync(it: Iterator[Any]) -> AsyncIterator[Any]:
    for chunk in it:
        yield chunk


__all__ = ["Anthropic", "AsyncAnthropic"]
