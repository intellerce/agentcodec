"""
``openai.OpenAI``-shaped client with optional reliability layer.

Design rules:
  * Without ``reliability=`` the wrapper is a thin proxy. The native
    ``openai.OpenAI`` instance is constructed lazily — there is zero
    overhead when reliability is off.
  * When ``reliability=`` is set, calls into ``client.chat.completions.create``
    go through a :class:`~agentcodec.ReliabilityModule`; everything else
    (``embeddings``, ``files``, ``models``, ...) still proxies to the
    native SDK.
  * Per-call ``reliability=`` on ``create()`` overrides the constructor
    setting. Pass ``reliability=False`` to bypass for one call.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from .._compat import LazyReliabilityResolver, resolve_reliability
from ..api import ReliabilityModule
from ._responses import completion_from_result, stream_from_event_iter

# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class OpenAI:
    """Drop-in for ``openai.OpenAI``.

    Constructor signature is a superset of the native client's: all standard
    kwargs (``api_key``, ``base_url``, ``organization``, ``timeout``,
    ``max_retries``, …) pass straight through to ``openai.OpenAI`` on
    construction. The added ``reliability=`` kwarg opts into the reliability
    layer when set.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, str] | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "organization": organization,
            "project": project,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "default_query": default_query,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=api_key, base_url=base_url,
        )
        # When True, intermediate reliability-layer roles (``draft``,
        # ``critique``, ``candidate``, ``verification``) get surfaced as
        # native stream chunks with ``delta.agentcodec_role`` sentinel
        # attributes. Default False keeps the shim looking exactly like
        # the native OpenAI SDK — answer + thinking only.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None  # lazy
        self.chat = _ChatNamespace(self)

    # Lazy proxy to the real SDK. Constructed once on first use.
    def _get_native(self) -> Any:
        if self._native is None:
            from openai import OpenAI as _NativeOpenAI  # lazy import
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = _NativeOpenAI(**kw)
        return self._native

    # Any attribute access we don't explicitly model (``embeddings``, ``files``,
    # ``models``, ...) forwards to the native SDK. This keeps the wrapper
    # compatible with the full openai surface without us having to enumerate
    # every namespace.
    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal attribute lookup misses, so ``chat`` and the
        # explicit attributes stay fast.
        return getattr(self._get_native(), name)


class _ChatNamespace:
    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self.completions = _ChatCompletionsNamespace(client)


class _ChatCompletionsNamespace:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]],
        stream: bool = False,
        reliability: ReliabilityModule | str | Mapping[str, Any] | bool | None = None,
        expose_reliability_stream: bool | None = None,
        **kw: Any,
    ) -> Any:
        # Per-call ``reliability=`` overrides the client default. Explicit
        # ``False`` forces passthrough for this single call.
        effective = _select_reliability(
            constructor=self._client._reliability, per_call=reliability,
            model=model,
            api_key=self._client._native_kwargs.get("api_key"),
            base_url=self._client._native_kwargs.get("base_url"),
        )
        if effective is None:
            return self._client._get_native().chat.completions.create(
                model=model, messages=messages, stream=stream, **kw,
            )
        # Per-call ``expose_reliability_stream`` overrides the client default.
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._client._expose_reliability_stream
        )
        return _run_reliability(
            effective, model=model, messages=messages, stream=stream, kw=kw,
            expose_reliability_stream=expose,
        )


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncOpenAI:
    """Async counterpart to :class:`OpenAI`.

    Mirrors ``openai.AsyncOpenAI``. The reliability path runs synchronous
    techniques in a worker thread via ``loop.run_in_executor`` — same
    pattern :meth:`ReliabilityModule.arun` uses.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, str] | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "organization": organization,
            "project": project,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "default_query": default_query,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=api_key, base_url=base_url,
        )
        # See sync ``OpenAI.__init__`` for semantics.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None
        self.chat = _AsyncChatNamespace(self)

    def _get_native(self) -> Any:
        if self._native is None:
            from openai import AsyncOpenAI as _NativeAsync
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = _NativeAsync(**kw)
        return self._native

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_native(), name)


class _AsyncChatNamespace:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client
        self.completions = _AsyncChatCompletionsNamespace(client)


class _AsyncChatCompletionsNamespace:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]],
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
            return await self._client._get_native().chat.completions.create(
                model=model, messages=messages, stream=stream, **kw,
            )
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._client._expose_reliability_stream
        )
        # Reliability path: run the sync technique pipeline in an executor.
        loop = asyncio.get_running_loop()
        if stream:
            # Build the sync iterator off the loop, then re-yield each chunk
            # into the async loop via an aiter wrapper.
            sync_iter = await loop.run_in_executor(
                None,
                lambda: _run_reliability(
                    effective, model=model, messages=messages, stream=True, kw=kw,
                    expose_reliability_stream=expose,
                ),
            )
            return _aiter_sync(sync_iter)
        return await loop.run_in_executor(
            None,
            lambda: _run_reliability(
                effective, model=model, messages=messages, stream=False, kw=kw,
                expose_reliability_stream=expose,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers shared by sync + async paths
# ---------------------------------------------------------------------------


def _select_reliability(
    *,
    constructor: LazyReliabilityResolver,
    per_call: ReliabilityModule | str | Mapping[str, Any] | bool | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> ReliabilityModule | None:
    """Resolve which (if any) ReliabilityModule applies to this call.

    ``per_call=False`` forces passthrough. Otherwise the per-call spec
    wins over the client default. Both branches honor the ``model=`` from
    the current ``create()`` so preset strings can bind to the right
    model lazily.
    """
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
    messages: list[Mapping[str, Any]],
    stream: bool,
    kw: Mapping[str, Any],
    expose_reliability_stream: bool = False,
) -> Any:
    """Translate OpenAI kwargs to ``ReliabilityModule.run``/``stream`` and adapt the result."""
    run_kwargs = _translate_openai_kwargs(kw)
    if stream:
        events = mod.stream(messages=list(messages), **run_kwargs)
        return stream_from_event_iter(
            events, model=model,
            expose_reliability_stream=expose_reliability_stream,
        )
    # Always populate the trace on wrapper calls. The trace is a small dict
    # (a few KB even for multi-channel techniques), and surfacing it on
    # `resp.reliability.trace` means callers can inspect per-branch quality
    # / judge checklists without a second dispatch on the same prompt.
    result = mod.run(messages=list(messages), return_trace=True, **run_kwargs)
    return completion_from_result(result, model=model)


def _translate_openai_kwargs(kw: Mapping[str, Any]) -> dict[str, Any]:
    """Pluck the kwargs ReliabilityModule.run understands; drop the rest.

    Unknown kwargs are silently dropped on the reliability path — they
    are intentionally not threaded into the channel-level call because
    that would defeat the technique's parameter discipline. Power users
    who need raw passthrough should set ``reliability=False`` per call.
    """
    out: dict[str, Any] = {}
    for src, dst in _OPENAI_TO_RELIABILITY.items():
        if src in kw and kw[src] is not None:
            out[dst] = kw[src]
    return out


_OPENAI_TO_RELIABILITY: dict[str, str] = {
    "tools": "tools",
    "tool_choice": "tool_choice",
    "response_format": "response_format",
    "stop": "stop",
    "seed": "seed",
    "top_p": "top_p",
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "max_completion_tokens": "max_tokens",
    "user": "task_id",
    "metadata": "metadata",
}


async def _aiter_sync(it: Iterator[Any]) -> AsyncIterator[Any]:
    for chunk in it:
        yield chunk


__all__ = ["AsyncOpenAI", "OpenAI"]
