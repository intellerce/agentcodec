"""
``ollama.Client``-shaped client with optional reliability layer.

Constructor mirrors ``ollama.Client(host=..., timeout=..., headers=...)``;
the added ``reliability=`` kwarg opts into the reliability layer. The
``chat`` / ``generate`` / ``embed`` / etc. methods proxy to the native
client when reliability is off; ``chat`` is intercepted when it's on.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from .._compat import LazyReliabilityResolver, resolve_reliability
from ..api import ReliabilityModule
from ._responses import chat_dict_from_result, stream_chat_dicts

# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class Client:
    """Drop-in for ``ollama.Client``."""

    def __init__(
        self,
        host: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "host": host,
            "headers": headers,
            "timeout": timeout,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=None,
            base_url=_ollama_base_url(host),
        )
        # See agentcodec.openai.client.OpenAI for the semantics — same flag.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None

    def _get_native(self) -> Any:
        if self._native is None:
            import ollama
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = ollama.Client(**kw)
        return self._native

    # ``chat`` is the reliability-aware entry point. Everything else
    # falls through __getattr__ to the native client.
    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]] | None = None,
        *,
        stream: bool = False,
        format: str | Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        think: bool | None = None,
        keep_alive: float | str | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | bool | None = None,
        expose_reliability_stream: bool | None = None,
        **kw: Any,
    ) -> Any:
        effective = _select_reliability(
            constructor=self._reliability, per_call=reliability,
            model=model,
            api_key=None,
            base_url=_ollama_base_url(self._native_kwargs.get("host")),
        )
        if effective is None:
            return self._get_native().chat(
                model=model, messages=list(messages or []),
                stream=stream, format=format, options=options,
                tools=tools, think=think, keep_alive=keep_alive, **kw,
            )
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._expose_reliability_stream
        )
        return _run_reliability(
            effective, model=model, messages=messages or [],
            stream=stream, format=format, options=options,
            tools=tools, kw=kw,
            expose_reliability_stream=expose,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_native(), name)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncClient:
    """Async counterpart to :class:`Client`."""

    def __init__(
        self,
        host: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | None = None,
        expose_reliability_stream: bool = False,
        **other_native_kwargs: Any,
    ) -> None:
        self._native_kwargs: dict[str, Any] = {
            "host": host,
            "headers": headers,
            "timeout": timeout,
            **other_native_kwargs,
        }
        self._reliability = LazyReliabilityResolver(
            reliability, api_key=None,
            base_url=_ollama_base_url(host),
        )
        # See agentcodec.openai.client.OpenAI for the semantics — same flag.
        self._expose_reliability_stream = expose_reliability_stream
        self._native: Any | None = None

    def _get_native(self) -> Any:
        if self._native is None:
            import ollama
            kw = {k: v for k, v in self._native_kwargs.items() if v is not None}
            self._native = ollama.AsyncClient(**kw)
        return self._native

    async def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]] | None = None,
        *,
        stream: bool = False,
        format: str | Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        think: bool | None = None,
        keep_alive: float | str | None = None,
        reliability: ReliabilityModule | str | Mapping[str, Any] | bool | None = None,
        expose_reliability_stream: bool | None = None,
        **kw: Any,
    ) -> Any:
        effective = _select_reliability(
            constructor=self._reliability, per_call=reliability,
            model=model,
            api_key=None,
            base_url=_ollama_base_url(self._native_kwargs.get("host")),
        )
        if effective is None:
            return await self._get_native().chat(
                model=model, messages=list(messages or []),
                stream=stream, format=format, options=options,
                tools=tools, think=think, keep_alive=keep_alive, **kw,
            )
        expose = (
            expose_reliability_stream
            if expose_reliability_stream is not None
            else self._expose_reliability_stream
        )
        loop = asyncio.get_running_loop()
        if stream:
            sync_iter = await loop.run_in_executor(
                None,
                lambda: _run_reliability(
                    effective, model=model, messages=messages or [],
                    stream=True, format=format, options=options,
                    tools=tools, kw=kw,
                    expose_reliability_stream=expose,
                ),
            )
            return _aiter_sync(sync_iter)
        return await loop.run_in_executor(
            None,
            lambda: _run_reliability(
                effective, model=model, messages=messages or [],
                stream=False, format=format, options=options,
                tools=tools, kw=kw,
                expose_reliability_stream=expose,
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_native(), name)


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
    stream: bool,
    format: str | Mapping[str, Any] | None,
    options: Mapping[str, Any] | None,
    tools: Sequence[Mapping[str, Any]] | None,
    kw: Mapping[str, Any],
    expose_reliability_stream: bool = False,
) -> Any:
    """Translate ``ollama.Client.chat`` kwargs to ReliabilityModule kwargs."""
    opts = dict(options or {})
    temperature = opts.pop("temperature", None)
    seed = opts.pop("seed", None)
    top_p = opts.pop("top_p", None)
    stop = opts.pop("stop", None)
    max_tokens = opts.pop("num_predict", None)
    response_format: Mapping[str, Any] | None
    if format == "json":
        response_format = {"type": "json_object"}
    elif isinstance(format, Mapping):
        response_format = format
    else:
        response_format = None
    run_kwargs: dict[str, Any] = {
        "tools": list(tools) if tools else None,
        "response_format": response_format,
        "stop": stop,
        "seed": seed,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Strip Nones so we don't override sensible run() defaults.
    run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
    if stream:
        events = mod.stream(messages=list(messages), **run_kwargs)
        return stream_chat_dicts(
            events, model=model,
            expose_reliability_stream=expose_reliability_stream,
        )
    # Same rationale as the OpenAI / Anthropic wrappers: always populate
    # the trace so callers can drill into per-branch quality and judge
    # checklists from the response dict without a second dispatch.
    result = mod.run(messages=list(messages), return_trace=True, **run_kwargs)
    return chat_dict_from_result(result, model=model)


def _ollama_base_url(host: str | None) -> str:
    """Translate an Ollama host into an OpenAI-compat base_url.

    The reliability presets construct channels that talk to providers via
    the OpenAI-compatible endpoint. Ollama exposes that at ``/v1`` under
    the same host.
    """
    if not host:
        return "http://localhost:11434/v1"
    h = host.rstrip("/")
    if h.endswith("/v1"):
        return h
    return f"{h}/v1"


async def _aiter_sync(it: Iterator[Any]) -> AsyncIterator[Any]:
    for chunk in it:
        yield chunk


__all__ = ["AsyncClient", "Client"]
