"""
Shared helpers for the OpenAI / Anthropic / Ollama compatibility shims.

This is an internal module — the public surface is the per-provider
``agentcodec.{openai,anthropic,ollama}`` packages, not this file. Code
here is deliberately small: a resolver that turns the ``reliability=``
constructor kwarg into a :class:`ReliabilityModule` (possibly lazily,
once a model name is known per-call).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .api import ReliabilityModule


def resolve_reliability(
    spec: ReliabilityModule | str | Mapping[str, Any] | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ReliabilityModule | None:
    """Normalize the ``reliability=`` value into a concrete module.

    Accepted shapes:

    * ``None``  — no reliability layer (pure passthrough).
    * ``ReliabilityModule``  — used as-is.
    * ``str``  — preset name. Requires ``model`` to be supplied (the shims
      pass the per-call ``model=`` here, so the model is the one the user
      actually wants to call).
    * ``Mapping``  — full :meth:`LibraryConfig.from_dict` payload.
    """
    if spec is None:
        return None
    if isinstance(spec, ReliabilityModule):
        return spec
    if isinstance(spec, str):
        return ReliabilityModule.from_preset(
            spec, model=model, api_key=api_key, base_url=base_url,
        )
    if isinstance(spec, Mapping):
        return ReliabilityModule.from_dict(dict(spec))
    raise TypeError(
        f"reliability= must be None, a preset string, a Mapping (config "
        f"dict), or a ReliabilityModule — got {type(spec).__name__}"
    )


class LazyReliabilityResolver:
    """Caches per-model ``ReliabilityModule`` instances behind a single spec.

    Compat shims hold one of these on the client; ``.for_model(model)``
    returns the module for that model, constructing on first use and
    sharing it across subsequent calls with the same model. A
    ``ReliabilityModule`` or ``None`` spec short-circuits caching since
    the module is already concrete.
    """
    def __init__(
        self,
        spec: ReliabilityModule | str | Mapping[str, Any] | None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._spec = spec
        self._api_key = api_key
        self._base_url = base_url
        self._cache: dict[str, ReliabilityModule] = {}
        self._concrete: ReliabilityModule | None = (
            spec if isinstance(spec, ReliabilityModule) else None
        )

    @property
    def is_passthrough(self) -> bool:
        return self._spec is None

    def for_model(self, model: str | None) -> ReliabilityModule | None:
        if self._spec is None:
            return None
        if self._concrete is not None:
            return self._concrete
        # Need a per-call model to resolve preset strings; dict specs
        # carry their own model pool, so we cache by spec id.
        key = model or "_no_model_"
        if key not in self._cache:
            self._cache[key] = resolve_reliability(
                self._spec, model=model,
                api_key=self._api_key, base_url=self._base_url,
            )
        return self._cache[key]


__all__ = ["LazyReliabilityResolver", "resolve_reliability"]
