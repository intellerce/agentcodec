"""
One-line reliability presets.

The compat shims and the public :meth:`ReliabilityModule.from_preset`
constructor consume these to spare users a 40-line YAML when they just
want "give me HARQ-IR on this model, sensible defaults for everything
else." Every preset returns a dict suitable for
:meth:`LibraryConfig.from_dict`.

Presets are intentionally opinionated. A power user who wants to override
any of the chosen defaults can either (a) pass ``extras={...}`` to merge
into the result, (b) call :meth:`ReliabilityModule.from_dict` directly,
or (c) load a YAML.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Single-channel techniques are dispatched as a fixed strategy on one model.
# Multi-channel techniques need ≥ 2 models for spatial diversity; we duplicate
# the user's model with temperature variants when only one is supplied (still
# gives sampling diversity even when the model isn't varied).
_SINGLE_CHANNEL = {
    "baseline",
    "harq_ir", "harq_cc",
    "self_refine",
    "cov", "cisc",
    "fec", "turbo",
    "best_of_n", "weighted_best_of_n",
    "acm",
}
_MULTI_CHANNEL = {
    "diversity_mrc", "diversity_sc", "diversity_egc",
    "moa", "soft_decoding", "soft_diversity",
    "fountain",
}
_ROUTED = {
    "semknn", "acm_linear", "acm_table",
}


# Default params per technique. Empty when none are needed (most cases).
_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "harq_ir": {"max_rounds": 4},
    "harq_cc": {"max_rounds": 4},
    "self_refine": {"max_rounds": 3},
    "cov": {"num_verifications": 3},
    "best_of_n": {"n": 4},
    "weighted_best_of_n": {"n": 4},
    "fec": {"code_rate": 0.5},
    "moa": {"num_layers": 3},
    "fountain": {"num_samples": 6},
}


KNOWN_PRESETS = _SINGLE_CHANNEL | _MULTI_CHANNEL | _ROUTED


def build_preset_config(
    name: str,
    *,
    model: str | None = None,
    models: list[str | Mapping[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    judge: str | Mapping[str, Any] | None = None,
    critic: str | Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    temperature: float = 0.7,
    **technique_params: Any,
) -> dict[str, Any]:
    """Build a :class:`LibraryConfig` dict from a preset name + minimal kwargs.

    Args:
        name: Preset key (see :data:`KNOWN_PRESETS`).
        model: Single primary model. Either ``model=`` or ``models=`` must
            be set; ``model=`` wins when both are given.
        models: List of channel specs for multi-channel techniques. Each
            entry is either a model name (``str``) or a partial ModelConfig
            dict that overrides ``api_key`` / ``base_url`` / ``temperature``.
        api_key, base_url: Defaults applied to every channel and the judge.
        judge: Override for the judge channel. ``None`` → default to the
            primary model. ``str`` → judge model name. ``dict`` → full
            ``JudgeConfig`` override.
        critic: Same shape as ``judge`` for the iterative critic. ``None``
            → reuse the primary channel.
        extras: Top-level merge into the returned dict (e.g.
            ``{"defaults": {"on_error": "raise"}}``).
        temperature: Default temperature applied to channels lacking one.
        **technique_params: Override ``params`` on the chosen technique
            (e.g. ``max_rounds=6``).

    Returns:
        Dict consumable by :meth:`LibraryConfig.from_dict`.
    """
    if name not in KNOWN_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Choose from: {sorted(KNOWN_PRESETS)}"
        )
    if model is None and not models:
        # Last-ditch default — env-driven, mirrors examples/_common.py.
        env_model = os.environ.get("AGENTCODEC_EXAMPLE_MODEL_A")
        if env_model:
            model = env_model
        else:
            raise ValueError(
                f"preset {name!r} requires `model=` or `models=` (or set "
                f"AGENTCODEC_EXAMPLE_MODEL_A in env)"
            )

    # Resolve the channel pool.
    channels = _resolve_channels(
        name, model=model, models=models,
        api_key=api_key, base_url=base_url, temperature=temperature,
    )

    # Resolve the judge.
    judge_block = _resolve_judge(judge, primary_model=channels[0]["model"],
                                 api_key=api_key, base_url=base_url)

    # Resolve the critic when the technique calls for one. None → same channel.
    critic_block: dict[str, Any]
    if critic is None:
        critic_block = {"same": True}
    elif isinstance(critic, str):
        critic_block = {
            "model": critic, "base_url": base_url, "api_key": api_key,
        }
        critic_block = {k: v for k, v in critic_block.items() if v is not None}
    else:
        critic_block = dict(critic)

    # Resolve the strategy block.
    strategy = _resolve_strategy(name, technique_params=technique_params)

    cfg: dict[str, Any] = {
        "models": channels,
        "judge": judge_block,
        "critic": critic_block,
        "strategy": strategy,
        "defaults": {
            "category": "auto",
            "on_error": "fallback_baseline",
        },
    }
    if extras:
        _deep_merge(cfg, dict(extras))
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_channels(
    name: str,
    *,
    model: str | None,
    models: list[str | Mapping[str, Any]] | None,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
) -> list[dict[str, Any]]:
    if models:
        out: list[dict[str, Any]] = []
        for entry in models:
            if isinstance(entry, str):
                block: dict[str, Any] = {"model": entry, "temperature": temperature}
            else:
                block = dict(entry)
                block.setdefault("temperature", temperature)
            if api_key is not None:
                block.setdefault("api_key", api_key)
            if base_url is not None:
                block.setdefault("base_url", base_url)
            out.append(block)
        return out
    # Single-model path.
    assert model is not None
    primary = _make_channel(model, api_key, base_url, temperature)
    if name in _MULTI_CHANNEL and (models is None):
        # Duplicate the model with a perturbed temperature so techniques that
        # require ≥ 2 channels still work out of the box. Real spatial
        # diversity wants two different model families — users can pass
        # `models=[...]` for that.
        secondary = _make_channel(model, api_key, base_url, max(0.0, min(1.5, temperature + 0.2)))
        return [primary, secondary]
    return [primary]


def _make_channel(
    model: str, api_key: str | None, base_url: str | None, temperature: float,
) -> dict[str, Any]:
    block: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key is not None:
        block["api_key"] = api_key
    if base_url is not None:
        block["base_url"] = base_url
    return block


def _resolve_judge(
    judge: str | Mapping[str, Any] | None,
    *,
    primary_model: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    if judge is None:
        # Default to the env-configured judge, then the primary model.
        judge_model = (
            os.environ.get("AGENTCODEC_EXAMPLE_JUDGE")
            or os.environ.get("AGENTCODEC_JUDGE_MODEL")
            or primary_model
        )
        block: dict[str, Any] = {"model": judge_model}
        if api_key is not None:
            block["api_key"] = api_key
        if base_url is not None:
            block["base_url"] = base_url
        return block
    if isinstance(judge, str):
        block = {"model": judge}
        if api_key is not None:
            block["api_key"] = api_key
        if base_url is not None:
            block["base_url"] = base_url
        return block
    return dict(judge)


def _resolve_strategy(
    name: str, *, technique_params: Mapping[str, Any],
) -> dict[str, Any]:
    if name in _ROUTED:
        # Router strategies need a more elaborate config; we ship the
        # smallest sensible defaults.
        if name == "semknn":
            router: dict[str, Any] = {"type": "semknn", "lambda": 1.0}
        elif name == "acm_linear":
            cache = technique_params.pop("cache", None)
            if not cache:
                raise ValueError(
                    "preset 'acm_linear' requires cache=<weights.json> "
                    "(no built-in default)"
                )
            router = {"type": "acm_linear", "cache": cache}
        else:
            table = technique_params.pop("table", None)
            if not table:
                raise ValueError(
                    "preset 'acm_table' requires table=[{...}] "
                    "(no built-in default)"
                )
            router = {"type": "acm_table", "table": table}
        return {"type": "routed", "router": router}

    # Fixed strategies.
    params = dict(_DEFAULT_PARAMS.get(name, {}))
    params.update(technique_params)
    block: dict[str, Any] = {"type": "fixed", "technique": name}
    if params:
        block["params"] = params
    return block


def _deep_merge(dest: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dest.get(k), dict):
            _deep_merge(dest[k], dict(v))
        else:
            dest[k] = v


__all__ = ["KNOWN_PRESETS", "build_preset_config"]
