"""
Agent-as-Channel abstraction.

Models an LLM agent as a stochastic channel: Y = A(X) + N
where X is the input prompt, Y is the output, and N is noise
(hallucination, omission, reasoning errors).

Module size note
----------------
This module is currently ~1900 LOC because it absorbs three concerns
that share the AgentChannel class state:

  * transports — the per-backend ``_transmit_*`` paths (OpenAI-compat,
    Anthropic native SDK, Ollama native ``/api/chat`` fallback).
  * thinking translation — the per-backend ``thinking`` → ``extra_body``
    matrix and the inverse parsing of reasoning tokens out of responses.
  * pricing tables — ``MODEL_COSTS`` plus ``_infer_cost_from_name`` plus
    ``_resolve_costs`` (which delegates to ``agentcodec.pricing`` for the
    live OpenRouter catalog).

A clean split lives on the v0.4 roadmap (see CHANGELOG):

    channel/transports/{openai,anthropic,ollama}.py
    channel/thinking.py
    channel/pricing.py        # hardcoded MODEL_COSTS table
    channel/scoring.py        # QualityScorer

It is **deliberately deferred** until after the first public release —
the transport branches share parsing state and a premature split risks
silently changing response-handling behavior. Contributions toward that
split are welcome; please ship them with the existing 75-test suite
green and add per-transport unit tests as you go.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Mapping
from typing import Any

# Provider SDKs are lazy-imported to keep the core install lightweight.
# - openai:    needed for any OpenAI-compatible endpoint (Ollama, vLLM,
#              OpenRouter, etc., as well as openai.com). Lazy-imported in
#              AgentChannel.__init__ via _import_openai().
# - anthropic: needed only when the channel model name starts with
#              "claude-". Lazy-imported via _import_anthropic().
# Each helper raises a clear, actionable ImportError so users know which
# pip extra to install.
from .messages import ChannelChunk, ChannelDone, ChatRequest, ToolCall
from .models import AgentOutput


def _import_openai():
    """Return the openai.OpenAI class, with a clear hint on missing extras."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "agentcodec needs the `openai` package to talk to any "
            "OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, GPT-*, "
            "etc.). Install it with:\n"
            "    pip install 'agentcodec[openai]'\n"
            "or `pip install openai`. Original error: " + repr(e)
        ) from e
    return OpenAI


def _import_openai_async():
    """Return the openai.AsyncOpenAI class."""
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "agentcodec needs the `openai` package for async transmit. "
            "Install with `pip install 'agentcodec[openai]'`. "
            "Original error: " + repr(e)
        ) from e
    return AsyncOpenAI


def _import_anthropic():
    """Return the `anthropic` module, with a clear hint on missing extras."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "agentcodec needs the `anthropic` package to talk to Claude "
            "models (model names starting with `claude-`). Install it with:\n"
            "    pip install 'agentcodec[anthropic]'\n"
            "or `pip install anthropic`. Original error: " + repr(e)
        ) from e
    return anthropic


def _import_anthropic_async():
    """Return the `anthropic.AsyncAnthropic` class."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError(
            "agentcodec needs the `anthropic` package for async transmit. "
            "Install with `pip install 'agentcodec[anthropic]'`. "
            "Original error: " + repr(e)
        ) from e
    return AsyncAnthropic

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input, output) — cloud-equivalent estimates.
# For local models (Ollama, vLLM), we use the cheapest hosted API price
# for an equivalent model so that cost comparisons remain meaningful
# across local and cloud setups.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # --- Cloud APIs ---
    # OpenAI GPT-5 family (verify current pricing at https://openai.com/api/pricing/)
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    # OpenAI reasoning (o-series)
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # OpenAI GPT-4 family
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Claude model aliases (latest)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # Claude dated versions
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-5-20250514": (3.00, 15.00),
    "claude-sonnet-4-6-20250725": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-opus-4-5-20250610": (15.00, 75.00),
    "claude-opus-4-6-20250725": (15.00, 75.00),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    # DeepSeek-R1 local (Ollama) — priced at cloud-equivalent hosted rates
    "deepseek-r1:1.5b": (0.04, 0.04),
    "deepseek-r1:7b": (0.10, 0.10),
    "deepseek-r1:8b": (0.10, 0.10),
    "deepseek-r1:14b": (0.20, 0.20),
    "deepseek-r1:32b": (0.55, 2.19),
    "deepseek-r1:70b": (0.90, 0.90),
    # DeepSeek-R1 distilled variants (AWQ-quantized vLLM builds)
    "stelterlab/DeepSeek-R1-Distill-Qwen-14B-AWQ": (0.20, 0.20),
    # --- Open-weight models (cloud-equivalent pricing) ---
    # Llama 3 / 3.1 family — priced at typical hosted rates (e.g. Together, Fireworks)
    "llama3:8b": (0.10, 0.10),
    "llama3.1:8b": (0.10, 0.10),
    "llama3.2:1b": (0.02, 0.02),
    "llama3.2:3b": (0.04, 0.04),
    "llama3.1:70b": (0.90, 0.90),
    "llama3.1:405b": (3.00, 3.00),
    # HuggingFace / vLLM full names
    "meta-llama/Llama-3.1-8B-Instruct": (0.10, 0.10),
    "meta-llama/Llama-3.1-70B-Instruct": (0.90, 0.90),
    # Qwen family
    "qwen2:7b": (0.10, 0.10),
    "qwen2.5:3b": (0.04, 0.04),
    "qwen2.5:7b": (0.10, 0.10),
    "qwen2.5:14b": (0.20, 0.20),
    "qwen3:0.6b": (0.04, 0.04),
    "qwen3:1.7b": (0.04, 0.04),
    "qwen3:4b": (0.06, 0.06),
    "qwen3:8b": (0.10, 0.10),
    "qwen3:14b": (0.20, 0.20),
    "qwen3:32b": (0.30, 0.30),
    "qwen3:30b-a3b": (0.10, 0.10),   # MoE, 3B active
    "qwen3:235b-a22b": (0.90, 0.90), # MoE, 22B active
    # Qwen3.6 (dense) — 35B sits between qwen3:32b and the ≤70B tier;
    # keep it at the 32B family price for consistency.
    "qwen3.6:35b": (0.30, 0.30),
    "qwen2.5:72b": (0.90, 0.90),
    "Qwen/Qwen2.5-7B-Instruct": (0.10, 0.10),
    # Mistral family
    "mistral:7b": (0.10, 0.10),
    "mixtral:8x7b": (0.50, 0.50),
    # Gemma family
    "gemma2:9b": (0.10, 0.10),
    "gemma2:27b": (0.30, 0.30),
    "gemma3:1b": (0.04, 0.04),
    "gemma3:4b": (0.06, 0.06),
    "gemma3:12b": (0.15, 0.15),
    "gemma3:12b-cloud": (0.15, 0.15),
    "gemma3:27b": (0.30, 0.30),
    # Gemma 4 (Ollama Cloud preview) — priced at the 27–32B dense tier.
    "gemma4:31b-cloud": (0.30, 0.30),
    # Devstral 2 (Mistral code-specialist, Ollama Cloud) — 24B dense.
    "devstral-small-2:24b-cloud": (0.30, 0.30),
    # Nemotron-3 nano (NVIDIA, Ollama Cloud) — 30B dense.
    "nemotron-3-nano:30b-cloud": (0.30, 0.30),
    # GLM family (Z.ai / ZhipuAI, Ollama Cloud).
    # GLM-5.1 is the successor to GLM-4.5/4.6 — large-class MoE (~100B+),
    # priced at the heaviest tier matching qwen3:235b-a22b.
    "glm-5.1:cloud": (0.90, 0.90),
    # Phi family
    "phi3:3.8b": (0.10, 0.10),
    "phi3:14b": (0.20, 0.20),
    # Phi-4 is 14B — AWQ-quantized vLLM build
    "stelterlab/phi-4-AWQ": (0.20, 0.20),
    # CodeLlama
    "codellama:7b": (0.10, 0.10),
    "codellama:34b": (0.60, 0.60),
    # Fallback
    "default": (2.00, 8.00),
}

# Rough cost tier by parameter count (input, output per 1M tokens).
# Used when a model name isn't in MODEL_COSTS but we can infer its size.
_SIZE_TIERS: list[tuple[int, tuple[float, float]]] = [
    (3,   (0.04, 0.04)),   # ≤3B
    (8,   (0.10, 0.10)),   # ≤8B
    (14,  (0.20, 0.20)),   # ≤14B
    (32,  (0.30, 0.30)),   # ≤32B
    (70,  (0.90, 0.90)),   # ≤70B
    (140, (2.00, 2.00)),   # ≤140B
]


def _infer_cost_from_name(model: str) -> tuple[float, float] | None:
    """Try to guess pricing from parameter-count hints in the model name."""
    import re
    # Match patterns like :8b, :70b, -7B, _3B, -8x7b (MoE → use total)
    m = re.search(r"[:\-_](\d+)x(\d+)[bB]", model)
    if m:
        param_b = int(m.group(1)) * int(m.group(2))
    else:
        m = re.search(r"(\d+)[bB]", model)
        if not m:
            return None
        param_b = int(m.group(1))

    for threshold, cost in _SIZE_TIERS:
        if param_b <= threshold:
            return cost
    return _SIZE_TIERS[-1][1]


# Per-model price cache populated by _resolve_costs: model name → (input, output)
# per-1M-token USD. Avoids re-running the OpenRouter fuzzy-match on every call.
_RESOLVED_PRICE_CACHE: dict[str, tuple[float, float]] = {}


def _resolve_costs(model: str) -> tuple[float, float]:
    """
    Resolve (input_per_1M, output_per_1M) for `model`.

    Resolution order:
      1. AGENTCODEC_DISABLE_OPENROUTER=1 → skip the network entirely.
      2. OpenRouter catalog (disk cache, 7-day TTL).
      3. Hardcoded MODEL_COSTS exact match.
      4. Parameter-count heuristic (_infer_cost_from_name).
      5. MODEL_COSTS["default"].
    """
    if model in _RESOLVED_PRICE_CACHE:
        return _RESOLVED_PRICE_CACHE[model]

    costs: tuple[float, float] | None = None
    if os.environ.get("AGENTCODEC_DISABLE_OPENROUTER") not in ("1", "true", "True"):
        try:
            from . import pricing
            result = pricing.lookup(model)
            if result is not None:
                costs = (result[0], result[1])
        except Exception as e:  # pragma: no cover -- defensive
            logger.debug(f"OpenRouter lookup failed for {model!r}: {e!r}")

    if costs is None:
        costs = MODEL_COSTS.get(model)
    if costs is None:
        costs = _infer_cost_from_name(model) or MODEL_COSTS["default"]

    _RESOLVED_PRICE_CACHE[model] = costs
    return costs


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    user_override: tuple[float, float] | None = None,
) -> float:
    """Legacy cost estimator. Returns just the dollar amount.

    For the full breakdown (rate source, caveats), use
    ``agentcodec.cost.compute_cost`` instead — this wrapper delegates to it
    for behavioral consistency but discards the source tier.
    """
    if user_override is not None:
        rate = user_override
    else:
        rate = _resolve_costs(model)
    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000


def _is_anthropic_model(model: str) -> bool:
    """Check if a model name is an Anthropic model."""
    return model.startswith("claude-") or model.startswith("claude3")


def _is_openai_reasoning_model(model: str) -> bool:
    """
    Detect GPT-5 / o-series models that use the reasoning API.

    These models differ from classic chat models in two ways:
    - They use `max_completion_tokens` instead of `max_tokens`
    - They only accept temperature=1.0 (the API rejects other values)
    """
    m = model.lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _is_strict_reasoning_model(model: str) -> bool:
    """GPT-5 and o-series only accept temperature=1.0."""
    m = model.lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


# Single source of truth for thinking-tag detection AND stripping.
# Detection (`_THINKING_TAG_RE`) matches just the opening tag; stripping
# (in QualityScorer._strip_thinking) matches the full block including the
# closing tag. Keeping the open/close fragments paired here prevents the
# bug where we warn about a tag form but fail to strip it.
_THINKING_OPEN = (
    r"(?:<(?:think|thinking|reasoning|reflection|analysis)>"
    r"|<\|(?:begin_of_thought|im_thinking|thought)\|>)"
)
_THINKING_CLOSE = (
    r"(?:</(?:think|thinking|reasoning|reflection|analysis)>"
    r"|<\|(?:end_of_thought|im_end_thinking|end_thought)\|>)"
)
_THINKING_TAG_RE = re.compile(_THINKING_OPEN, re.IGNORECASE)


def _has_thinking(text: str) -> bool:
    """Return True if `text` contains a thinking-style opening tag."""
    if not text:
        return False
    return bool(_THINKING_TAG_RE.search(text))


_THINKING_CLOSED_BLOCK_RE = re.compile(
    _THINKING_OPEN + r"(.*?)" + _THINKING_CLOSE,
    flags=re.DOTALL | re.IGNORECASE,
)
_THINKING_UNCLOSED_BLOCK_RE = re.compile(
    _THINKING_OPEN + r"(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _split_inline_thinking(text: str) -> tuple[str, str]:
    """Split inline `<think>...</think>` blocks out of `text`.

    Returns (clean_text, thinking_text). The thinking_text concatenates the
    body of each matched closed block followed by the body of any trailing
    unclosed block (model hit max_tokens while reasoning). Mirrors the strip
    logic in QualityScorer._strip_thinking so detection and capture stay
    paired — if one changes, the other must too.
    """
    if not text:
        return "", ""
    thinking_parts: list[str] = []
    for m in _THINKING_CLOSED_BLOCK_RE.finditer(text):
        thinking_parts.append(m.group(1))
    text = _THINKING_CLOSED_BLOCK_RE.sub("", text)
    m = _THINKING_UNCLOSED_BLOCK_RE.search(text)
    if m:
        thinking_parts.append(m.group(1))
        text = _THINKING_UNCLOSED_BLOCK_RE.sub("", text)
    captured = "\n".join(p.strip() for p in thinking_parts if p.strip())
    return text.strip(), captured


# Substring patterns that identify thinking-capable model families.
# Match is done against model.lower(), so patterns are lowercase substrings.
# Covers: DeepSeek-R1, Qwen3.x, GLM-4.5/4.6/5.x (ZhipuAI), Nemotron-3 nano
# (NVIDIA hybrid reasoner), and Phi-4-reasoning variants.
_THINKING_MODEL_PATTERNS = (
    "deepseek-r1",
    "qwen3",
    "glm-4.5",
    "glm-4.6",
    "glm-5",
    "nemotron",
    "phi-4-reasoning",
    "phi4-reasoning",
    "gemma4"
)


def _is_thinking_model(model: str) -> bool:
    """
    Detect models that emit thinking blocks (inline tags or server-side
    reasoning_content channels) in their output.

    These models include internal reasoning wrapped in XML-style tags or
    routed through a separate reasoning channel. When thinking is not
    explicitly disabled, the runtime must either strip the tags or read
    `reasoning_content` to get the actual answer.
    """
    m = model.lower()
    return any(p in m for p in _THINKING_MODEL_PATTERNS)


def _thinking_disable_token(model: str) -> str | None:
    """Family-specific prompt directive that disables thinking, or None if
    no documented opt-out exists for the family. Used as a fallback when
    server-side `enable_thinking: false` is silently ignored by the backend
    (observed on Ollama Cloud's GLM-5.1)."""
    m = model.lower()
    if "qwen3" in m:
        return "/no_think"
    if "glm-4.5" in m or "glm-4.6" in m or "glm-5" in m:
        return "/nothink"
    if "nemotron" in m:
        return "/no_think"
    return None


def _is_ollama_endpoint(base_url: str | None) -> bool:
    """Heuristic: True if the base_url targets an Ollama server (cloud or
    local). Used to enable the native /api/chat code path, which honors the
    `think: false` flag that the OpenAI-compat shim silently drops on some
    cloud-hosted thinking models (observed: glm-5.1:cloud)."""
    if not base_url:
        return False
    u = base_url.lower()
    return ("ollama.com" in u) or ("localhost:11434" in u) or ("127.0.0.1:11434" in u)


def _ollama_native_url(base_url: str) -> str:
    """Convert an OpenAI-compat Ollama base URL to the native /api/chat URL."""
    u = base_url.rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u + "/api/chat"


def _user_disabled_thinking(extra_body: dict | None) -> bool:
    """True by Default - Unless user has changed that."""

    if not extra_body:
        return True
    for k in ("enable_thinking", "thinking", "think", "reasoning"):
        if extra_body.get(k) is True:
            return False
    cks = extra_body.get("chat_template_kwargs") or {}
    if isinstance(cks, dict):
        for k in ("enable_thinking", "thinking", "think"):
            if cks.get(k) is True:
                return False
    return True


def _openai_tool_to_anthropic(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI-shaped tool definition to Anthropic's shape.

    OpenAI: ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Anthropic: ``{"name", "description", "input_schema"}``

    Pass-through if the input already looks Anthropic-shaped (has
    ``input_schema``).
    """
    if "input_schema" in tool and "function" not in tool:
        return dict(tool)
    fn = tool.get("function") or {}
    out: dict[str, Any] = {
        "name": fn.get("name") or tool.get("name", ""),
        "input_schema": fn.get("parameters") or tool.get("parameters") or {"type": "object"},
    }
    desc = fn.get("description") or tool.get("description")
    if desc:
        out["description"] = desc
    return out


def _normalize_anthropic_tool_choice(
    tool_choice: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Translate OpenAI tool_choice values into Anthropic's tool_choice dict.

    OpenAI accepts ``"auto"``, ``"none"``, ``"required"`` or ``{"type":"function","function":{"name":"..."}}``.
    Anthropic uses ``{"type":"auto"|"any"|"none"|"tool", "name": "..."}``.
    """
    if isinstance(tool_choice, str):
        return {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "required": {"type": "any"},
            "any": {"type": "any"},
        }.get(tool_choice, {"type": "auto"})
    # Mapping form
    if tool_choice.get("type") in ("auto", "any", "none", "tool"):
        return dict(tool_choice)
    fn = tool_choice.get("function") or {}
    name = fn.get("name") or tool_choice.get("name")
    if name:
        return {"type": "tool", "name": name}
    return {"type": "auto"}


def _extract_ollama_tool_calls(msg: Any) -> tuple[ToolCall, ...] | None:
    """Pull tool calls off an Ollama-shaped message.

    Ollama's response message exposes ``tool_calls`` as a list of objects
    with ``.function.name`` and ``.function.arguments`` (dict, not JSON
    string). We canonicalize ``arguments`` to a JSON string to match the
    OpenAI/library-wide convention.
    """
    if msg is None:
        return None
    raw = (
        msg.get("tool_calls") if isinstance(msg, dict)
        else getattr(msg, "tool_calls", None)
    )
    if not raw:
        return None
    out: list[ToolCall] = []
    for tc in raw:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name", "")
            arguments = fn.get("arguments")
        else:
            fn = getattr(tc, "function", None)
            name = (getattr(fn, "name", "") if fn is not None else "") or ""
            arguments = getattr(fn, "arguments", None) if fn is not None else None
        if arguments is None:
            arguments = {}
        out.append(ToolCall(
            id=(tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")) or "",
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ))
    return tuple(out)


def _extract_openai_tool_calls(choice: Any) -> tuple[ToolCall, ...] | None:
    """Pull tool calls off an OpenAI-shaped completion choice.

    Returns None when the model produced none. Defensive against SDK
    variation: the OpenAI Python client exposes tool_calls as objects with
    ``id`` / ``function.name`` / ``function.arguments``; some compat
    backends return dicts.
    """
    msg = getattr(choice, "message", None)
    raw = getattr(msg, "tool_calls", None) if msg is not None else None
    if not raw:
        return None
    out: list[ToolCall] = []
    for tc in raw:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            out.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name") or tc.get("name", ""),
                arguments=(
                    fn.get("arguments") if isinstance(fn.get("arguments"), str)
                    else json.dumps(fn.get("arguments") or {})
                ),
            ))
        else:
            fn = getattr(tc, "function", None)
            args = getattr(fn, "arguments", None) if fn is not None else None
            out.append(ToolCall(
                id=getattr(tc, "id", "") or "",
                name=(getattr(fn, "name", "") if fn is not None else "") or "",
                arguments=args if isinstance(args, str) else json.dumps(args or {}),
            ))
    return tuple(out)


class AgentChannel:
    """
    Wraps any LLM endpoint as a stochastic channel.

    Supports multiple providers:
    - OpenAI (default)
    - Anthropic (native SDK, auto-detected from model name)
    - Ollama, vLLM, DeepSeek, Together, OpenRouter, etc. (OpenAI-compatible)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 32768,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        extra_body: dict | None = None,
        category_temperatures: dict[str, float] | None = None,
        cost_per_1m: tuple[float, float] | dict | None = None,
        thinking: bool | dict | str | None = None,
        timeout_s: float | None = None,
    ):
        # cost_per_1m: per-channel pricing override. Accepts either a
        # (input_per_1M, output_per_1M) tuple or a {"input": x, "output": y}
        # dict. Wins over MODEL_COSTS / parameter-count inference. When set,
        # AgentOutput.cost_source on every emitted output is "exact_user_rate".
        if cost_per_1m is None:
            self.cost_per_1m: tuple[float, float] | None = None
        elif isinstance(cost_per_1m, dict):
            self.cost_per_1m = (float(cost_per_1m["input"]), float(cost_per_1m["output"]))
        else:
            self.cost_per_1m = (float(cost_per_1m[0]), float(cost_per_1m[1]))

        # thinking: high-level switch translated into provider-specific keys
        # by _translate_thinking_to_extra_body() and merged into extra_body.
        # Accepts:
        #   None / False / "auto" — disabled (default; backward-compatible)
        #   True                  — enabled, no token budget
        #   {"enabled": bool, "budget_tokens": int|None}
        # Translation per backend:
        #   - Anthropic: thinking={"type":"enabled","budget_tokens":N}
        #   - OpenAI o-series / GPT-5: reasoning_effort=low|medium|high
        #   - Ollama / vLLM: chat_template_kwargs.enable_thinking=True
        self.thinking_config = self._normalize_thinking_config(thinking)
        extra_body = self._translate_thinking_to_extra_body(
            self.thinking_config, model, extra_body,
        )

        self.model = model
        # `temperature` is the *current effective* value used by transmit().
        # `base_temperature` is the configured fallback that we restore after
        # a per-task swap. `category_temperatures` maps a TaskCategory.value
        # (e.g. "qa", "code") to the temperature to use for that category.
        # The runner swaps `temperature` in/out around each technique run when
        # `ExperimentConfig.per_category_temperature` is enabled.
        self.temperature = temperature
        self.base_temperature = temperature
        self.category_temperatures = dict(category_temperatures or {})
        # Per-thread temperature override. The runner sets
        # `_tls.temperature_override` from `_category_temperature_swap` before
        # dispatching a task and clears it afterward. Storing it per-thread
        # keeps concurrent (task, repeat) workers isolated when
        # parallel_tasks > 1, so two tasks of different categories running on
        # the same channel each see their own temperature.
        self._tls = threading.local()
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.extra_body = extra_body or {}
        self.is_anthropic = _is_anthropic_model(model)
        self._warned_thinking = False
        # Stored for the native-Ollama fallback path; the OpenAI client wraps
        # them, but we need them again to talk to /api/chat directly when
        # the OpenAI-compat shim drops `think: false`.
        self.base_url = base_url
        # API key resolution:
        #   - Anthropic endpoint: handled below in the anthropic branch.
        #   - Ollama endpoint: prefer OLLAMA_API_KEY from env (loaded via
        #     dotenv from .env by run_benchmark.py). This is the right key
        #     for auth-fronted Ollama deployments (e.g. behind a reverse
        #     proxy or hosted Ollama). The historical YAML placeholder
        #     `api_key: "ollama"` was just a non-empty filler that real
        #     Ollama servers ignore — when env supplies a real key it wins
        #     over that placeholder so existing configs work unchanged.
        #     A genuine YAML api_key (anything other than "ollama") still
        #     wins over the env var. Falls back to "ollama" so the OpenAI
        #     client doesn't refuse to construct on a missing key.
        #     Crucially, we do NOT fall back to OPENAI_API_KEY here — that
        #     would leak the real OpenAI key into Ollama traffic.
        #   - Other (OpenAI / OpenRouter / etc.): OPENAI_API_KEY env fallback.
        if _is_ollama_endpoint(base_url):
            if api_key and api_key != "ollama":
                self.api_key = api_key
            else:
                self.api_key = os.environ.get("OLLAMA_API_KEY") or api_key or "ollama"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Effective per-call HTTP timeout for OpenAI-compat clients: explicit
        # `timeout_s` wins, else the built-in default (Ollama is slower to
        # first token on cold / large / reasoning models, so it gets a longer
        # default). `_timeout_override` is the user's explicit value (or None)
        # — used to decide whether to override Anthropic's own SDK default.
        self._timeout_override = timeout_s
        self._timeout_s = timeout_s or (
            300 if _is_ollama_endpoint(base_url) else 240
        )

        if self.is_anthropic:
            anthropic = _import_anthropic()
            anthropic_kwargs: dict[str, Any] = {
                "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
            }
            # Only override Anthropic's (generous) SDK default when asked.
            if timeout_s is not None:
                anthropic_kwargs["timeout"] = timeout_s
            self.anthropic_client = anthropic.Anthropic(**anthropic_kwargs)
            self.client = None
        else:
            OpenAI = _import_openai()
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=self._timeout_s,
            )
            self.anthropic_client = None

        # Native async clients are created lazily on first .atransmit() /
        # .atransmit_stream() call so importing the channel doesn't pull
        # the async SDK paths for callers that only use the sync API.
        self._async_client: Any | None = None
        self._async_anthropic_client: Any | None = None

    def _get_async_client(self) -> Any:
        """Lazy-init the native async OpenAI-compat client (also used for
        Ollama via the OpenAI-compat shim and for vLLM/OpenRouter/etc.)."""
        if self._async_client is None:
            AsyncOpenAI = _import_openai_async()
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._timeout_s,
            )
        return self._async_client

    def _get_async_anthropic_client(self) -> Any:
        """Lazy-init the native AsyncAnthropic client."""
        if self._async_anthropic_client is None:
            AsyncAnthropic = _import_anthropic_async()
            anthropic_kwargs: dict[str, Any] = {
                "api_key": self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            }
            if self._timeout_override is not None:
                anthropic_kwargs["timeout"] = self._timeout_override
            self._async_anthropic_client = AsyncAnthropic(**anthropic_kwargs)
        return self._async_anthropic_client

    def transmit(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
        request_logprobs: bool = False,
    ) -> AgentOutput:
        """
        Send a chat request through the channel and receive a noisy output.
        This is the fundamental channel operation: Y = A(X) + N

        Accepts either a plain string (back-compat, wrapped in a single-turn
        :class:`ChatRequest` using the channel's configured ``system_prompt``)
        or a :class:`ChatRequest` carrying a full conversation, tools,
        response_format, etc.

        Note: thinking/reasoning tags in the output are detected and stripped;
        the first occurrence per channel instance is logged as a warning so
        callers can spot when `enable_thinking: false` (or equivalent) is not
        actually being honored by the backend.

        If request_logprobs=True, the output's token_logprobs / mean_logprob
        fields are populated (when the backend supports it). This enables
        soft-output techniques that use per-token confidence.
        """
        # Normalize input to a ChatRequest so downstream transports see one shape.
        # The channel's configured system_prompt is honored when the request
        # didn't bring its own — keeping every legacy `transmit("string")` call
        # unchanged in behavior.
        if isinstance(prompt_or_request, str):
            request = ChatRequest.from_prompt(
                prompt_or_request, system=self.system_prompt,
            )
        else:
            request = prompt_or_request
            if request.system is None and self.system_prompt:
                request = request.with_system(self.system_prompt)
        if request_logprobs and not request.request_logprobs:
            request = request.with_extra(request_logprobs=True)

        # Resolve temperature: explicit kwarg > per-thread override (set by
        # the runner's per-category swap) > channel default. The TLS path is
        # what makes parallel_tasks > 1 safe with per_category_temperature.
        if temperature is not None:
            temp = temperature
        else:
            temp = getattr(self._tls, "temperature_override", None)
            if temp is None:
                temp = self.temperature
        # A per-call temperature on the request wins over the channel default
        # so users can pass `temperature=` to `mod.run(messages=…)`.
        if request.temperature is not None and temperature is None:
            temp = request.temperature

        # Transient backend failures (HTTP 5xx, connection resets, timeouts)
        # used to kill a run after a single attempt — `_validate_run_or_raise`
        # would then drop the entire task. Retry the transmission with
        # exponential backoff before giving up. Anthropic's SDK already does
        # its own retries, so we only wrap the OpenAI / Ollama-native paths.
        MAX_TRANSMIT_ATTEMPTS = 3
        t0 = time.time()
        if self.is_anthropic:
            try:
                logger.info(f"Calling the LLM ({self.model}, temp: {temp}) now...")
                return self._transmit_anthropic(request, temp, prompt_variant, t0)
            except Exception as e:
                logger.error(f"Channel error ({self.model}): {e}")
                return AgentOutput(
                    text=f"[ERROR: {e}]",
                    model=self.model, temperature=temp, prompt_variant=prompt_variant,
                    quality_score=0.0, latency_s=time.time() - t0,
                )

        last_err: Exception | None = None
        for attempt in range(MAX_TRANSMIT_ATTEMPTS):
            try:
                logger.info(
                    f"Calling the LLM ({self.model}, temp: {temp}) "
                    f"now (attempt {attempt + 1}/{MAX_TRANSMIT_ATTEMPTS})..."
                )
                # Native Ollama fallback: only when the OpenAI-compat shim is
                # known to drop the thinking-disable flag for this model.
                # Logprobs aren't supported by the native API, so soft-output
                # techniques stay on the OpenAI-compat path.
                if (
                    # not request_logprobs
                    _is_ollama_endpoint(self.base_url)
                    # and _is_thinking_model(self.model)
                    # and _user_disabled_thinking(self.extra_body)
                ):
                    # Backend selection (default = ollama Python library):
                    # AGENTCODEC_OLLAMA_BACKEND=python -> ollama library (default)
                    # AGENTCODEC_OLLAMA_BACKEND=urllib -> stdlib urllib fallback
                    if os.environ.get("AGENTCODEC_OLLAMA_BACKEND", "python") == "urllib":
                        return self._transmit_ollama_native(request, temp, prompt_variant, t0)
                    return self._transmit_ollama_python(request, temp, prompt_variant, t0, request_logprobs)
                return self._transmit_openai(
                    request, temp, prompt_variant, t0,
                    request_logprobs=request_logprobs,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    f"Channel error ({self.model}, attempt "
                    f"{attempt + 1}/{MAX_TRANSMIT_ATTEMPTS}): {e}"
                )
                if attempt < MAX_TRANSMIT_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s
        logger.error(
            f"Channel error ({self.model}) after {MAX_TRANSMIT_ATTEMPTS} attempts: "
            f"{last_err}"
        )
        return AgentOutput(
            text=f"[ERROR: {last_err}]",
            model=self.model, temperature=temp, prompt_variant=prompt_variant,
            quality_score=0.0, latency_s=time.time() - t0,
        )

    async def atransmit(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
    ) -> AgentOutput:
        """Async one-shot transmission via the native provider async SDK.

        Internally drives :meth:`atransmit_stream` and discards intermediate
        chunks, so any provider-supported thinking content still lands on
        the returned :class:`AgentOutput` (``thinking_text``, ``thinking_*``)
        — same shape as sync :meth:`transmit`.
        """
        final_output: AgentOutput | None = None
        async for frame in self.atransmit_stream(
            prompt_or_request,
            temperature=temperature,
            prompt_variant=prompt_variant,
        ):
            if isinstance(frame, ChannelDone):
                final_output = frame.output
        if final_output is None:
            raise RuntimeError(
                "atransmit_stream did not produce a ChannelDone frame — "
                "backend implementation is broken."
            )
        return final_output

    async def atransmit_stream(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
    ) -> AsyncIterator[ChannelChunk | ChannelDone]:
        """Stream a chat request through the channel asynchronously.

        Yields :class:`ChannelChunk` frames as the model produces tokens
        (role ``"answer"`` for the user-facing answer, ``"thinking"`` for
        provider-internal reasoning, ``"tool_call"`` for tool invocations).
        Closes with exactly one :class:`ChannelDone` carrying the fully
        aggregated :class:`AgentOutput` — same shape sync :meth:`transmit`
        would have returned, including all ``thinking_*`` telemetry.

        Backends:

        - Anthropic models (``claude-*``) → native ``messages.stream``,
          parses ``thinking_delta`` / ``text_delta`` / ``input_json_delta``.
        - Ollama-native endpoint with native streaming enabled →
          line-delimited JSON from ``/api/chat``, parses ``content`` /
          ``thinking`` fields.
        - Everything else (OpenAI, OpenRouter, vLLM, Ollama-via-compat) →
          ``AsyncOpenAI.chat.completions.create(stream=True)``, parses
          ``delta.content`` / ``delta.reasoning_content`` / ``delta.tool_calls``.

        Cancellation: closing the iterator early closes the underlying
        provider stream — unlike sync :meth:`transmit`, which can't
        interrupt an in-flight HTTP read.
        """
        if isinstance(prompt_or_request, str):
            request = ChatRequest.from_prompt(
                prompt_or_request, system=self.system_prompt,
            )
        else:
            request = prompt_or_request
            if request.system is None and self.system_prompt:
                request = request.with_system(self.system_prompt)

        # Resolve effective temperature — same precedence as sync transmit().
        if temperature is not None:
            temp = temperature
        else:
            temp = getattr(self._tls, "temperature_override", None)
            if temp is None:
                temp = self.temperature
        if request.temperature is not None and temperature is None:
            temp = request.temperature

        t0 = time.time()
        if self.is_anthropic:
            async for frame in self._atransmit_stream_anthropic(
                request, temp, prompt_variant, t0,
            ):
                yield frame
        elif _is_ollama_endpoint(self.base_url):
            # Mirror the sync routing: native /api/chat path for Ollama
            # endpoints so `think: false` is actually honored when set.
            async for frame in self._atransmit_stream_ollama_native(
                request, temp, prompt_variant, t0,
            ):
                yield frame
        else:
            async for frame in self._atransmit_stream_openai(
                request, temp, prompt_variant, t0,
            ):
                yield frame

    @staticmethod
    def _normalize_thinking_config(
        thinking: bool | dict | str | None,
    ) -> dict:
        """Normalize the user's `thinking` value to a canonical dict.

        Returns {"enabled": bool, "budget_tokens": int|None, "raw": <as_given>}.
        The "raw" value is preserved so downstream layers can detect "auto"
        vs explicit-False, which differ in the construction-time warning
        ("auto" is silent; explicit-False is also silent; *unspecified* on a
        thinking-capable model triggers a one-time notice).
        """
        if thinking is None or thinking is False or thinking == "auto":
            return {"enabled": False, "budget_tokens": None, "raw": thinking}
        if thinking is True:
            return {"enabled": True, "budget_tokens": None, "raw": True}
        if isinstance(thinking, dict):
            enabled = bool(thinking.get("enabled", True))
            budget = thinking.get("budget_tokens")
            if budget is not None:
                budget = int(budget)
            # Reject the seconds-budget gotcha loudly.
            if "budget_seconds" in thinking:
                raise ValueError(
                    "thinking.budget_seconds is not supported. The underlying "
                    "APIs (Anthropic, OpenAI o-series, Ollama, vLLM) only "
                    "support token budgets. Use `budget_tokens` instead, or "
                    "wrap the call in your own asyncio timeout."
                )
            return {"enabled": enabled, "budget_tokens": budget, "raw": thinking}
        raise ValueError(
            f"thinking must be bool, dict, 'auto', or None — got {thinking!r}"
        )

    @staticmethod
    def _translate_thinking_to_extra_body(
        thinking_cfg: dict,
        model: str,
        extra_body: dict | None,
    ) -> dict:
        """Inject backend-specific thinking flags into extra_body.

        - Anthropic: extra_body["thinking"] = {"type": "enabled", "budget_tokens": N}
        - OpenAI reasoning (gpt-5, o-series): extra_body["reasoning_effort"] = ...
        - Ollama / vLLM thinking models: extra_body["chat_template_kwargs"]["enable_thinking"] = True

        User-provided keys in extra_body always win over our auto-translation
        so callers can override per-call.
        """
        eb: dict = dict(extra_body or {})
        if not thinking_cfg["enabled"]:
            # When explicitly disabled, do NOT set enable_thinking:false here —
            # the existing _user_disabled_thinking() logic already handles that
            # with its default-disabled semantics. We only inject when enabling.
            return eb

        budget = thinking_cfg["budget_tokens"]

        if _is_anthropic_model(model):
            if "thinking" not in eb:
                payload: dict[str, Any] = {"type": "enabled"}
                if budget is not None:
                    payload["budget_tokens"] = budget
                eb["thinking"] = payload
        elif _is_openai_reasoning_model(model):
            if "reasoning_effort" not in eb:
                # Coarse mapping from token budget to OpenAI's qualitative scale.
                if budget is None:
                    eb["reasoning_effort"] = "medium"
                elif budget < 2048:
                    eb["reasoning_effort"] = "low"
                elif budget < 8192:
                    eb["reasoning_effort"] = "medium"
                else:
                    eb["reasoning_effort"] = "high"
        elif _is_thinking_model(model):
            cks = dict(eb.get("chat_template_kwargs") or {})
            cks.setdefault("enable_thinking", True)
            eb["chat_template_kwargs"] = cks
            # Ollama native /api/chat reads `think` at top level.
            eb.setdefault("think", True)
        return eb

    def _attribute_thinking_cost(
        self,
        cost_breakdown: Any,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
    ) -> tuple[float, float]:
        """Split total cost into (thinking_cost, answer_cost).

        Both thinking and answer tokens are billed at the output rate by every
        provider we support. We split proportionally so the host can see how
        much of the bill went to hidden reasoning.
        """
        if output_tokens <= 0:
            return 0.0, cost_breakdown.cost_usd
        thinking_share = max(0.0, min(1.0, thinking_tokens / output_tokens))
        output_cost = output_tokens * cost_breakdown.rate_output_per_1m / 1_000_000.0
        input_cost = input_tokens * cost_breakdown.rate_input_per_1m / 1_000_000.0
        thinking_cost = output_cost * thinking_share
        answer_cost = input_cost + (output_cost - thinking_cost)
        return thinking_cost, answer_cost

    def _build_thinking_kwargs(
        self,
        *,
        thinking_text: str,
        answer_text: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens_exact: int | None = None,
        thinking_tokens_source: str | None = None,
    ) -> dict[str, Any]:
        """Compute the thinking_* AgentOutput kwargs from captured reasoning.

        Single source of truth used by every transmit path so the fields stay
        in sync across backends.

        - ``thinking_text``: captured reasoning content. Empty string when the
          model did not emit thinking on this call.
        - ``answer_text``: final user-facing text (post-strip). Used to
          estimate thinking-token share when the API doesn't break it down.
        - ``thinking_tokens_exact``: pass when the backend reports it
          separately (e.g. OpenAI's ``completion_tokens_details.reasoning_tokens``).
          When ``None`` we estimate via char-share of total output text.
        - ``thinking_tokens_source``: label describing how the count was
          derived. Suggested values: ``"api_exact"``, ``"char_share_estimate"``,
          ``"inline_tag_strip"``.
        """
        thinking_text = thinking_text or ""
        answer_text = answer_text or ""
        emitted = bool(thinking_text)
        chars = len(thinking_text)

        supported = (
            _is_thinking_model(self.model)
            or _is_openai_reasoning_model(self.model)
            or _is_anthropic_model(self.model)
        )
        cfg = self.thinking_config or {}
        enabled = bool(cfg.get("enabled", False))

        if not emitted:
            thinking_tokens = 0
            tokens_source: str | None = None
        elif thinking_tokens_exact is not None:
            thinking_tokens = max(0, int(thinking_tokens_exact))
            tokens_source = thinking_tokens_source or "api_exact"
        else:
            total_chars = chars + len(answer_text)
            if total_chars > 0 and output_tokens > 0:
                share = chars / total_chars
                thinking_tokens = round(output_tokens * share)
                tokens_source = thinking_tokens_source or "char_share_estimate"
            else:
                thinking_tokens = 0
                tokens_source = thinking_tokens_source

        rate = self.cost_per_1m or _resolve_costs(self.model)
        input_rate, output_rate = rate[0], rate[1]
        output_cost = output_tokens * output_rate / 1_000_000.0
        input_cost = input_tokens * input_rate / 1_000_000.0
        if output_tokens > 0:
            share = max(0.0, min(1.0, thinking_tokens / output_tokens))
        else:
            share = 0.0
        thinking_cost = output_cost * share
        answer_cost = input_cost + (output_cost - thinking_cost)

        return dict(
            thinking_supported=supported,
            thinking_enabled=enabled,
            thinking_emitted=emitted,
            thinking_text=(thinking_text if emitted else None),
            thinking_chars=chars,
            thinking_tokens=thinking_tokens,
            thinking_tokens_source=tokens_source,
            thinking_cost_usd=thinking_cost,
            answer_tokens=max(0, output_tokens - thinking_tokens),
            answer_cost_usd=answer_cost,
        )

    def temperature_for_category(self, category: Any) -> float:
        """
        Return the effective temperature for a given task category.

        Falls back to `base_temperature` when no override is configured for
        that category (or when `category_temperatures` is empty).
        Accepts a TaskCategory enum or its `.value` string.
        """
        if not self.category_temperatures:
            return self.base_temperature
        key = category.value if hasattr(category, "value") else str(category)
        return self.category_temperatures.get(key, self.base_temperature)

    def _warn_thinking_once(self, text: str) -> None:
        """Log a warning every time a model emits thinking-style tags.
        Name kept for backwards compatibility — no longer deduped per-instance.
        """
        match = _THINKING_TAG_RE.search(text)
        tag = match.group(0) if match else "<think>"
        logger.warning(
            f"Detected {tag} in output from model={self.model!r}. "
            f"Tags will be stripped before downstream consumers see the text. "
            f"If you intended thinking to be disabled, check that the backend "
            f"honors your extra_body (e.g. chat_template_kwargs.enable_thinking)."
        )

    def _transmit_openai(
        self, request: ChatRequest, temp: float, prompt_variant: str, t0: float,
        request_logprobs: bool = False,
    ) -> AgentOutput:
        # If the user opted out of thinking via extra_body but the backend
        # ignores it (observed on Ollama Cloud's GLM-5.1), prepend the
        # family-specific prompt directive (e.g. /no_think for qwen3,
        # /nothink for glm-4.5+). Belt-and-suspenders: the extra_body flags
        # stay in place; this is a fallback when the server silently drops
        # them.
        if _is_thinking_model(self.model) and _user_disabled_thinking(self.extra_body):
            tok = _thinking_disable_token(self.model)
            if tok:
                request = request.with_user(f"{tok}\n{request.last_user_text}")

        messages = request.to_openai_messages()
        kwargs: dict = {"model": self.model, "messages": messages}

        # GPT-5 / o-series use max_completion_tokens instead of max_tokens.
        max_tokens = request.max_tokens or self.max_tokens
        if _is_openai_reasoning_model(self.model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        # o1/o3/o4 only accept temperature=1.0. GPT-5 accepts normal temps,
        # but if reasoning_effort > minimal is set, temperature is ignored anyway.
        if _is_strict_reasoning_model(self.model):
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temp

        # Per-call request-level kwargs. None values are skipped so we don't
        # send `tools=None` to backends that reject it.
        if request.tools:
            kwargs["tools"] = [dict(t) for t in request.tools]
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            kwargs["response_format"] = dict(request.response_format)
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Soft-output: request token log-probabilities for soft-decision techniques.
        if request_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5

        # Known SDK-level kwargs that the OpenAI client accepts directly
        # (e.g., reasoning_effort for o-series). Everything else goes via
        # extra_body so it's sent in the HTTP request body (needed for
        # Ollama's enable_thinking, etc.).
        _SDK_KWARGS = {"reasoning_effort"}
        sdk_kwargs = {k: v for k, v in self.extra_body.items() if k in _SDK_KWARGS}
        body_kwargs = {k: v for k, v in self.extra_body.items() if k not in _SDK_KWARGS}
        # Per-request extra_body wins over the channel's configured extras
        # so callers can override per call.
        if request.extra_body:
            body_kwargs = {**body_kwargs, **dict(request.extra_body)}
        kwargs.update(sdk_kwargs)
        if body_kwargs:
            kwargs["extra_body"] = body_kwargs

        response = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        choice = response.choices[0]
        text = choice.message.content or ""
        # Some OpenAI-compatible backends (vLLM/SGLang/Ollama with a reasoning
        # parser, GLM-4.5+/Nemotron via Ollama-cloud) split the response into
        # `reasoning_content` (the <think> block) and `content` (the answer).
        # If we only read `content`, a model that exhausts max_tokens during
        # reasoning returns silently empty text. Surface that condition.
        reasoning_text = (
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
            or ""
        )
        finish_reason = getattr(choice, "finish_reason", None)
        # Always notify on any thinking, even when a clean answer also arrived.
        # We want loud, every-call signal that thinking is active — silent
        # success masks misconfigured `enable_thinking` flags.
        if reasoning_text:
            content_state = "empty" if not text.strip() else f"{len(text)} chars"
            logger.warning(
                f"Thinking detected from model={self.model!r}: "
                f"reasoning_content={len(reasoning_text)} chars, "
                f"content={content_state}, finish_reason={finish_reason!r}, "
                f"max_tokens={self.max_tokens}. "
                f"To silence, disable thinking via extra_body "
                f"(e.g. chat_template_kwargs.enable_thinking=false)."
            )
            if not text.strip() and finish_reason == "length":
                logger.warning(
                    "  ^ content is empty AND finish_reason='length' — model "
                    "exhausted max_tokens during thinking. Raise max_tokens or "
                    "disable thinking to recover an answer."
                )
        elif not text.strip():
            logger.warning(
                f"Empty content from model={self.model!r} "
                f"(finish_reason={finish_reason!r}, max_tokens={self.max_tokens}). "
                f"No reasoning_content present either."
            )
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        # OpenAI o-series / GPT-5 report reasoning tokens separately. Older
        # backends (vLLM, Ollama) generally don't.
        reasoning_tokens_exact: int | None = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                rt = getattr(details, "reasoning_tokens", None)
                if rt is not None:
                    reasoning_tokens_exact = int(rt)

        logger.info(f"Response received: {text[:100].strip().replace(chr(10), '')}...")

        # Extract token-level logprobs if requested and available.
        token_logprobs: list[float] | None = None
        mean_logprob: float | None = None
        top_logprobs_per_token: list[dict[str, float]] | None = None
        if request_logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [t.logprob for t in choice.logprobs.content]
            if token_logprobs:
                mean_logprob = sum(token_logprobs) / len(token_logprobs)
            # Capture per-position top-k alternatives. Each `t.top_logprobs`
            # is a list of objects with `.token` and `.logprob` (OpenAI SDK
            # shape, also returned by vLLM/SGLang/Ollama-cloud).
            top_logprobs_per_token = []
            for t in choice.logprobs.content:
                tops = getattr(t, "top_logprobs", None) or []
                top_logprobs_per_token.append(
                    {tt.token: tt.logprob for tt in tops}
                )

        # Strip <think>...</think> blocks from reasoning models (DeepSeek-R1,
        # Qwen3) so downstream consumers (critic, synthesizer, judge) see only
        # the final answer. Capture the stripped span into thinking_text so
        # the host can still inspect the reasoning.
        inline_thinking = ""
        if _is_thinking_model(self.model):
            if _has_thinking(text):
                self._warn_thinking_once(text)
                text, inline_thinking = _split_inline_thinking(text)
            else:
                text = QualityScorer._strip_thinking(text)
        elif _has_thinking(text):
            self._warn_thinking_once(text)
            text, inline_thinking = _split_inline_thinking(text)

        thinking_text = "\n".join(p for p in [reasoning_text, inline_thinking] if p)
        if reasoning_tokens_exact is not None and reasoning_tokens_exact > 0:
            tokens_source = "openai_reasoning_tokens"
            tokens_exact: int | None = reasoning_tokens_exact
        elif reasoning_text:
            tokens_source = "reasoning_content_field"
            tokens_exact = None
        elif inline_thinking:
            tokens_source = "inline_tag_strip"
            tokens_exact = None
        else:
            tokens_source = None
            tokens_exact = None
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_exact=tokens_exact,
            thinking_tokens_source=tokens_source,
        )

        # Surface tool_calls when the model emitted any. None when absent
        # (most reliability runs don't request tools).
        tool_calls = _extract_openai_tool_calls(choice)

        return AgentOutput(
            text=text,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            token_logprobs=token_logprobs,
            mean_logprob=mean_logprob,
            top_logprobs_per_token=top_logprobs_per_token,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )

    def _transmit_ollama_native(
        self, request: ChatRequest, temp: float, prompt_variant: str, t0: float
    ) -> AgentOutput:
        """Talk to Ollama's native /api/chat endpoint with `think: false` in
        the request body — the OpenAI-compat shim drops this field for some
        cloud-hosted thinking models (e.g. glm-5.1:cloud), so we bypass it.
        Logprobs are not exposed by the native API; this path is therefore
        only routed to from `transmit()` when request_logprobs=False."""
        max_tokens = request.max_tokens or self.max_tokens
        options: dict[str, Any] = {
            "temperature": temp,
            "num_predict": max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = list(request.stop)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request.to_ollama_messages(),
            "think": False,
            "stream": False,
            "options": options,
        }
        if request.tools:
            body["tools"] = [dict(t) for t in request.tools]
        if request.response_format is not None:
            # Ollama accepts a JSON schema or the literal "json".
            rf = dict(request.response_format)
            body["format"] = "json" if rf.get("type") == "json_object" else rf
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = _ollama_native_url(self.base_url or "")
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        latency = time.time() - t0

        msg = payload.get("message") or {}
        text = msg.get("content") or ""
        reasoning_text = msg.get("thinking") or ""
        finish_reason = "length" if payload.get("done_reason") == "length" else (
            payload.get("done_reason") or ("stop" if payload.get("done") else None)
        )

        

        if reasoning_text:
            content_state = "empty" if not text.strip() else f"{len(text)} chars"
            logger.warning(
                f"Thinking detected from model={self.model!r} (native Ollama): "
                f"thinking={len(reasoning_text)} chars, content={content_state}, "
                f"done_reason={finish_reason!r}, max_tokens={self.max_tokens}. "
                f"Native `think: false` was sent and still not honored — the "
                f"model's chat template hard-codes thinking; swap the judge."
            )
        elif not text.strip():
            logger.warning(
                f"Empty content from model={self.model!r} (native Ollama, "
                f"done_reason={finish_reason!r}, max_tokens={self.max_tokens})."
            )
        input_tokens = int(payload.get("prompt_eval_count") or 0)
        output_tokens = int(payload.get("eval_count") or 0)

        logger.info(f"Response received: {text[:100].strip().replace(chr(10), '')}...")

        inline_thinking = ""
        if _is_thinking_model(self.model):
            if _has_thinking(text):
                self._warn_thinking_once(text)
                text, inline_thinking = _split_inline_thinking(text)
            else:
                text = QualityScorer._strip_thinking(text)
        elif _has_thinking(text):
            self._warn_thinking_once(text)
            text, inline_thinking = _split_inline_thinking(text)

        thinking_text = "\n".join(p for p in [reasoning_text, inline_thinking] if p)
        if reasoning_text:
            tokens_source = "ollama_thinking_field"
        elif inline_thinking:
            tokens_source = "inline_tag_strip"
        else:
            tokens_source = None
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_source=tokens_source,
        )

        tool_calls = _extract_ollama_tool_calls(msg)

        return AgentOutput(
            text=text,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )

    def _transmit_ollama_python(
        self, request: ChatRequest, temp: float, prompt_variant: str, t0: float, request_logprobs: bool
    ) -> AgentOutput:
        """Talk to Ollama via the official `ollama` Python library — typed
        access to think=False without the urllib JSON marshaling. Opt-in via
        AGENTCODEC_OLLAMA_BACKEND=python. Logprobs are not exposed by the
        library; routed only when request_logprobs=False (same as the urllib
        path)."""
        import ollama

        # Derive host from base_url (strip /v1 suffix used by the OpenAI-compat
        # configuration; the ollama client wants the bare host).
        host = (self.base_url or "http://localhost:11434").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = ollama.Client(host=host, headers=headers, timeout=300)
        max_tokens = request.max_tokens or self.max_tokens
        options: dict[str, Any] = {
            "temperature": temp,
            "num_predict": max_tokens,
            # "logprobs": request_logprobs,
            # # Bumped from 1 → 10 so CISC's P(True) extraction can read
            # # the logprobs of the literal "0" / "1" tokens at the first
            # # generated position. Negligible cost; only takes effect
            # # when logprobs are requested.
            # 'num_probs': 10 if request_logprobs else 1,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = list(request.stop)
        chat_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=request.to_ollama_messages(),
            think=False,
            stream=False,
            logprobs=request_logprobs,
            top_logprobs=10 if request_logprobs else 1,
            options=options,
        )
        if request.tools:
            chat_kwargs["tools"] = [dict(t) for t in request.tools]
        if request.response_format is not None:
            rf = dict(request.response_format)
            chat_kwargs["format"] = "json" if rf.get("type") == "json_object" else rf
        response = client.chat(**chat_kwargs)
        latency = time.time() - t0

        msg = response.message
        text = (msg.content if msg else "") or ""
        reasoning_text = (getattr(msg, "thinking", None) if msg else "") or ""
        done_reason = getattr(response, "done_reason", None)
        finish_reason = "length" if done_reason == "length" else (
            done_reason or ("stop" if getattr(response, "done", False) else None)
        )

        # logger.info(f"MSG:{msg}")

        if reasoning_text:
            content_state = "empty" if not text.strip() else f"{len(text)} chars"
            logger.warning(
                f"Thinking detected from model={self.model!r} (ollama-py): "
                f"thinking={len(reasoning_text)} chars, content={content_state}, "
                f"done_reason={finish_reason!r}, max_tokens={self.max_tokens}. "
                f"`think=False` was sent and still not honored — the model's "
                f"chat template hard-codes thinking; swap the judge."
            )
            if _user_disabled_thinking(self.extra_body):
                raise RuntimeError(
                    f"Thinking detected from model={self.model!r} (ollama-py) "
                    f"despite think=False; the model's chat template hard-codes "
                    f"thinking. Swap the model or remove the disable flag."
                )
        elif not text.strip():
            logger.warning(
                f"Empty content from model={self.model!r} (ollama-py, "
                f"done_reason={finish_reason!r}, max_tokens={self.max_tokens})."
            )
        input_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
        output_tokens = int(getattr(response, "eval_count", 0) or 0)

        logger.info(f"Response received: {text[:100].strip().replace(chr(10), '')}...")

        inline_thinking = ""
        if _is_thinking_model(self.model):
            if _has_thinking(text):
                self._warn_thinking_once(text)
                text, inline_thinking = _split_inline_thinking(text)
            else:
                text = QualityScorer._strip_thinking(text)
        elif _has_thinking(text):
            self._warn_thinking_once(text)
            text, inline_thinking = _split_inline_thinking(text)

        # Parse the Ollama-python logprobs payload defensively. Different
        # library versions expose logprobs at slightly different paths
        # (`response.logprobs`, `response.message.logprobs`); each item is
        # expected to expose `.logprob` and `.top_logprobs` (a list of
        # alternatives with `.token` / `.logprob`).
        token_logprobs: list[float] | None = None
        mean_logprob: float | None = None
        top_logprobs_per_token: list[dict[str, float]] | None = None
        raw_lp = getattr(response, "logprobs", None)
        if raw_lp is None and msg is not None:
            raw_lp = getattr(msg, "logprobs", None)
        if request_logprobs and raw_lp:
            try:
                token_logprobs = [float(item.logprob) for item in raw_lp]
                if token_logprobs:
                    mean_logprob = sum(token_logprobs) / len(token_logprobs)
                top_logprobs_per_token = []
                for item in raw_lp:
                    tops = getattr(item, "top_logprobs", None) or []
                    top_logprobs_per_token.append(
                        {tt.token: float(tt.logprob) for tt in tops}
                    )
            except Exception as e:  # pragma: no cover -- defensive
                logger.warning(
                    f"Failed to parse Ollama logprobs for {self.model!r}: "
                    f"{e!r}. CISC P(True) will be unavailable on this run."
                )
                token_logprobs = None
                mean_logprob = None
                top_logprobs_per_token = None

        thinking_text = "\n".join(p for p in [reasoning_text, inline_thinking] if p)
        if reasoning_text:
            tokens_source = "ollama_thinking_field"
        elif inline_thinking:
            tokens_source = "inline_tag_strip"
        else:
            tokens_source = None
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_source=tokens_source,
        )

        tool_calls = _extract_ollama_tool_calls(msg)

        return AgentOutput(
            text=text,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            token_logprobs=token_logprobs,
            mean_logprob=mean_logprob,
            top_logprobs_per_token=top_logprobs_per_token,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )

    def _transmit_anthropic(
        self, request: ChatRequest, temp: float, prompt_variant: str, t0: float
    ) -> AgentOutput:
        # The Anthropic SDK accepts thinking={"type": "enabled", "budget_tokens": N}
        # to enable extended thinking. There is no "disabled" value — omitting the
        # parameter is the canonical way to turn thinking off (which is the default
        # for Claude 4.x anyway). Strip the {"type": "disabled"} shim so users can
        # express "explicitly disabled" in YAML without the API rejecting it.
        extra = dict(self.extra_body)
        thinking_cfg = extra.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "disabled":
            extra.pop("thinking", None)
        # Per-request extra_body wins over channel-configured extras.
        if request.extra_body:
            extra = {**extra, **dict(request.extra_body)}

        # Anthropic system is a top-level kwarg, never a message. The
        # request's system message (channel default OR caller-supplied) is
        # hoisted into `system_text` by the serializer.
        system_text, messages_payload = request.to_anthropic_payload()
        if system_text is None:
            system_text = self.system_prompt

        api_kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system_text,
            messages=messages_payload,
            temperature=temp,
            max_tokens=request.max_tokens or self.max_tokens,
        )
        if request.tools:
            # Translate from our OpenAI-shaped tool dicts back to Anthropic's
            # {name, description, input_schema} on the way out.
            api_kwargs["tools"] = [_openai_tool_to_anthropic(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = _normalize_anthropic_tool_choice(request.tool_choice)
        if request.stop:
            api_kwargs["stop_sequences"] = list(request.stop)
        if request.top_p is not None:
            api_kwargs["top_p"] = request.top_p
        api_kwargs.update(extra)

        response = self.anthropic_client.messages.create(**api_kwargs)
        latency = time.time() - t0
        # If thinking were enabled, response.content would contain a ThinkingBlock
        # before the TextBlock — find the first text block defensively. Also
        # extract any tool_use blocks so we can surface them as tool_calls.
        text = ""
        thinking_parts: list[str] = []
        tool_calls_list: list[ToolCall] = []
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                tt = getattr(block, "thinking", None) or getattr(block, "text", "") or ""
                if tt:
                    thinking_parts.append(tt)
            elif btype == "text" and not text:
                text = block.text
            elif btype == "tool_use":
                tool_calls_list.append(ToolCall(
                    id=getattr(block, "id", "") or "",
                    name=getattr(block, "name", "") or "",
                    arguments=json.dumps(getattr(block, "input", None) or {}),
                ))
        if thinking_parts:
            logger.info(
                f"Anthropic model={self.model!r} returned ThinkingBlock(s) "
                f"({sum(len(p) for p in thinking_parts)} chars total); "
                f"captured into thinking_text. To turn extended thinking off, "
                f"omit the `thinking` extra_body."
            )
        inline_thinking = ""
        if _has_thinking(text):
            self._warn_thinking_once(text)
            text, inline_thinking = _split_inline_thinking(text)
        thinking_text = "\n".join(p for p in [*thinking_parts, inline_thinking] if p)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        finish_reason = getattr(response, "stop_reason", None)
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_source=(
                "anthropic_thinking_block" if thinking_parts else
                ("inline_tag_strip" if inline_thinking else None)
            ),
        )

        return AgentOutput(
            text=text,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            tool_calls=tuple(tool_calls_list) if tool_calls_list else None,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )

    # ----------------------------------------------------------------
    # Native async streaming — Phase 2 surface used by api.astream()
    # and per-technique astream() generators (Phase 3).
    # ----------------------------------------------------------------

    async def _atransmit_stream_openai(
        self,
        request: ChatRequest,
        temp: float,
        prompt_variant: str,
        t0: float,
    ) -> AsyncIterator[ChannelChunk | ChannelDone]:
        """Stream from any OpenAI-compatible endpoint via AsyncOpenAI.

        Covers OpenAI, OpenRouter, vLLM/SGLang, DeepSeek, Together, and
        Ollama-via-OpenAI-compat. Captures both ``delta.content`` (answer)
        and ``delta.reasoning_content`` (provider-exposed thinking).
        Inline ``<think>`` tags emitted as part of ``delta.content`` are
        post-processed at stream close — we can't split them on a per-chunk
        basis without buffering across an unknown boundary, so they land in
        ``thinking_text`` of the final ``AgentOutput`` while still appearing
        as ``"answer"`` chunks mid-stream (downstream events can re-tag).
        """
        client = self._get_async_client()
        # Use sync transmit's request-building helpers in reverse: the
        # OpenAI shape is what the SDK already wants.
        openai_messages = request.to_openai_messages()
        api_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=openai_messages,
            temperature=temp,
            max_tokens=request.max_tokens or self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if request.tools:
            api_kwargs["tools"] = [dict(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = request.tool_choice
        if request.stop:
            api_kwargs["stop"] = list(request.stop)
        if request.top_p is not None:
            api_kwargs["top_p"] = request.top_p
        if request.response_format is not None:
            api_kwargs["response_format"] = dict(request.response_format)
        if self.extra_body:
            api_kwargs["extra_body"] = dict(self.extra_body)
        if request.extra_body:
            eb = dict(api_kwargs.get("extra_body") or {})
            eb.update(dict(request.extra_body))
            api_kwargs["extra_body"] = eb

        answer_buf: list[str] = []
        reasoning_buf: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens_exact: int | None = None

        stream = await client.chat.completions.create(**api_kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                prompt_tokens = getattr(u, "prompt_tokens", prompt_tokens) or prompt_tokens
                completion_tokens = (
                    getattr(u, "completion_tokens", completion_tokens) or completion_tokens
                )
                details = getattr(u, "completion_tokens_details", None)
                if details is not None:
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt is not None:
                        reasoning_tokens_exact = int(rt)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                answer_buf.append(content)
                yield ChannelChunk(role="answer", text=content)
            reasoning_delta = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if reasoning_delta:
                reasoning_buf.append(reasoning_delta)
                yield ChannelChunk(role="thinking", text=reasoning_delta)
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0)
                slot = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                func = getattr(tc, "function", None)
                if func is not None:
                    if getattr(func, "name", None):
                        slot["name"] = func.name
                    arg_delta = getattr(func, "arguments", None) or ""
                    if arg_delta:
                        slot["args"] += arg_delta
            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                finish_reason = fr

        latency = time.time() - t0
        raw_answer = "".join(answer_buf)
        reasoning_text = "".join(reasoning_buf)

        # Inline <think> tag stripping for backends that emit them in content.
        inline_thinking = ""
        if _is_thinking_model(self.model):
            if _has_thinking(raw_answer):
                self._warn_thinking_once(raw_answer)
                clean_answer, inline_thinking = _split_inline_thinking(raw_answer)
            else:
                clean_answer = QualityScorer._strip_thinking(raw_answer)
        elif _has_thinking(raw_answer):
            self._warn_thinking_once(raw_answer)
            clean_answer, inline_thinking = _split_inline_thinking(raw_answer)
        else:
            clean_answer = raw_answer

        thinking_text = "\n".join(p for p in [reasoning_text, inline_thinking] if p)
        if reasoning_tokens_exact is not None and reasoning_tokens_exact > 0:
            tokens_source = "openai_reasoning_tokens"
            tokens_exact: int | None = reasoning_tokens_exact
        elif reasoning_text:
            tokens_source = "reasoning_content_field"
            tokens_exact = None
        elif inline_thinking:
            tokens_source = "inline_tag_strip"
            tokens_exact = None
        else:
            tokens_source = None
            tokens_exact = None
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=clean_answer,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            thinking_tokens_exact=tokens_exact,
            thinking_tokens_source=tokens_source,
        )

        tool_calls_tuple: tuple[ToolCall, ...] | None = None
        if tool_acc:
            tool_calls_tuple = tuple(
                ToolCall(
                    id=v["id"] or f"call_{i}",
                    name=v["name"] or "",
                    arguments=v["args"] or "{}",
                )
                for i, v in sorted(tool_acc.items())
            )

        output = AgentOutput(
            text=clean_answer,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=prompt_tokens + completion_tokens,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=_estimate_cost(self.model, prompt_tokens, completion_tokens),
            tool_calls=tool_calls_tuple,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )
        yield ChannelDone(output=output)

    async def _atransmit_stream_anthropic(
        self,
        request: ChatRequest,
        temp: float,
        prompt_variant: str,
        t0: float,
    ) -> AsyncIterator[ChannelChunk | ChannelDone]:
        """Stream from Anthropic via AsyncAnthropic.messages.stream().

        Parses ``content_block_delta`` events with delta types:
        - ``text_delta`` → answer chunks
        - ``thinking_delta`` → thinking chunks
        - ``input_json_delta`` → accumulating tool-call arguments
        """
        client = self._get_async_anthropic_client()

        # Same translation as sync _transmit_anthropic.
        extra = dict(self.extra_body)
        thinking_cfg = extra.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "disabled":
            extra.pop("thinking", None)
        if request.extra_body:
            extra = {**extra, **dict(request.extra_body)}

        system_text, messages_payload = request.to_anthropic_payload()
        if system_text is None:
            system_text = self.system_prompt

        api_kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system_text,
            messages=messages_payload,
            temperature=temp,
            max_tokens=request.max_tokens or self.max_tokens,
        )
        if request.tools:
            api_kwargs["tools"] = [_openai_tool_to_anthropic(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = _normalize_anthropic_tool_choice(request.tool_choice)
        if request.stop:
            api_kwargs["stop_sequences"] = list(request.stop)
        if request.top_p is not None:
            api_kwargs["top_p"] = request.top_p
        api_kwargs.update(extra)

        answer_buf: list[str] = []
        thinking_buf: list[str] = []
        # Per-content-block tool accumulator: block_index -> dict.
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        # Track which content-block index is currently open and its type so
        # input_json_delta events route to the right tool accumulator.
        current_block_type: dict[int, str] = {}

        async with client.messages.stream(**api_kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    usage = getattr(msg, "usage", None) if msg else None
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                elif etype == "content_block_start":
                    idx = getattr(event, "index", 0)
                    block = getattr(event, "content_block", None)
                    btype = getattr(block, "type", None) if block else None
                    if btype:
                        current_block_type[idx] = btype
                    if btype == "tool_use":
                        tool_acc[idx] = {
                            "id": getattr(block, "id", "") or "",
                            "name": getattr(block, "name", "") or "",
                            "args": "",
                        }
                elif etype == "content_block_delta":
                    idx = getattr(event, "index", 0)
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        t = getattr(delta, "text", "") or ""
                        if t:
                            answer_buf.append(t)
                            yield ChannelChunk(role="answer", text=t)
                    elif dtype == "thinking_delta":
                        t = getattr(delta, "thinking", "") or ""
                        if t:
                            thinking_buf.append(t)
                            yield ChannelChunk(role="thinking", text=t)
                    elif dtype == "input_json_delta":
                        slot = tool_acc.get(idx)
                        if slot is not None:
                            slot["args"] += getattr(delta, "partial_json", "") or ""
                elif etype == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        # Cumulative output tokens — overwrite each time.
                        output_tokens = (
                            getattr(usage, "output_tokens", output_tokens) or output_tokens
                        )
                    md = getattr(event, "delta", None)
                    if md is not None:
                        sr = getattr(md, "stop_reason", None)
                        if sr is not None:
                            finish_reason = sr

        latency = time.time() - t0
        raw_answer = "".join(answer_buf)
        thinking_text_native = "".join(thinking_buf)
        inline_thinking = ""
        if _has_thinking(raw_answer):
            self._warn_thinking_once(raw_answer)
            clean_answer, inline_thinking = _split_inline_thinking(raw_answer)
        else:
            clean_answer = raw_answer

        thinking_text = "\n".join(
            p for p in [thinking_text_native, inline_thinking] if p
        )
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=clean_answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_source=(
                "anthropic_thinking_block" if thinking_text_native else
                ("inline_tag_strip" if inline_thinking else None)
            ),
        )

        tool_calls_tuple: tuple[ToolCall, ...] | None = None
        if tool_acc:
            tool_calls_tuple = tuple(
                ToolCall(
                    id=v["id"] or f"call_{i}",
                    name=v["name"] or "",
                    arguments=v["args"] or "{}",
                )
                for i, v in sorted(tool_acc.items())
            )

        output = AgentOutput(
            text=clean_answer,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            tool_calls=tool_calls_tuple,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )
        yield ChannelDone(output=output)

    async def _atransmit_stream_ollama_native(
        self,
        request: ChatRequest,
        temp: float,
        prompt_variant: str,
        t0: float,
    ) -> AsyncIterator[ChannelChunk | ChannelDone]:
        """Stream from Ollama's native /api/chat using httpx.AsyncClient.

        Each line on the response is one JSON object with incremental
        ``message.content`` / ``message.thinking`` deltas and a final
        line with ``done: true`` carrying token counts.
        """
        import httpx

        max_tokens = request.max_tokens or self.max_tokens
        options: dict[str, Any] = {
            "temperature": temp,
            "num_predict": max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = list(request.stop)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request.to_ollama_messages(),
            "think": False,
            "stream": True,
            "options": options,
        }
        if request.tools:
            body["tools"] = [dict(t) for t in request.tools]
        if request.response_format is not None:
            rf = dict(request.response_format)
            body["format"] = "json" if rf.get("type") == "json_object" else rf
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = _ollama_native_url(self.base_url or "")

        answer_buf: list[str] = []
        thinking_buf: list[str] = []
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        tool_calls_list: list[ToolCall] = []

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST", url, json=body, headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    msg = payload.get("message") or {}
                    content = msg.get("content") or ""
                    thinking = msg.get("thinking") or ""
                    if content:
                        answer_buf.append(content)
                        yield ChannelChunk(role="answer", text=content)
                    if thinking:
                        thinking_buf.append(thinking)
                        yield ChannelChunk(role="thinking", text=thinking)
                    raw_tool_calls = msg.get("tool_calls") or []
                    for raw in raw_tool_calls:
                        func = raw.get("function") or {}
                        tool_calls_list.append(ToolCall(
                            id=raw.get("id") or f"call_{len(tool_calls_list)}",
                            name=func.get("name", "") or "",
                            arguments=(
                                json.dumps(func.get("arguments") or {})
                                if not isinstance(func.get("arguments"), str)
                                else func.get("arguments")
                            ),
                        ))
                    if payload.get("done"):
                        input_tokens = int(payload.get("prompt_eval_count") or 0)
                        output_tokens = int(payload.get("eval_count") or 0)
                        dr = payload.get("done_reason")
                        finish_reason = "length" if dr == "length" else (
                            dr or "stop"
                        )

        latency = time.time() - t0
        raw_answer = "".join(answer_buf)
        thinking_text_native = "".join(thinking_buf)
        inline_thinking = ""
        if _has_thinking(raw_answer):
            self._warn_thinking_once(raw_answer)
            clean_answer, inline_thinking = _split_inline_thinking(raw_answer)
        else:
            clean_answer = raw_answer

        thinking_text = "\n".join(
            p for p in [thinking_text_native, inline_thinking] if p
        )
        thinking_kwargs = self._build_thinking_kwargs(
            thinking_text=thinking_text,
            answer_text=clean_answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens_source=(
                "ollama_thinking_field" if thinking_text_native else
                ("inline_tag_strip" if inline_thinking else None)
            ),
        )

        output = AgentOutput(
            text=clean_answer,
            model=self.model,
            temperature=temp,
            prompt_variant=prompt_variant,
            latency_s=latency,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
            tool_calls=tuple(tool_calls_list) if tool_calls_list else None,
            finish_reason=finish_reason,
            **thinking_kwargs,
        )
        yield ChannelDone(output=output)


class QualityScorer:
    """
    Channel estimator — estimates the 'received SNR' of an agent output.

    Uses an LLM judge to score output quality on [0, 1].
    This is analogous to channel estimation / SNR measurement.

    Design: Binary checklist scoring (v3 — sigma-delta)
    ---------------------------------------------------
    Instead of asking the judge for a numeric rating (which causes severe
    quantization — small models like gemma3:12b cluster on "7" for everything),
    we ask for BINARY yes/no answers on 15 weighted criteria.

    This works because:
    1. Binary decisions are MUCH more reliable for small LLMs than numeric ratings
    2. 15 weighted binary checks → 2^15 = 32768 unique score combinations
    3. Each check is concrete and verifiable, not abstract/subjective
    4. Analogous to sigma-delta ADC: many 1-bit measurements combine to
       high-resolution output via oversampling + noise shaping

    See ``CHECKS_WITH_REF`` and ``CHECKS_NO_REF`` below for the canonical
    list, weights, and which checks are marked STRICT.
    """

    # Binary checklist criteria: (label, weight) — weights sum to 1.0
    #
    # 15 checks with varied weights produce 2^15 = 32768 combinations.
    # Criteria are calibrated so a "good but not perfect" response should
    # pass ~10-12 of 15 checks (score 0.65-0.85), leaving room for
    # iterative techniques to improve into the 0.85-1.0 range.
    #
    # The "strict" checks (marked with ★) are deliberately hard to pass —
    # they fail for "good enough" answers, creating score differentiation
    # in the 0.7-0.9 range where technique comparisons happen.
    CHECKS_WITH_REF = [
        # Correctness cluster (35%)
        ("main_conclusion_matches_reference", 0.12),
        ("first_key_detail_from_reference", 0.08),
        ("second_key_detail_from_reference", 0.08),
        ("third_key_detail_from_reference", 0.07),
        # Accuracy cluster (20%) — ★ strict
        ("zero_factual_errors", 0.10),         # ★ strict: ANY error = false
        ("every_specific_fact_verifiable", 0.05),  # ★ no vague/unverifiable claims
        ("no_extra_claims_beyond_reference", 0.05), # ★ penalize hallucinated extras
        # Completeness cluster (20%)
        ("all_parts_of_prompt_answered", 0.07),
        ("explains_why_not_just_states_what", 0.07),  # ★ must have causal reasoning
        ("concrete_example_or_evidence_given", 0.06),  # ★ generic statement = false
        # Reasoning cluster (12%)
        ("each_logical_step_follows", 0.06),
        ("no_internal_contradictions", 0.06),
        # Presentation cluster (13%)
        ("uses_specific_technical_terms", 0.05),  # ★ vague wording = false
        ("structured_with_clear_sections", 0.04),
        ("concise_no_unnecessary_repetition", 0.04),  # ★ repetition = false
    ]
    CHECKS_NO_REF = [
        # Relevance cluster (14%)
        ("directly_answers_the_question", 0.08),
        ("no_off_topic_tangents", 0.06),
        # Correctness cluster (26%) — ★ strict
        ("main_answer_factually_correct", 0.10),
        ("all_supporting_facts_verifiable", 0.08),  # ★ unverifiable = false
        ("zero_hallucinated_details", 0.08),         # ★ any fabrication = false
        # Depth cluster (24%) — ★ strict
        ("covers_multiple_aspects_of_topic", 0.08),
        ("goes_beyond_surface_level", 0.06),         # ★ shallow = false
        ("includes_specific_example", 0.05),          # ★ must have concrete example
        ("explains_mechanisms_not_just_facts", 0.05), # ★ must explain HOW/WHY
        # Reasoning cluster (12%)
        ("reasoning_chain_is_valid", 0.06),
        ("no_logical_contradictions", 0.06),
        # Presentation cluster (16%) — ★ strict
        ("well_structured_with_clear_flow", 0.05),
        ("precise_language_no_vagueness", 0.05),  # ★ "it depends" without detail = false
        ("not_repetitive_or_verbose", 0.04),       # ★ any redundancy = false
        ("actionable_or_educational", 0.02),
    ]

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
        score_strategy: str = "blended",
    ):
        # Scoring strategy for tasks that carry a `score_mode` (deterministic
        # check available). Three values:
        #   "blended" (default) — final_quality = 0.6 * deterministic + 0.4 * judge
        #     on the model's free-form output. Pays the judge call cost but
        #     gives a rich diagnostic score that distinguishes "right answer
        #     for the right reason" from "right answer for the wrong reason".
        #   "exact" — pure deterministic, no judge call. Returns {0, 1}.
        #     Cheapest and noise-free; loses reasoning-quality signal.
        #   "judge" — pure judge, ignores score_mode entirely. The legacy
        #     pre-tiered-scoring path; kept for backward-compat A/B runs.
        # Tasks WITHOUT a score_mode (free-form like HumanEval, MMMU open
        # with multi-alternative answers) always go through the judge
        # regardless of strategy — there's no deterministic check to blend.
        if score_strategy not in ("blended", "exact", "judge"):
            raise ValueError(
                f"score_strategy must be 'blended', 'exact', or 'judge', "
                f"got {score_strategy!r}"
            )
        self.score_strategy = score_strategy
        # Reasoning-model judges (gpt-5, o-series) burn tokens on hidden
        # reasoning that count against max_completion_tokens. With only 768
        # tokens, reasoning often consumes the whole budget and returns an
        # empty message, causing "Failed to parse judge score" warnings.
        # Give them a much larger budget and force reasoning_effort=minimal
        # so almost all tokens go to the visible JSON answer.
        #
        # Local thinking models (qwen3, deepseek-r1 and their distills) have
        # the same failure mode: they emit <think>...</think> preambles that
        # _strip_thinking removes, and at 768 tokens they routinely run out
        # mid-reasoning and return empty text. Give them the large budget too.
        is_openai_reasoning = _is_openai_reasoning_model(judge_model)
        is_local_thinking = _is_thinking_model(judge_model)
        is_reasoning_judge = is_openai_reasoning or is_local_thinking
        max_tokens = 32768 if is_reasoning_judge else 8192
        merged_extra: dict = {}
        if is_openai_reasoning:
            merged_extra["reasoning_effort"] = "minimal"
        if extra_body:
            # User-provided keys win over auto-detected defaults so callers can
            # override (e.g. disable thinking on a qwen3/deepseek-r1 judge via
            # `chat_template_kwargs: {enable_thinking: false}` for Ollama, or
            # `thinking: {type: enabled, budget_tokens: N}` for Anthropic).
            merged_extra.update(extra_body)
        self.judge = AgentChannel(
            model=judge_model,
            temperature=0.1,  # low temp for consistent scoring
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            system_prompt=(
                "You are a quality evaluator. You check AI responses against "
                "specific criteria using yes/no answers.\n"
                "You MUST respond with ONLY a JSON object, nothing else.\n"
                "Do NOT wrap in markdown code blocks.\n"
                "Do NOT add any explanation outside the JSON."
            ),
            extra_body=merged_extra or None,
        )
        # Accumulates all judge LLM call outputs for cost tracking.
        # Call collect_judge_outputs() to retrieve and reset.
        # Stored per-thread so the runner's parallel mode (parallel_tasks>1)
        # keeps judge accounting separated across concurrent (task, repeat)
        # workers — each worker collects only its own judge calls.
        self._judge_outputs_tls = threading.local()

    def score(
        self,
        prompt: str,
        output: str,
        reference: str | None = None,
        criteria: str | None = None,
        _retry: bool = True,
        task: Any = None,
    ) -> float:
        """
        Score an output's quality. Returns float in [0, 1].

        Uses binary checklist scoring: the judge answers 10 yes/no criteria,
        and the weighted sum produces a fine-grained score. This is analogous
        to sigma-delta ADC — many 1-bit measurements combine to high-resolution
        output. Binary decisions are far more reliable for small LLMs than
        numeric ratings.

        If the task has objective_checks, blends LLM judge score with
        objective verification (60% objective, 40% judge) for more reliable
        scoring on tasks with verifiable answers.

        How score_mode + score_strategy combine:
          - task.score_mode is None / "judge": always run pure judge.
          - score_strategy="exact":  score_mode is honored → return {0, 1}
            deterministic check, no judge call.
          - score_strategy="blended" (default): compute the deterministic
            check (binary correctness signal) AND run the judge on the
            free-form output (reasoning-quality signal), then return
            0.6 * det + 0.4 * judge. Two complementary signals folded into
            one continuous score.
          - score_strategy="judge": ignore score_mode entirely (legacy A/B).
        """
        # Compute the deterministic correctness signal up front, if a
        # score_mode is set on the task. We need this whether we end up
        # returning it directly (exact strategy) or blending it with the
        # judge (blended strategy). Pure judge / pure judge-mode tasks
        # skip the deterministic check entirely.
        deterministic_score: float | None = None
        if task is not None and reference is not None:
            mode = getattr(task, "score_mode", None)
            if mode and mode != "judge":
                from . import scoring as _sc
                if mode not in _sc.SUPPORTED_MODES:
                    raise ValueError(
                        f"Unknown score_mode {mode!r} on task {getattr(task, 'id', '?')}. "
                        f"Supported: {sorted(_sc.SUPPORTED_MODES)}."
                    )
                # `metadata` is consulted only by metadata-aware scorers
                # (e.g. "code", which needs the test fixture). Other modes
                # ignore it. Pass the task's metadata dict directly so we
                # don't have to special-case the call site.
                deterministic_score = _sc.score_deterministic(
                    mode, output, reference,
                    metadata=getattr(task, "metadata", None),
                )

        # Mode + extracted-value info for logging. Captured here so all
        # return paths can emit a uniform breakdown line.
        mode = getattr(task, "score_mode", None) if task is not None else None
        extracted: Any = None
        if mode and mode != "judge" and reference is not None:
            from . import scoring as _sc
            if mode == "exact_letter":
                extracted = _sc.extract_letter(output)
            elif mode == "yes_no":
                extracted = _sc.extract_yes_no(output)
            elif mode in ("numeric", "relaxed"):
                extracted = _sc.extract_number(output)

        # Pure-deterministic short-circuit ("exact" strategy or no judge needed).
        if self.score_strategy == "exact" and deterministic_score is not None:
            self._log_score_breakdown(
                task=task, strategy="exact", mode=mode,
                deterministic=deterministic_score, extracted=extracted,
                reference=reference, judge_score=None, checklist_breakdown=None,
                final=deterministic_score, blend_formula="exact (no judge)",
            )
            return deterministic_score

        judge_prompt = self._build_judge_prompt(prompt, output, reference, criteria)
        checks = self.CHECKS_WITH_REF if reference else self.CHECKS_NO_REF

        # Bounded retry loop for the checklist judge call. We cap at
        # MAX_CHECKLIST_ATTEMPTS to prevent unbounded transmits when the
        # judge consistently returns un-parseable output (e.g. a thinking
        # model that exhausts max_tokens during reasoning).
        MAX_CHECKLIST_ATTEMPTS = 3
        judge_score: float | None = None
        checklist_breakdown: dict[str, bool] | None = None
        result = None
        for attempt in range(MAX_CHECKLIST_ATTEMPTS):
            try:
                result = self.judge.transmit(judge_prompt, temperature=0.1)
                self._judge_outputs_for_thread().append(result)
                judge_score, checklist_breakdown = self._parse_checklist_score(result.text, checks)
                if judge_score is not None:
                    # Persist the parsed checklist on the judge output's
                    # metadata so callers running with `return_trace=True`
                    # can render the breakdown (which yes/no criteria passed,
                    # their weights, the weighted sum) without re-running
                    # the judge or scraping the INFO log line.
                    if checklist_breakdown is not None:
                        result.metadata["checklist"] = {
                            "breakdown": dict(checklist_breakdown),
                            "weights": {name: w for name, w in checks},
                            "weighted_score": judge_score,
                            "passed": sum(1 for v in checklist_breakdown.values() if v),
                            "total": len(checklist_breakdown),
                        }
                    break
            except Exception as e:
                logger.info(f"Judge transmit failed on attempt {attempt + 1}: {e}")
            if attempt < MAX_CHECKLIST_ATTEMPTS - 1:
                logger.info("Retrying due to failed score parse.")
                time.sleep(1)

        # Fall back to single-score parsing on the last result
        if judge_score is None and result is not None:
            judge_score = self._parse_score(result.text)

        if judge_score is None and _retry:
            logger.debug(
                f"Retrying score with simpler prompt "
                f"(original: {result.text[:100] if result else '<no result>'})"
            )
            retry_result = self.judge.transmit(
                f"Rate this AI response from 0.0 to 1.0 (0=terrible, 1=perfect). "
                f"Reply with ONLY a number.\n\nTask: {prompt[:200]}\n\nResponse: {output[:500]}",
                temperature=0.0,
            )
            self._judge_outputs_for_thread().append(retry_result)
            judge_score = self._parse_score(retry_result.text)

        if judge_score is None:
            text_preview = (result.text[:200] if result and result.text else "<empty>")
            tokens = result.token_count if result else 0
            logger.warning(
                f"Failed to parse judge score after retry, defaulting to 0.5. "
                f"model={self.judge.model} tokens={tokens} text={text_preview!r}"
            )
            sys.exit()
            # judge_score = 0.5

        # Blend with the deterministic correctness signal when:
        #   - score_strategy=="blended" AND task has a score_mode → preferred path
        # The deterministic check is treated as ground truth on correctness;
        # the judge contributes a reasoning-quality assessment of the prose.
        # 0.6 / 0.4 weighting follows the codebase's existing convention for
        # blending verified facts with judge opinions ("facts are more
        # reliable than the LLM judge").
        if (
            self.score_strategy == "blended"
            and deterministic_score is not None
        ):
            blended = 0.6 * deterministic_score + 0.4 * judge_score
            self._log_score_breakdown(
                task=task, strategy="blended", mode=mode,
                deterministic=deterministic_score, extracted=extracted,
                reference=reference, judge_score=judge_score,
                checklist_breakdown=checklist_breakdown,
                final=blended,
                blend_formula=f"0.6×{deterministic_score:.3f} + 0.4×{judge_score:.3f}",
            )
            return blended

        # Legacy `objective_checks` path: tasks that opted into the older
        # regex-based deterministic check (not the score_mode system) still
        # get the 0.6/0.4 blend. Skipped when score_strategy=="judge".
        if task is not None and self.score_strategy != "judge":
            obj_score = task.verify_objective(output)
            if obj_score is not None:
                blended = 0.6 * obj_score + 0.4 * judge_score
                self._log_score_breakdown(
                    task=task, strategy="blended-legacy", mode=None,
                    deterministic=obj_score, extracted="objective_checks",
                    reference=reference, judge_score=judge_score,
                    checklist_breakdown=checklist_breakdown,
                    final=blended,
                    blend_formula=f"0.6×{obj_score:.3f} + 0.4×{judge_score:.3f}",
                )
                return blended

        # Pure-judge return path (free-form tasks, or score_strategy=="judge")
        self._log_score_breakdown(
            task=task,
            strategy=self.score_strategy,
            mode=mode,
            deterministic=deterministic_score,
            extracted=extracted,
            reference=reference,
            judge_score=judge_score,
            checklist_breakdown=checklist_breakdown,
            final=judge_score,
            blend_formula="judge-only",
        )

        return judge_score

    def _log_score_breakdown(
        self,
        task: Any,
        strategy: str,
        mode: str | None,
        deterministic: float | None,
        extracted: Any,
        reference: str | None,
        judge_score: float | None,
        checklist_breakdown: dict[str, bool] | None,
        final: float,
        blend_formula: str,
    ) -> None:
        """
        Emit a per-task score-breakdown log line so users can see how the
        final score was assembled. Goes to INFO so it shows up in the
        default benchmark logs.

        Format (one or two lines per task):
            [SCORE <task_id>] strategy=<S> mode=<M>
              det=<D> [extracted='X' ref='Y']  judge=<J> [P/N checks]  final=<F> (<blend>)

        The checklist breakdown (which criteria passed/failed) is logged at
        DEBUG to keep the default output tight.
        """
        task_id = getattr(task, "id", "?") if task is not None else "?"
        # Reference preview — truncate long references so the line stays readable.
        ref_preview: str
        if reference is None:
            ref_preview = "<none>"
        else:
            r = str(reference)
            ref_preview = (r[:30] + "…") if len(r) > 30 else r

        # Per-component renders
        det_str = f"{deterministic:.3f}" if deterministic is not None else "—"
        judge_str = f"{judge_score:.3f}" if judge_score is not None else "—"

        # Extracted-value render
        if extracted is None:
            ext_str = "<unparseable>"
        elif isinstance(extracted, bool):
            ext_str = "Yes" if extracted else "No"
        else:
            ext_str = repr(str(extracted)[:30])

        det_detail = ""
        if deterministic is not None and mode:
            det_detail = f" [extracted={ext_str} ref={ref_preview!r}]"

        # Judge detail: pass/fail count + score
        judge_detail = ""
        if judge_score is not None and checklist_breakdown is not None:
            passed = sum(1 for v in checklist_breakdown.values() if v)
            total = len(checklist_breakdown)
            judge_detail = f" [{passed}/{total} checks]"

        logger.info(
            f"[SCORE {task_id}] strategy={strategy} mode={mode or '—'}  "
            f"det={det_str}{det_detail}  judge={judge_str}{judge_detail}  "
            f"final={final:.3f} ({blend_formula})"
        )

        # Per-check breakdown at DEBUG level so heavy users can grep it
        # without drowning the default log.
        if checklist_breakdown and logger.isEnabledFor(logging.DEBUG):
            failed = [k for k, v in checklist_breakdown.items() if not v]
            passed = [k for k, v in checklist_breakdown.items() if v]
            logger.debug(
                f"[SCORE {task_id}] checks passed: {passed}; "
                f"failed: {failed}"
            )

    def _judge_outputs_for_thread(self) -> list[AgentOutput]:
        """Per-thread accumulator for judge call outputs."""
        lst = getattr(self._judge_outputs_tls, "outputs", None)
        if lst is None:
            lst = []
            self._judge_outputs_tls.outputs = lst
        return lst

    def collect_judge_outputs(self) -> list[AgentOutput]:
        """Retrieve and reset this thread's accumulated judge call outputs.

        Per-thread storage means the runner's parallel mode collects judge
        calls scoped to the current (task, repeat) worker — not a global
        soup mixed across concurrent tasks.
        """
        outputs = self._judge_outputs_for_thread()
        self._judge_outputs_tls.outputs = []
        return outputs

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove all thinking/reasoning blocks from model output."""
        # Remove closed thinking blocks (various tag names)
        text = re.sub(
            _THINKING_OPEN + r".*?" + _THINKING_CLOSE,
            "", text, flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove unclosed thinking blocks (model hit max_tokens while thinking)
        text = re.sub(
            _THINKING_OPEN + r".*",
            "", text, flags=re.DOTALL | re.IGNORECASE,
        )
        return text.strip()

    def _parse_score(self, text: str) -> float | None:
        """Extract a score from judge output. Returns None if no score found."""
        import json
        import re

        raw_text = text  # keep original for fallback search

        # Strip thinking blocks first
        text = self._strip_thinking(text)

        # If stripping left nothing, fall back to searching the raw text
        if not text:
            text = raw_text

        # Try 1: Direct JSON parse
        try:
            parsed = json.loads(text)
            score = parsed.get("score")
            if score is not None:
                return max(0.0, min(1.0, float(score)))
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Try 2: Extract JSON from markdown code blocks
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            try:
                parsed = json.loads(code_block.group(1))
                score = parsed.get("score")
                if score is not None:
                    return max(0.0, min(1.0, float(score)))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try 3: Find any JSON object containing "score" anywhere in full text
        for search_text in [text, raw_text]:
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*[0-9.]+[^{}]*\}', search_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    score = parsed.get("score")
                    if score is not None:
                        return max(0.0, min(1.0, float(score)))
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try 4: Find "score": <number> or score: <number> pattern
        for search_text in [text, raw_text]:
            score_match = re.search(
                r"[\"']?score[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)", search_text, re.IGNORECASE
            )
            if score_match:
                try:
                    return max(0.0, min(1.0, float(score_match.group(1))))
                except ValueError:
                    pass

        # Try 5: Handle "X/10", "X out of 10", "X/5", percentage formats
        for search_text in [text, raw_text]:
            # "7/10", "8.5/10"
            frac_match = re.search(r"(\d+\.?\d*)\s*/\s*(\d+)", search_text)
            if frac_match:
                try:
                    num, denom = float(frac_match.group(1)), float(frac_match.group(2))
                    if denom > 0:
                        return max(0.0, min(1.0, num / denom))
                except ValueError:
                    pass

            # "7 out of 10"
            out_of_match = re.search(r"(\d+\.?\d*)\s+out\s+of\s+(\d+)", search_text, re.IGNORECASE)
            if out_of_match:
                try:
                    num, denom = float(out_of_match.group(1)), float(out_of_match.group(2))
                    if denom > 0:
                        return max(0.0, min(1.0, num / denom))
                except ValueError:
                    pass

            # "85%" or "85 percent"
            pct_match = re.search(r"(\d+\.?\d*)\s*(?:%|percent)", search_text, re.IGNORECASE)
            if pct_match:
                try:
                    return max(0.0, min(1.0, float(pct_match.group(1)) / 100.0))
                except ValueError:
                    pass

        # Try 6: Find a standalone decimal between 0 and 1
        for search_text in [text, raw_text]:
            decimal_match = re.search(r"\b(0\.\d+|1\.0)\b", search_text)
            if decimal_match:
                try:
                    return float(decimal_match.group(1))
                except ValueError:
                    pass

        # Try 7: Bare single digit 0-9 (interpret as X/10 scale)
        bare_digit = re.search(r"^\s*(\d)\s*$", text)
        if bare_digit:
            return int(bare_digit.group(1)) / 10.0

        return None

    def score_comparative(
        self,
        prompt: str,
        candidate: str,
        baseline: str,
        baseline_score: float,
        reference: str | None = None,
        task: Any = None,
    ) -> float:
        """
        Score a candidate relative to a baseline using differential scoring.

        Instead of scoring the candidate independently (which introduces
        uncorrelated noise between iterations), we score BOTH candidate and
        baseline with the SAME judge call. This is analogous to differential
        decoding in communications: by measuring the DIFFERENCE, common-mode
        noise (judge bias, prompt sensitivity) cancels out.

        The key insight: independent scoring of candidate and baseline produces
        two noisy measurements with uncorrelated errors. In iterative techniques
        like turbo/HARQ-IR, the candidate differs from the baseline by only a
        few corrections — the absolute score noise (~0.05-0.10) is often larger
        than the actual quality delta (~0.02-0.05). Differential scoring
        measures the delta directly, with much lower noise.

        Returns a score for the candidate on [0, 1].
        """
        # Score the candidate independently first
        candidate_score = self.score(prompt, candidate, reference=reference, _retry=True, task=task)

        # If no baseline provided, return independent score
        if not baseline or baseline_score is None:
            return candidate_score

        # Differential correction: score the baseline with the same judge call
        # pattern to measure the systematic bias, then adjust.
        # This reduces noise by cancelling common-mode judge variance.
        # We score the baseline again (same judge, same prompt) and compute
        # the delta relative to the ORIGINAL baseline_score.
        baseline_rescore = self.score(prompt, baseline, reference=reference, _retry=True, task=task)

        # The differential delta removes common-mode judge noise:
        # If the judge is "harsh today" (both scores lower), the delta is preserved.
        # If the judge is "lenient today" (both scores higher), the delta is preserved.
        delta = candidate_score - baseline_rescore

        # Apply the delta to the original baseline_score
        adjusted = baseline_score + delta
        return max(0.0, min(1.0, adjusted))

    def score_batch(self, prompt: str, outputs: list[AgentOutput], reference: str | None = None, task: Any = None) -> list[AgentOutput]:
        """Score a batch of outputs and set their quality_score fields."""
        for output in outputs:
            output.quality_score = self.score(prompt, output.text, reference, task=task)
        return outputs

    def _parse_checklist_score(
        self, text: str, checks: list[tuple[str, float]]
    ) -> tuple[float | None, dict[str, bool] | None]:
        """
        Parse a binary checklist response and compute weighted score.

        Expected format: {"main_conclusion_correct": true, "key_detail_1_present": false, ...}
        Each criterion is true/false. Score = weighted sum of true values.

        Returns (score, breakdown) where:
            score:     weighted score in [0, 1], or None if parsing failed
            breakdown: {check_name: passed_bool}, or None if parsing failed
        The breakdown lets callers log per-check pass/fail without re-parsing.
        """
        import json

        text = self._strip_thinking(text)
        if not text:
            return None, None

        check_names = {name for name, _ in checks}

        for candidate in self._extract_json_candidates(text):
            try:
                data = json.loads(candidate)
                if not isinstance(data, dict):
                    continue

                # Check if we have at least 3 checklist criteria
                found = {}
                for k, v in data.items():
                    if k in check_names:
                        if isinstance(v, bool):
                            found[k] = v
                        elif isinstance(v, str):
                            found[k] = v.lower() in ("true", "yes", "1")
                        elif isinstance(v, (int, float)):
                            found[k] = bool(v)

                if len(found) < 3:
                    continue

                # Compute weighted score
                total_weight = 0.0
                weighted_sum = 0.0
                for name, weight in checks:
                    if name in found:
                        weighted_sum += weight * (1.0 if found[name] else 0.0)
                        total_weight += weight

                if total_weight > 0:
                    score = weighted_sum / total_weight
                    passed = sum(1 for v in found.values() if v)
                    logger.debug(
                        f"Checklist: {passed}/{len(found)} passed → {score:.4f}"
                    )
                    return max(0.0, min(1.0, score)), found
            except (json.JSONDecodeError, ValueError):
                continue

        return None, None

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        """Extract potential JSON strings from text (various formats)."""
        import re
        candidates = []
        # Raw text as-is
        candidates.append(text.strip())
        # Strip outer backticks
        candidates.append(text.strip().strip("`").strip())
        # Code blocks
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            candidates.append(m.group(1))
        # Any {...} block
        for m in re.finditer(r"\{[^{}]*\}", text):
            candidates.append(m.group(0))
        return candidates

    @staticmethod
    def _check_descriptions(has_reference: bool) -> dict[str, str]:
        """Human-readable descriptions for each checklist criterion."""
        if has_reference:
            return {
                "main_conclusion_matches_reference": "The main conclusion/answer matches the reference answer",
                "first_key_detail_from_reference": "The FIRST key supporting detail from the reference is clearly present",
                "second_key_detail_from_reference": "The SECOND key supporting detail from the reference is clearly present",
                "third_key_detail_from_reference": "A THIRD key detail from the reference is present (false if reference has 3+ details and this one is missing)",
                "zero_factual_errors": "STRICT: There are absolutely NO factual errors — even one wrong fact means false",
                "every_specific_fact_verifiable": "STRICT: Every specific claim is verifiable and precise — vague claims like 'it can be used in many ways' count as false",
                "no_extra_claims_beyond_reference": "STRICT: No significant claims are added that go beyond or contradict the reference",
                "all_parts_of_prompt_answered": "Every part/sub-question in the prompt is addressed",
                "explains_why_not_just_states_what": "STRICT: Explains WHY or HOW things work, not just states WHAT — a bare list of facts without explanation = false",
                "concrete_example_or_evidence_given": "STRICT: At least one concrete, specific example is given — generic statements = false",
                "each_logical_step_follows": "Each reasoning step follows logically from the previous one",
                "no_internal_contradictions": "No internal contradictions within the response",
                "uses_specific_technical_terms": "STRICT: Uses precise technical terminology where appropriate — vague wording where specific terms exist = false",
                "structured_with_clear_sections": "Response has clear structure (paragraphs, lists, or logical sections)",
                "concise_no_unnecessary_repetition": "STRICT: No unnecessary repetition — if the same point is made twice in different words = false",
            }
        return {
            "directly_answers_the_question": "Directly and clearly answers the question asked",
            "no_off_topic_tangents": "Stays on topic without irrelevant tangents",
            "main_answer_factually_correct": "The main answer/claim is factually correct",
            "all_supporting_facts_verifiable": "STRICT: Every supporting fact is specific and verifiable — vague generalities = false",
            "zero_hallucinated_details": "STRICT: Absolutely NO made-up facts, dates, names, or statistics",
            "covers_multiple_aspects_of_topic": "Covers multiple distinct aspects or dimensions of the topic",
            "goes_beyond_surface_level": "STRICT: Goes beyond what anyone already knows — surface-level restatement of the question = false",
            "includes_specific_example": "STRICT: Includes at least one SPECIFIC, concrete example — abstract generalization = false",
            "explains_mechanisms_not_just_facts": "STRICT: Explains HOW or WHY things work, not just lists facts",
            "reasoning_chain_is_valid": "The reasoning chain is logically valid",
            "no_logical_contradictions": "No internal contradictions or logical errors",
            "well_structured_with_clear_flow": "Well-structured with logical flow between ideas",
            "precise_language_no_vagueness": "STRICT: Uses precise language — 'it depends', 'various factors', 'in some cases' without specifics = false",
            "not_repetitive_or_verbose": "STRICT: Not repetitive or verbose — same idea in different words = false",
            "actionable_or_educational": "Reader would learn something specific from this response",
        }

    def _build_judge_prompt(
        self, prompt: str, output: str, reference: str | None, criteria: str | None
    ) -> str:
        """
        Build a binary checklist scoring prompt.

        Instead of asking for numeric ratings (which small models quantize
        heavily), asks the judge to evaluate 10 concrete yes/no criteria.
        Binary decisions are far more reliable for small LLMs.
        """
        checks = self.CHECKS_WITH_REF if reference else self.CHECKS_NO_REF
        descs = self._check_descriptions(reference is not None)

        parts = [
            f"## Task Prompt\n{prompt}",
            f"\n## AI Response\n{output}",
        ]
        if reference:
            parts.append(f"\n## Correct/Reference Answer\n{reference}")
        if criteria:
            parts.append(f"\n## Additional Criteria\n{criteria}")

        parts.append(
            "\n## Evaluation Checklist\n"
            "For each criterion below, answer true (passes) or false (fails).\n"
            "Be STRICT and CRITICAL. Criteria marked STRICT have high bars:\n"
            "- A good response should pass about 10-12 of 15 checks, NOT all 15\n"
            "- Only a truly exceptional response passes all 15\n"
            "- When in doubt on a STRICT criterion, mark false\n\n"
        )

        for name, _ in checks:
            desc = descs.get(name, name)
            parts.append(f"- **{name}**: {desc}\n")

        # Build example JSON
        check_names = [name for name, _ in checks]
        example_json = "{" + ", ".join(f'"{n}": true' for n in check_names) + "}"

        parts.append(
            "\n## OUTPUT FORMAT\n"
            f"Respond with ONLY this JSON (no other text):\n"
            f"{example_json}\n\n"
            "Rules:\n"
            "- Each value MUST be true or false (boolean, not string)\n"
            "- Evaluate each criterion INDEPENDENTLY — don't just mark all true or all false\n"
            "- Be honest: a response can be correct but poorly organized, or well-written but incomplete\n"
            "- Do NOT wrap in markdown code blocks\n"
            "- Do NOT add any text before or after the JSON"
        )
        return "\n".join(parts)
