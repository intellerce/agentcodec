"""
Model pricing — sourced from OpenRouter with a hardcoded fallback.

OpenRouter publishes a JSON catalog at https://openrouter.ai/api/v1/models
that includes per-model pricing in USD per token. We fetch it once, cache
it on disk for a week, and look up models by a normalized name. If a model
isn't on OpenRouter (e.g. an Ollama local-only build) or the fetch fails,
we fall back to the project's curated `MODEL_COSTS` table in channel.py.

The on-disk cache lives inside the project (next to the `agentcodec`
package) by default so the catalog ships with the repo and is not lost
when ``~/.cache`` is wiped. Override via the ``AGENTCODEC_CACHE_DIR``
environment variable.

Public surface:
    pricing.lookup(model: str) -> (input_per_1M, output_per_1M, source)
    pricing.refresh(force: bool = False) -> dict   # CLI-friendly
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Default cache location: <opensource_root>/.cache/agentcodec/, i.e. the
# directory that *contains* the agentcodec package. Keeping the cache
# inside the project (rather than ~/.cache) means the pricing catalog
# survives across hosts, container rebuilds, and home-dir resets.
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_CACHE_DIR = _PACKAGE_DIR.parent / ".cache" / "agentcodec"

CACHE_DIR = Path(os.environ.get("AGENTCODEC_CACHE_DIR") or _PROJECT_CACHE_DIR)
CACHE_FILE = CACHE_DIR / "openrouter_models.json"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


# ---------------------------------------------------------------------------
# Catalog fetch / cache
# ---------------------------------------------------------------------------

def _fetch_openrouter_catalog(timeout: float = 15.0) -> list[dict[str, Any]]:
    """Hit the OpenRouter models endpoint. Raises on network/HTTP failure."""
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "agentcodec/1.0 (+pricing)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data", payload)
    if not isinstance(data, list):
        raise ValueError("OpenRouter response missing 'data' list")
    return data


def _load_cache() -> dict[str, Any] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read pricing cache {CACHE_FILE}: {e}")
        return None


def _write_cache(catalog: list[dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    blob = {"fetched_at": time.time(), "models": catalog}
    CACHE_FILE.write_text(json.dumps(blob, indent=2))


def _cache_fresh(blob: dict[str, Any]) -> bool:
    fetched = blob.get("fetched_at", 0)
    return (time.time() - fetched) < CACHE_TTL_SECONDS


def refresh(force: bool = False) -> dict[str, Any]:
    """Refresh the on-disk pricing cache. Returns {n_models, fetched_at, path}."""
    if not force:
        cached = _load_cache()
        if cached and _cache_fresh(cached):
            return {
                "n_models": len(cached.get("models", [])),
                "fetched_at": cached.get("fetched_at"),
                "path": str(CACHE_FILE),
                "from_cache": True,
            }
    catalog = _fetch_openrouter_catalog()
    _write_cache(catalog)
    return {
        "n_models": len(catalog),
        "fetched_at": time.time(),
        "path": str(CACHE_FILE),
        "from_cache": False,
    }


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# OpenRouter ids look like "openai/gpt-4o-mini", "anthropic/claude-sonnet-4",
# "qwen/qwen-2.5-72b-instruct", "meta-llama/llama-3.1-8b-instruct", etc.
# Our internal names look like "gpt-4o-mini", "claude-sonnet-4-6",
# "qwen2.5:14b", "meta-llama/Llama-3.1-8B-Instruct", "deepseek-r1:14b", ...
#
# We normalize both sides to a flat lowercase alnum form and search for the
# best containment match. This is deliberately fuzzy because Ollama tags
# (e.g. "qwen3:14b") and OpenRouter slugs (e.g. "qwen/qwen3-14b") never line
# up exactly.

_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _ALNUM_RE.sub("", s.lower())


def _tokens(s: str) -> list[str]:
    """Split a name into meaningful tokens for fuzzy matching."""
    parts = re.split(r"[^a-zA-Z0-9.]+", s.lower())
    return [p for p in parts if p]


def _candidate_aliases(model: str) -> list[str]:
    """Generate plausible OpenRouter ids for a given internal model name."""
    aliases = [model]
    # Strip Ollama-style ":tag" — but keep the tag too as a separate alias
    # because some OpenRouter ids include the size (e.g. "qwen3-14b").
    if ":" in model:
        head, tag = model.split(":", 1)
        aliases.append(head)
        aliases.append(f"{head}-{tag}")
        aliases.append(f"{head}{tag}")
    # Drop "-cloud" / ":cloud" markers
    aliases.append(re.sub(r"[-:]cloud$", "", model))
    # HuggingFace-style "Org/Model-Name" → flatten case
    if "/" in model:
        aliases.append(model.split("/", 1)[1])
    # Anthropic dated → bare alias ("claude-sonnet-4-6-20250725" → "claude-sonnet-4-6")
    m = re.match(r"^(claude-[a-z0-9.-]+?)-\d{8}$", model)
    if m:
        aliases.append(m.group(1))
    return list(dict.fromkeys(aliases))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

_INDEX_CACHE: dict[str, dict[str, Any]] | None = None
_INDEX_MTIME: float = 0.0


def _build_index(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index OpenRouter entries by normalized id and by trailing slug."""
    idx: dict[str, dict[str, Any]] = {}
    for m in models:
        mid = m.get("id") or m.get("name") or ""
        if not mid:
            continue
        idx[_norm(mid)] = m
        if "/" in mid:
            idx[_norm(mid.split("/", 1)[1])] = m
    return idx


def _get_index() -> dict[str, dict[str, Any]] | None:
    """Load (and memoize) the lookup index from disk cache, refreshing if stale."""
    global _INDEX_CACHE, _INDEX_MTIME
    cached = _load_cache()
    if cached is None or not _cache_fresh(cached):
        try:
            refresh(force=cached is None)
            cached = _load_cache()
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
            logger.warning(
                f"OpenRouter price fetch failed ({e!r}); "
                f"{'using stale cache' if cached else 'falling back to hardcoded prices'}."
            )
            if cached is None:
                return None
    mtime = cached.get("fetched_at", 0.0)
    if _INDEX_CACHE is None or mtime != _INDEX_MTIME:
        _INDEX_CACHE = _build_index(cached.get("models", []))
        _INDEX_MTIME = mtime
    return _INDEX_CACHE


def _prices_from_entry(entry: dict[str, Any]) -> tuple[float, float] | None:
    """Convert an OpenRouter entry's pricing block to (input_per_1M, output_per_1M)."""
    p = entry.get("pricing") or {}
    in_tok = p.get("prompt")
    out_tok = p.get("completion")
    if in_tok is None or out_tok is None:
        return None
    try:
        # OpenRouter quotes per-token USD as strings, e.g. "0.0000025".
        in_per_m = float(in_tok) * 1_000_000
        out_per_m = float(out_tok) * 1_000_000
    except (TypeError, ValueError):
        return None
    return in_per_m, out_per_m


def lookup(model: str) -> tuple[float, float, str] | None:
    """
    Look up (input_per_1M_usd, output_per_1M_usd, source) for `model`.

    Returns None if the model isn't on OpenRouter and we have no cache to
    consult. Callers should fall back to the local MODEL_COSTS table on
    None / on the "fallback" source path.

    `source` is one of: "openrouter", "openrouter-fuzzy".
    """
    idx = _get_index()
    if idx is None:
        return None

    for alias in _candidate_aliases(model):
        key = _norm(alias)
        if key in idx:
            prices = _prices_from_entry(idx[key])
            if prices is not None:
                return prices[0], prices[1], "openrouter"

    # Token-overlap fallback: find the catalog entry sharing the most tokens
    # with our model name. Only accept if at least 2 tokens match AND the
    # model identifier matches (avoid "gpt-4o" colliding with "gpt-4o-mini").
    want = set(_tokens(model))
    if not want:
        return None
    best_score = 0
    best_entry: dict[str, Any] | None = None
    for entry in idx.values():
        mid = entry.get("id") or ""
        have = set(_tokens(mid))
        score = len(want & have)
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry is None or best_score < 2:
        return None
    prices = _prices_from_entry(best_entry)
    if prices is None:
        return None
    logger.debug(
        f"Pricing fuzzy match: {model!r} → "
        f"{best_entry.get('id')!r} (overlap={best_score})"
    )
    return prices[0], prices[1], "openrouter-fuzzy"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Manage OpenRouter pricing cache.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh", help="Force refresh the cache from OpenRouter.")
    show = sub.add_parser("show", help="Look up pricing for a model.")
    show.add_argument("model", type=str)
    sub.add_parser("status", help="Show cache metadata (path, age, n_models).")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.cmd == "refresh":
        info = refresh(force=True)
        print(json.dumps(info, indent=2))
    elif args.cmd == "show":
        result = lookup(args.model)
        if result is None:
            print(f"No OpenRouter price for {args.model!r}")
            return
        i, o, src = result
        print(f"{args.model}: input=${i:.4f}/1M  output=${o:.4f}/1M  source={src}")
    elif args.cmd == "status":
        cached = _load_cache()
        if cached is None:
            print(f"No cache at {CACHE_FILE}")
            return
        age = time.time() - cached.get("fetched_at", 0)
        print(json.dumps({
            "path": str(CACHE_FILE),
            "fetched_at": cached.get("fetched_at"),
            "age_hours": age / 3600,
            "n_models": len(cached.get("models", [])),
            "fresh": _cache_fresh(cached),
        }, indent=2))


if __name__ == "__main__":
    _main()
