"""
RemoteSemKNNRouter — talks to a SemKNN backend over HTTP.

The trained SemKNN cache (q-matrix + training embeddings) is not redistributed
with the public package. Instead, the client:

  1. Encodes the user's prompt locally with the same BGE model the backend
     was trained against (default ``BAAI/bge-small-en-v1.5``; use
     ``BAAI/bge-large-en-v1.5`` for slightly better routing quality at the
     cost of ~3x slower encoding and a 1.3 GB download).
  2. POSTs ``{embedding, lambda, k, user_config, strict_match}`` to ``/route``.
  3. Maps the response into a :class:`RouterDecision`.

The server picks the closest trained *profile* (set of channels + temperatures
the SemKNN was trained against), masks techniques the user can't structurally
run (e.g. ``fountain`` with only one channel), and returns a recommendation
with an ``estimate`` flag when the match isn't exact.

The user's prompt never leaves the client process. The embedding is a unit
vector (384-d for bge-small, 1024-d for bge-large) — a lossy compression of
the input that is hard to invert to readable text in practice (research-grade
attacks like vec2text can partially recover topic-level content from higher-
dimensional embeddings; verbatim recovery from a single unit-norm vector is
unreliable, and the smaller bge-small default reduces this further).

See PLAN.md for the full design rationale.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import httpx

from ..models import TaskItem
from .base import Router, RouterDecision

logger = logging.getLogger(__name__)


# bge-small (384-d, ~130 MB) is the default for fast first-run / low-RAM use.
# bge-large-en-v1.5 (1024-d, ~1.3 GB) gives slightly better routing quality at
# ~3x slower encoding — pass it explicitly when the backend was trained with it.
DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Model-family canonicalization
# ---------------------------------------------------------------------------

_ALIAS_CACHE: dict[str, dict] = {}


def _load_alias_table() -> dict:
    """Load (and cache) the shipped model_aliases.json from package data.

    The same JSON is shipped on the server side; clients and server must
    canonicalize model names identically for Jaccard scoring to be sound.
    """
    key = "default"
    cached = _ALIAS_CACHE.get(key)
    if cached is not None:
        return cached
    path = Path(__file__).with_name("model_aliases.json")
    with open(path) as f:
        table = json.load(f)
    _ALIAS_CACHE[key] = table
    return table


def canonical_family(model_name: str, alias_table: dict | None = None) -> str:
    """Map a concrete model identifier to its canonical family key.

    Lowercased substring match; first family whose any-pattern matches wins.
    Unknown models pass through as the lowercased original string.
    """
    if not model_name:
        return ""
    table = alias_table or _load_alias_table()
    name_lower = model_name.lower()
    families = table.get("families", {})
    for family, spec in families.items():
        patterns = spec.get("patterns", []) if isinstance(spec, dict) else []
        for pat in patterns:
            if pat.lower() in name_lower:
                return family
    return name_lower


# Quantization markers we recognise in a model id (lowercased). Best-effort:
# absence means "unknown / provider default", NOT "full precision".
_QUANT_RE = re.compile(
    r"\b(q\d(?:_[0-9a-z]+)*|fp16|bf16|fp8|fp4|int8|int4|awq|gptq|gguf)\b"
)


def parse_params_b(model_name: str) -> float | None:
    """Best-effort parameter count, in billions, parsed from a model id.

    Recognises size tags like ``:8b``, ``-70B``, ``_3b``, decimals
    (``qwen3:0.6b`` → 0.6), and MoE ``8x7b`` (→ experts × size = 56).
    Returns ``None`` when the name carries no size hint — e.g. closed
    models (``gpt-4o``, ``claude-sonnet-4``) or ``*-cloud`` tags without a
    number. Callers treat ``None`` as "size unknown".

    Whole numbers come back as ``int`` (``30``), fractional as ``float``
    (``0.6``), so the JSON fingerprint stays clean.
    """
    if not model_name:
        return None
    name = model_name.lower()
    # MoE: total active scale ≈ experts × per-expert size (matches the
    # heuristic already used for cost inference in channel.py).
    m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*b\b", name)
    if m:
        val = float(m.group(1)) * float(m.group(2))
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
        if not m:
            return None
        val = float(m.group(1))
    return int(val) if val.is_integer() else val


def parse_quant(model_name: str) -> str | None:
    """Best-effort quantization tag parsed from a model id (or ``None``).

    Quant level materially changes realised quality (a 70B at q2 behaves
    far below a 70B at fp16), so it rides alongside ``params_b`` in the
    fingerprint. Absence is reported as ``None`` (unknown), never assumed.
    """
    if not model_name:
        return None
    m = _QUANT_RE.search(model_name.lower())
    return m.group(1) if m else None


def _derive_user_config(config) -> dict[str, Any]:
    """Build the small fingerprint the backend uses to pick the right profile.

    Reads from a :class:`LibraryConfig`. Imported lazily to avoid a circular
    import between routing/ and config.py at package init time.

    ``channel_specs`` carries per-channel size so the backend can match (and
    the q-matrix can be conditioned) at *model-size* granularity, not just
    family — a profile trained on ``nemotron:30b`` is NOT interchangeable
    with ``nemotron:70b`` even though both canonicalize to ``nemotron``.
    It is additive: ``model_families`` is unchanged for back-compat, and a
    backend that doesn't yet read ``channel_specs`` keeps working.
    """
    families: list[str] = []
    seen: set[str] = set()
    channel_specs: list[dict[str, Any]] = []
    for m in config.models:
        fam = canonical_family(m.model)
        if fam and fam not in seen:
            families.append(fam)
            seen.add(fam)
        # Per-channel (not deduped; order = config order), so the backend
        # sees the exact pool — two nemotron channels of different sizes
        # remain distinguishable.
        channel_specs.append({
            "family": fam,
            "params_b": parse_params_b(m.model),
            "quant": parse_quant(m.model),
        })

    # Union of per-model category_temperatures; primary model wins on conflict.
    cat_temps: dict[str, float] = {}
    for m in config.models:
        if m.category_temperatures:
            for cat, t in m.category_temperatures.items():
                cat_temps.setdefault(cat, float(t))

    has_separate_critic = bool(
        getattr(config, "critic", None) is not None
        and not getattr(config.critic, "same", True)
    )

    return {
        "model_families": families,
        "channel_specs": channel_specs,
        "n_distinct_channels": len({m.model for m in config.models}),
        "primary_temperature": float(config.models[0].temperature),
        "category_temperatures": cat_temps,
        "has_separate_critic": has_separate_critic,
    }


# ---------------------------------------------------------------------------
# BGE encoder caching
# ---------------------------------------------------------------------------
#
# Two backends with the same `encode(text) -> list[float]` shape:
#
#   1. **fastembed (default)** — ONNX-runtime, ~50 MB install, no torch.
#      Ships bge-small-en-v1.5 + bge-large-en-v1.5 out of the box. This is
#      a CORE dependency: telemetry needs an embedding per event, and the
#      retraining loop refuses to be useful without one, so the default
#      install must include a working encoder.
#
#   2. **sentence-transformers (override)** — installed only via the
#      `[remote-semknn]` extra. ~1.5 GB with torch. Use when you need a
#      model fastembed doesn't ship, or you already have torch on the box.
#
# Both backends produce a unit-norm vector at the same dim for the same
# canonical name; the SemKNN backend doesn't care which backend produced
# the vector as long as the model id matches the artifact.

_ENCODER_CACHE: dict[str, Any] = {}
_ENCODER_LOCK = threading.Lock()


class _UnifiedEncoder:
    """Wraps either fastembed or sentence-transformers behind a single
    ``encode(text) -> list[float]`` method that always returns a unit-norm
    Python list of floats (for JSON serialization)."""

    __slots__ = ("_backend", "_kind", "_model_name")

    def __init__(self, backend: Any, kind: str, model_name: str):
        self._backend = backend
        self._kind = kind            # "fastembed" | "sentence_transformers"
        self._model_name = model_name

    def encode(self, text: str) -> list[float]:
        if self._kind == "fastembed":
            # fastembed.TextEmbedding.embed() yields np.ndarrays; vectors
            # are already unit-norm by construction for BGE models.
            vec = next(self._backend.embed([text]))
            return [float(x) for x in vec]
        # sentence-transformers
        vec = self._backend.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]


def _load_encoder(model_name: str) -> _UnifiedEncoder:
    """Load (and process-wide cache) the BGE encoder for ``model_name``.

    Resolution order:
      1. ``fastembed`` — pure-ONNX, ~50 MB. The default, always-installed path.
      2. ``sentence_transformers`` — only if fastembed can't load the model
         AND ``[remote-semknn]`` is installed.

    Raises ``ImportError`` with an actionable hint if neither backend is
    available — that's a misconfigured install (core dep missing), not a
    runtime fallback.
    """
    enc = _ENCODER_CACHE.get(model_name)
    if enc is not None:
        return enc
    with _ENCODER_LOCK:
        enc = _ENCODER_CACHE.get(model_name)
        if enc is not None:
            return enc

        # --- Try fastembed first (the core dep) ---
        try:
            from fastembed import TextEmbedding
        except ImportError:
            fastembed_err: Exception | None = ImportError(
                "fastembed is missing from the install. This is a core "
                "dependency — without it, anonymous telemetry events ship "
                "without embeddings (no retraining signal) and the SemKNN "
                "router cannot encode prompts. Fix:\n"
                "    pip install fastembed>=0.3\n"
                "or reinstall agentcodec to pick up the current core deps."
            )
        else:
            try:
                logger.info(
                    f"loading BGE encoder {model_name!r} via fastembed "
                    f"(ONNX, ~130 MB for bge-small on first use)"
                )
                enc = _UnifiedEncoder(
                    backend=TextEmbedding(model_name=model_name),
                    kind="fastembed",
                    model_name=model_name,
                )
                _ENCODER_CACHE[model_name] = enc
                return enc
            except Exception as e:
                # fastembed installed but can't serve this model (unknown
                # name, network failure on first download, etc.) — fall
                # through to sentence-transformers.
                fastembed_err = e
                logger.warning(
                    f"fastembed could not load {model_name!r} ({e!r}); "
                    f"trying sentence-transformers fallback"
                )

        # --- Fall back to sentence-transformers (the [remote-semknn] extra) ---
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f"BGE encoder unavailable for {model_name!r}.\n"
                f"  fastembed (core):           {fastembed_err!r}\n"
                f"  sentence-transformers:      {e!r}\n"
                "Install the optional heavyweight encoder with:\n"
                "    pip install 'agentcodec[remote-semknn]'\n"
                "or pick a model name fastembed knows about (BAAI/bge-small-en-v1.5, "
                "BAAI/bge-large-en-v1.5)."
            ) from e
        logger.info(
            f"loading BGE encoder {model_name!r} via sentence-transformers "
            f"(~1.3 GB for bge-large on first use)"
        )
        enc = _UnifiedEncoder(
            backend=SentenceTransformer(model_name),
            kind="sentence_transformers",
            model_name=model_name,
        )
        _ENCODER_CACHE[model_name] = enc
        return enc


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class RemoteSemKNNRouter:
    """SemKNN routing via a remote HTTP service.

    Construction is lazy where it can be — the BGE encoder loads on the
    first :meth:`choose` call, and ``/meta`` is fetched on demand to
    validate that the client's BGE model matches the server's.
    """

    def __init__(
        self,
        server_url: str | None = None,
        *,
        lambda_: float,
        k: int | None = None,
        api_key: str | None = None,
        encoder: Any | None = None,
        bge_model: str | None = None,
        timeout_s: float = 10.0,
        fallback: Router | None = None,
        strict_match: bool | None = None,
        user_config: dict[str, Any] | None = None,
        verify_tls: bool = True,
    ) -> None:
        # server_url is optional — defaults to the public hosted backend
        # at agentcodec.intellerce.com. Override via the constructor, YAML,
        # or the AGENTCODEC_SEMKNN_SERVER_URL env var (applied in factory).
        if not server_url:
            from .._endpoints import AGENTCODEC_SERVER_URL
            server_url = AGENTCODEC_SERVER_URL
        if lambda_ is None or lambda_ < 0:
            raise ValueError("RemoteSemKNNRouter: lambda must be >= 0")

        self.server_url = server_url.rstrip("/")
        self.lambda_ = float(lambda_)
        self.k = int(k) if k is not None else None
        self.api_key = api_key or os.environ.get("AGENTCODEC_API_KEY") or None
        self.bge_model = bge_model or DEFAULT_BGE_MODEL
        self.timeout_s = float(timeout_s)
        self.fallback = fallback
        self.strict_match = strict_match
        self.user_config = user_config
        self.verify_tls = verify_tls

        self._encoder = encoder
        self._meta: dict[str, Any] | None = None
        self._warned: set[tuple[str, str]] = set()
        self._client = httpx.Client(
            timeout=self.timeout_s,
            verify=self.verify_tls,
            headers={"User-Agent": "agentcodec-remote-semknn/0.3"},
        )

    # ----- Router protocol -----

    def choose(self, task: TaskItem) -> RouterDecision:
        try:
            self._ensure_meta_loaded()
            embedding = self._encode(task.prompt or "")
            payload: dict[str, Any] = {
                "embedding": embedding,
                "lambda": self.lambda_,
            }
            if self.k is not None:
                payload["k"] = self.k
            if self.user_config is not None:
                payload["user_config"] = self.user_config
            if self.strict_match is not None:
                payload["strict_match"] = self.strict_match

            resp = self._post_route(payload)
        except _StrictMismatch:
            # 409 — never silently fall back even if a fallback is configured.
            raise
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if self.fallback is None:
                raise RuntimeError(
                    f"RemoteSemKNNRouter: backend at {self.server_url!r} "
                    f"unreachable ({e!r}); no fallback configured. Set "
                    f"router.fallback in YAML to degrade gracefully."
                ) from e
            logger.warning(
                f"RemoteSemKNNRouter: backend error {e!r}; falling back to "
                f"{type(self.fallback).__name__}"
            )
            return self.fallback.choose(task)

        decision = self._decision_from_response(resp)
        # Stash the embedding for telemetry. It's already in flight to the
        # SemKNN backend, so re-sending it to /telemetry adds no new IP risk.
        # `RouterDecision.to_dict()` filters this key out of user-facing traces.
        decision.extra["embedding"] = embedding
        return decision

    # ----- Internals -----

    def _ensure_meta_loaded(self) -> None:
        if self._meta is not None:
            return
        r = self._client.get(f"{self.server_url}/meta", headers=self._auth_headers())
        r.raise_for_status()
        meta = r.json()
        srv_bge = meta.get("bge_model")
        if srv_bge and srv_bge != self.bge_model:
            raise RuntimeError(
                f"RemoteSemKNNRouter: BGE-model mismatch. Client is configured "
                f"for {self.bge_model!r} but server expects {srv_bge!r}. "
                f"Update `bge_model=` in the router config, or point at a "
                f"compatible backend."
            )
        self._meta = meta

    def _post_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(
            f"{self.server_url}/route",
            json=payload,
            headers=self._auth_headers(),
        )
        if r.status_code == 409:
            try:
                detail = r.json()
            except Exception:
                detail = {"detail": r.text}
            raise _StrictMismatch(
                f"RemoteSemKNNRouter: strict-match refused by server. "
                f"No exact-match profile for your model lineup "
                f"({(self.user_config or {}).get('model_families')}). "
                f"To accept an estimate, set `router.strict_match: false` "
                f"in YAML or omit it to use the server's default policy. "
                f"Server detail: {detail}"
            )
        r.raise_for_status()
        return r.json()

    def _decision_from_response(self, resp: dict[str, Any]) -> RouterDecision:
        chosen = resp["chosen"]
        confidence = float(resp.get("confidence", 0.0))
        scores = resp.get("scores")
        match_quality = resp.get("match_quality")
        estimate = bool(resp.get("estimate", False))
        profile_used = resp.get("profile_used")
        warnings_list = resp.get("warnings") or []
        masked = resp.get("masked_techniques") or []

        # Emit a one-time WarningEvent per (profile, match_quality) pair when
        # the server told us the recommendation is an estimate. The warning
        # is surfaced through the RouterDecision's `extra` so ReliabilityModule
        # can attach it to the result's warnings list at trace time.
        warning_payload: dict[str, Any] | None = None
        if estimate and warnings_list:
            key = (str(profile_used), str(match_quality))
            if key not in self._warned:
                self._warned.add(key)
                warning_payload = {
                    "code": "semknn_estimate",
                    "severity": "warn",
                    "message": (
                        f"SemKNN routing is an estimate "
                        f"(match_quality={match_quality}, "
                        f"profile_used={profile_used}). "
                        + " ".join(warnings_list)
                    ),
                }

        extra: dict[str, Any] = {
            "lambda": self.lambda_,
            "k": self.k or (self._meta or {}).get("k_default"),
            "server_url": self.server_url,
            "server_version": (self._meta or {}).get("version"),
            "profile_used": profile_used,
            "match_quality": match_quality,
            "match_similarity": resp.get("match_similarity"),
            "estimate": estimate,
            "predicted_quality_for_chosen": resp.get("predicted_quality_for_chosen"),
            "predicted_cost_for_chosen": resp.get("predicted_cost_for_chosen"),
        }
        if masked:
            extra["masked_techniques"] = masked
        if warnings_list:
            extra["server_warnings"] = warnings_list
        if warning_payload is not None:
            extra["one_time_warning"] = warning_payload

        return RouterDecision(
            chosen=chosen,
            confidence=confidence,
            router_type="semknn_remote",
            candidates_score=dict(scores) if scores else None,
            extra=extra,
        )

    def _encode(self, prompt: str) -> list[float]:
        enc = self._encoder
        if enc is None:
            enc = _load_encoder(self.bge_model)
            self._encoder = enc
        # _UnifiedEncoder.encode() already returns a unit-norm list[float]
        # regardless of which backend (fastembed / sentence-transformers)
        # produced it.
        return enc.encode(prompt)

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class _StrictMismatch(RuntimeError):
    """Raised on HTTP 409 from /route when strict_match refused a fuzzy match.

    Distinct from generic HTTPError so the caller can decide not to fall back.
    """
