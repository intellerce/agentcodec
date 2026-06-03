"""
Process-wide runtime settings — the single source of truth for every
``AGENTCODEC_*`` (and provider) env var the library reads.

This is **not** ``LibraryConfig`` (which is per-deployment, YAML-shaped,
and describes models / strategy / judge). This is the layer underneath:
the handful of knobs that vary by machine and not by deployment.

Precedence (highest wins):

    1. Explicit ``Settings(field=value)`` kwargs.
    2. Environment variables in ``os.environ``.
    3. The hardcoded defaults below.

The library does NOT auto-load ``.env``. Call ``agentcodec.load_dotenv()``
from your own code (or use the examples in ``examples/`` which do this for
you) if you want ``.env`` honored. Shell exports always win over ``.env``.

Why a dedicated Settings class
------------------------------

Provider SDK convention (OpenAI, Anthropic) is to read keys from env on
demand inside the client constructor. That works but has two downsides:

  * the full set of env knobs is **undiscoverable** — you have to
    grep the source to know what the library will read;
  * tests that monkey-patch the env have to remember the variable
    names exactly.

This module fixes both: every knob is a field with a default, a docstring,
and a stable name. ``Settings()`` is cheap (just reads env) and immutable;
build a new one if you want to override at runtime.

Usage
-----

::

    from agentcodec import Settings

    # 1. Read whatever's in env now:
    s = Settings()
    print(s.semknn_server_url)            # → 'https://agentcodec.intellerce.com'
    print(s.telemetry_enabled)            # → True

    # 2. Override programmatically — kwargs win over env:
    s = Settings(semknn_server_url="http://127.0.0.1:18765")
    print(s.semknn_server_url)            # → 'http://127.0.0.1:18765'

    # 3. The library reads `Settings()` lazily where it matters — there's
    #    no process-wide singleton, so each `ReliabilityModule` constructor
    #    can be passed `settings=` to inject a different one (advanced).

What's NOT here
---------------

* **Provider API keys** (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
  ``OLLAMA_API_KEY``). These follow the standard SDK convention — read
  on demand by ``AgentChannel``, **not** centralized through Settings.
  We don't store secrets in a long-lived object. To override, pass
  ``api_key=`` on the per-channel ``ModelConfig`` (or just ``export``
  the variable before launching).
* **Per-deployment knobs** — models, judge, strategy, on_error,
  telemetry batching, etc. Those live in ``LibraryConfig`` (YAML).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._endpoints import AGENTCODEC_SERVER_URL, DEFAULT_TELEMETRY_ENDPOINT

_DISABLED_TRUTHY = frozenset({"0", "false", "no", "off", "disabled", "none"})
_ENABLED_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


def _env_bool(name: str, default: bool) -> bool:
    """Three-valued env read with `_ENABLED_TRUTHY` / `_DISABLED_TRUTHY`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _ENABLED_TRUTHY:
        return True
    if v in _DISABLED_TRUTHY:
        return False
    return default


def _env_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return v if v else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Process-wide runtime settings, sourced from env at construction.

    Every field corresponds to one ``AGENTCODEC_*`` env var. The
    ``env_name`` for each is documented inline so contributors can
    discover them without grepping. Construction is cheap — feel free
    to do ``Settings()`` per request if you want to pick up live env
    changes (though that's unusual).
    """

    # --- SemKNN routing ---------------------------------------------------
    #: SemKNN backend base URL. Public hosted backend by default.
    #: env: ``AGENTCODEC_SEMKNN_SERVER_URL``
    semknn_server_url: str = AGENTCODEC_SERVER_URL

    # --- Telemetry --------------------------------------------------------
    #: Anonymous usage telemetry kill switch. ``False`` disables entirely.
    #: env: ``AGENTCODEC_TELEMETRY`` (accepts 0/false/no/off/disabled)
    telemetry_enabled: bool = True

    #: Override the telemetry POST endpoint. ``None`` means "fall back to
    #: ``{semknn_server_url}/telemetry`` for SemKNN routes, or the
    #: hardcoded public collector otherwise".
    #: env: ``AGENTCODEC_TELEMETRY_ENDPOINT``
    telemetry_endpoint: str | None = None

    #: Silence the one-time "telemetry is on" stderr notice.
    #: env: ``AGENTCODEC_TELEMETRY_QUIET``
    telemetry_quiet: bool = False

    # --- Pricing catalog --------------------------------------------------
    #: Skip the OpenRouter live-pricing fetch (offline / locked-down).
    #: env: ``AGENTCODEC_DISABLE_OPENROUTER``
    disable_openrouter: bool = False

    #: Where to keep the OpenRouter pricing cache JSON. ``None`` →
    #: ``<repo>/.cache/agentcodec/``.
    #: env: ``AGENTCODEC_CACHE_DIR``
    cache_dir: str | None = None

    # --- Task lifecycle ---------------------------------------------------
    #: Wall-clock ceiling per task. The benchmark runner enforces it;
    #: the library facade uses its own ``defaults.task_timeout_s``.
    #: env: ``AGENTCODEC_TASK_TIMEOUT_S``
    task_timeout_s: int = 600

    #: Per-channel transmit attempts (= retries + 1).
    #: env: ``AGENTCODEC_TASK_MAX_ATTEMPTS``
    task_max_attempts: int = 3

    # --- Dotenv -----------------------------------------------------------
    #: Suppress `.env` autoloading even when ``agentcodec.load_dotenv()``
    #: is called explicitly. Useful for serverless / 12-factor.
    #: env: ``AGENTCODEC_DISABLE_DOTENV``
    disable_dotenv: bool = False

    # --- Loader -----------------------------------------------------------
    @classmethod
    def from_env(cls) -> Settings:
        """Read all fields from ``os.environ`` once and return a frozen
        instance. Identical to ``Settings()`` but spelled explicitly."""
        return cls()

    def __post_init__(self) -> None:
        # We're a frozen dataclass, so we have to use object.__setattr__
        # to apply the env-resolution defaults.
        kwargs_present = {k: object.__getattribute__(self, k) for k in self.__dataclass_fields__}

        # Only apply env overrides for fields the user didn't explicitly
        # override at construction. We detect "not overridden" by
        # comparing to the field's default — kwargs that match the
        # default are treated the same as "not passed"; that's the
        # frozen-dataclass tax.
        defaults = {
            "semknn_server_url":   AGENTCODEC_SERVER_URL,
            "telemetry_enabled":   True,
            "telemetry_endpoint":  None,
            "telemetry_quiet":     False,
            "disable_openrouter":  False,
            "cache_dir":           None,
            "task_timeout_s":      600,
            "task_max_attempts":   3,
            "disable_dotenv":      False,
        }
        env_resolvers = {
            "semknn_server_url":   lambda d: _env_str("AGENTCODEC_SEMKNN_SERVER_URL", d),
            "telemetry_enabled":   lambda d: _env_bool("AGENTCODEC_TELEMETRY", d),
            "telemetry_endpoint":  lambda d: _env_str("AGENTCODEC_TELEMETRY_ENDPOINT", d),
            "telemetry_quiet":     lambda d: _env_bool("AGENTCODEC_TELEMETRY_QUIET", d),
            "disable_openrouter":  lambda d: _env_bool("AGENTCODEC_DISABLE_OPENROUTER", d),
            "cache_dir":           lambda d: _env_str("AGENTCODEC_CACHE_DIR", d),
            "task_timeout_s":      lambda d: _env_int("AGENTCODEC_TASK_TIMEOUT_S", d),
            "task_max_attempts":   lambda d: _env_int("AGENTCODEC_TASK_MAX_ATTEMPTS", d),
            "disable_dotenv":      lambda d: _env_bool("AGENTCODEC_DISABLE_DOTENV", d),
        }
        for name, resolve in env_resolvers.items():
            current = kwargs_present[name]
            default = defaults[name]
            if current == default:
                object.__setattr__(self, name, resolve(default))


# Sentinel used by other modules: the default telemetry collector when no
# YAML / env override is present. Kept here so callers don't import from
# the internal `_endpoints` module.
DEFAULT_PUBLIC_TELEMETRY_ENDPOINT = DEFAULT_TELEMETRY_ENDPOINT


__all__ = ["DEFAULT_PUBLIC_TELEMETRY_ENDPOINT", "Settings"]
