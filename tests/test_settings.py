"""Tests for ``agentcodec.Settings``.

Pin the three resolution rules:

  1. Kwargs win over env vars.
  2. Env vars win over defaults.
  3. Defaults are sensible and documented.

Plus the env-var spellings, since they're part of the public surface
(users put these in `.env` or `export` them).
"""

from __future__ import annotations

import pytest

from agentcodec import Settings
from agentcodec._endpoints import AGENTCODEC_SERVER_URL

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_with_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nothing is set, defaults match the documented values."""
    for k in (
        "AGENTCODEC_SEMKNN_SERVER_URL",
        "AGENTCODEC_TELEMETRY",
        "AGENTCODEC_TELEMETRY_ENDPOINT",
        "AGENTCODEC_TELEMETRY_QUIET",
        "AGENTCODEC_DISABLE_OPENROUTER",
        "AGENTCODEC_CACHE_DIR",
        "AGENTCODEC_TASK_TIMEOUT_S",
        "AGENTCODEC_TASK_MAX_ATTEMPTS",
        "AGENTCODEC_DISABLE_DOTENV",
    ):
        monkeypatch.delenv(k, raising=False)

    s = Settings()
    assert s.semknn_server_url == AGENTCODEC_SERVER_URL
    assert s.telemetry_enabled is True
    assert s.telemetry_endpoint is None
    assert s.telemetry_quiet is False
    assert s.disable_openrouter is False
    assert s.cache_dir is None
    assert s.task_timeout_s == 600
    assert s.task_max_attempts == 3
    assert s.disable_dotenv is False


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("1",        True),
    ("true",     True),
    ("yes",      True),
    ("ON",       True),
    ("0",        False),
    ("false",    False),
    ("no",       False),
    ("off",      False),
    ("disabled", False),
])
def test_telemetry_env_truthiness(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool,
) -> None:
    monkeypatch.setenv("AGENTCODEC_TELEMETRY", value)
    s = Settings()
    assert s.telemetry_enabled is expected


def test_semknn_server_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCODEC_SEMKNN_SERVER_URL", "http://127.0.0.1:18765")
    s = Settings()
    assert s.semknn_server_url == "http://127.0.0.1:18765"


def test_telemetry_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AGENTCODEC_TELEMETRY_ENDPOINT",
        "https://my-collector.example.com/telemetry",
    )
    s = Settings()
    assert s.telemetry_endpoint == "https://my-collector.example.com/telemetry"


def test_task_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCODEC_TASK_TIMEOUT_S", "1200")
    s = Settings()
    assert s.task_timeout_s == 1200


def test_int_field_falls_back_to_default_on_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-numeric env values for int fields should silently fall back
    to the default rather than crash on import."""
    monkeypatch.setenv("AGENTCODEC_TASK_TIMEOUT_S", "not-a-number")
    s = Settings()
    assert s.task_timeout_s == 600


def test_disable_openrouter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCODEC_DISABLE_OPENROUTER", "1")
    s = Settings()
    assert s.disable_openrouter is True


def test_disable_dotenv_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCODEC_DISABLE_DOTENV", "yes")
    s = Settings()
    assert s.disable_dotenv is True


def test_cache_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCODEC_CACHE_DIR", "/var/lib/agentcodec")
    s = Settings()
    assert s.cache_dir == "/var/lib/agentcodec"


# ---------------------------------------------------------------------------
# Kwargs > env > defaults
# ---------------------------------------------------------------------------


def test_kwargs_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AGENTCODEC_SEMKNN_SERVER_URL", "http://from-env.example.com",
    )
    s = Settings(semknn_server_url="http://from-kwarg.example.com")
    assert s.semknn_server_url == "http://from-kwarg.example.com"


# ---------------------------------------------------------------------------
# Immutability — frozen dataclass
# ---------------------------------------------------------------------------


def test_settings_is_frozen() -> None:
    """Mutation must raise — settings should be passed around, not edited."""
    import dataclasses
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.semknn_server_url = "anything"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# from_env() alias
# ---------------------------------------------------------------------------


def test_from_env_matches_default_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTCODEC_TASK_TIMEOUT_S", "42")
    a = Settings()
    b = Settings.from_env()
    assert a.task_timeout_s == b.task_timeout_s == 42
