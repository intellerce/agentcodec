"""
Zero-dep ``.env`` loader, library-wide.

We auto-load ``.env`` on package import so every entry point — examples,
CLI, library facade, notebooks — honors the same file for:

  * provider API keys (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.)
  * SemKNN server URL (``AGENTCODEC_SEMKNN_SERVER_URL``)
  * telemetry endpoint (``AGENTCODEC_TELEMETRY_ENDPOINT``)
  * cost-pricing cache (``AGENTCODEC_DISABLE_OPENROUTER``, ``AGENTCODEC_CACHE_DIR``)
  * any other ``AGENTCODEC_*`` knob

Three resolution rules, in order:

  1. **Shell exports always win.** ``os.environ`` already-set keys are
     never overwritten — predictable behavior for CI, containers, and
     "I'm debugging, please honor my one-shot ``KEY=value cmd``".
  2. **First file found, walking up.** Start at the current working
     directory and walk up to ``/``. The first ``.env`` we see is the
     one we load. (Same shape as Vite / next.js / direnv.)
  3. **Missing file is fine.** No file? Silent no-op. We do NOT scold
     users for not having one.

Override via :func:`load_dotenv` if you need a custom path.

Disable entirely by setting ``AGENTCODEC_DISABLE_DOTENV=1`` in the shell
(useful for serverless / 12-factor deployments where ``.env`` is a smell).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# Markers that identify a "project root". We stop walking up the
# directory tree as soon as we hit one, so we never accidentally pick
# up a `.env` from the user's home directory or further up.
_PROJECT_ROOT_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    ".git",
)


def _find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for the first ``.env``.

    Stops at the first directory containing a project-root marker
    (``pyproject.toml``, ``setup.py``, ``setup.cfg``, ``.git``) — even if
    that directory has no ``.env`` itself. That fence keeps us from
    silently loading ``~/.env`` when a user runs an unrelated script
    inside a directory without its own project root.
    """
    here = Path(start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
        # Stop AFTER checking this dir if it's a project root — don't
        # cross the boundary into parent projects / the home dir.
        if any((parent / m).exists() for m in _PROJECT_ROOT_MARKERS):
            return None
    return None


def load_dotenv(
    path: Path | str | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Parse ``path`` (or auto-discovered ``.env``) and merge into env.

    Supports ``KEY=value``, ``export KEY=value``, blank lines,
    ``# comments`` (line + inline), and optional quoted values
    (single or double).

    Args:
        path:     explicit ``.env`` location. ``None`` auto-discovers
                  by walking up from cwd.
        override: when True, replace existing ``os.environ`` values.
                  Default False (shell exports win).

    Returns:
        A dict of every parsed ``key: value`` pair, even ones already
        in the env (so callers can audit what the file would set).
    """
    if os.environ.get("AGENTCODEC_DISABLE_DOTENV", "").lower() in (
        "1", "true", "yes", "on",
    ):
        return {}

    if path is None:
        found = _find_dotenv()
    else:
        candidate = Path(path)
        found = candidate if candidate.is_file() else None
    if found is None:
        return {}

    parsed: dict[str, str] = {}
    try:
        text = found.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("dotenv: could not read %s: %r", found, e)
        return {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip inline comments only when the value isn't quoted, so we
        # don't mangle `KEY="foo # bar"`.
        if value and value[0] not in ("'", '"'):
            value = value.split("#", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    if parsed:
        # INFO (not DEBUG) so users can audit what `.env` actually got
        # applied — useful when debugging "why does this process have
        # OPENAI_API_KEY set when I didn't export it?".
        # We log only the file path and key COUNT — never the values.
        logger.info(
            "agentcodec: loaded %d key(s) from %s (override=%s)",
            len(parsed), found, override,
        )
    return parsed
