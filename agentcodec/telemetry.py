"""
Anonymous usage telemetry.

The client emits one event per ``ReliabilityModule.run()`` / ``.stream()``
call with: embedding (no prompt text), lambda, technique used, predicted
vs. observed quality, latency, token counts, and the canonical model-family
fingerprint. Nothing that identifies the user, the prompt, the model
output, or the reference answer is ever sent.

Opt out:
    export AGENTCODEC_TELEMETRY=0       # also accepts false/no/off/disabled
    export AGENTCODEC_TELEMETRY_QUIET=1  # suppress the one-time stderr notice

YAML toggle:
    telemetry:
      enabled: false                    # overrides per-module
      endpoint: "https://your-telemetry-host/telemetry"

Implementation
--------------
- Background daemon thread drains a bounded ``queue.Queue``. Caller
  enqueues with ``put_nowait``; if the queue is full the event is dropped
  silently. Telemetry MUST NEVER block, slow, or fail the caller's
  request — that's the prime directive.
- Batches up to ``BATCH_MAX`` events or ``FLUSH_INTERVAL_S`` seconds before
  POSTing as a single ``{"events": [...]}`` body.
- Network/server errors → drop the batch, log at DEBUG only.
- ``atexit`` flush gives in-flight events one last chance.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
BATCH_MAX = 32
FLUSH_INTERVAL_S = 30.0
DEFAULT_QUEUE_MAX = 1000
DEFAULT_TIMEOUT_S = 5.0

_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled", "none"})
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------


def _env_disabled(env_var: str) -> bool | None:
    """Three-valued read of the master toggle.

    Returns:
        True  → caller set it to a disabled value (overrides YAML)
        False → caller set it to an enabled value  (overrides YAML)
        None  → unset (defer to YAML / default)
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return None
    v = raw.strip().lower()
    if not v:
        return None
    if v in _DISABLED_VALUES:
        return True
    if v in _ENABLED_VALUES:
        return False
    # Unknown values default to "treat as enabled" so a typo doesn't
    # silently disable telemetry — but log so the user notices.
    logger.warning(
        "AgentCodec: unrecognized telemetry env value %r; treating as 'on'",
        raw,
    )
    return False


# ---------------------------------------------------------------------------
# Config + sanitization
# ---------------------------------------------------------------------------


@dataclass
class TelemetryConfig:
    """Per-module telemetry settings. Constructed from LibraryConfig."""
    enabled: bool = True
    endpoint: str | None = None
    quiet_notice: bool = False
    flush_interval_s: float = FLUSH_INTERVAL_S
    queue_max: int = DEFAULT_QUEUE_MAX
    batch_max: int = BATCH_MAX
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_block(
        cls, block: dict[str, Any] | None, fallback_endpoint: str | None,
    ) -> TelemetryConfig:
        block = dict(block or {})
        endpoint = block.get("endpoint") or fallback_endpoint
        return cls(
            enabled=bool(block.get("enabled", True)),
            endpoint=endpoint,
            quiet_notice=bool(block.get("quiet_notice", False)),
            flush_interval_s=float(block.get("flush_interval_s", FLUSH_INTERVAL_S)),
            queue_max=int(block.get("queue_max", DEFAULT_QUEUE_MAX)),
            batch_max=int(block.get("batch_max", BATCH_MAX)),
            timeout_s=float(block.get("timeout_s", DEFAULT_TIMEOUT_S)),
        )


# Keys we will NEVER serialize, even if a caller mistakenly puts them on a
# payload. Belt-and-braces defense against future code accidentally leaking
# a prompt / output / secret.
_FORBIDDEN_KEYS = frozenset({
    "prompt", "prompt_text", "text",
    "output", "output_text", "completion", "response_text",
    "reference", "reference_answer", "gold",
    "api_key", "openai_api_key", "anthropic_api_key", "authorization",
    "task_id",                # user-supplied, could carry PII
    "metadata",                # opaque user dict
    "ip", "remote_addr", "user_id", "user", "account_id",
})


def _scrub(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop any key on the forbidden list before sending.

    Recursive over nested dicts/lists. Strings that look unbounded
    (>1024 chars) are also dropped — a safety net against accidentally
    embedding free-form text under an innocent-looking field name.
    """
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if str(k).lower() in _FORBIDDEN_KEYS:
                continue
            scrubbed = _scrub(v)
            if isinstance(scrubbed, str) and len(scrubbed) > 1024:
                # Drop unbounded text — telemetry has no business carrying
                # long strings.
                continue
            out[k] = scrubbed
        return out
    if isinstance(payload, list):
        return [_scrub(x) for x in payload]
    return payload


# ---------------------------------------------------------------------------
# Telemetry client
# ---------------------------------------------------------------------------


@dataclass
class _Notice:
    """One-time stderr notice. The text matches what the README says."""
    shown: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def maybe_show(self, cfg: TelemetryConfig, env_var: str) -> None:
        if cfg.quiet_notice or os.environ.get("AGENTCODEC_TELEMETRY_QUIET"):
            return
        with self._lock:
            if self.shown:
                return
            self.shown = True
        try:
            print(
                f"[agentcodec] anonymous telemetry on (embedding + metrics, "
                f"no prompts/outputs/keys); opt out: {env_var}=0, "
                f"silence: AGENTCODEC_TELEMETRY_QUIET=1.",
                file=sys.stderr,
            )
        except Exception:
            # Never let the notice itself break anything.
            pass


class Telemetry:
    """Per-module telemetry client. Cheap to construct; thread-safe to use.

    The expensive parts (background thread, http client) are lazy — they
    start on the first ``record()`` call. If telemetry is disabled
    everything short-circuits to a no-op.
    """

    ENV_VAR_MASTER = "AGENTCODEC_TELEMETRY"
    ENV_VAR_ENDPOINT = "AGENTCODEC_TELEMETRY_ENDPOINT"

    _notice = _Notice()      # process-wide; "enabled" notice appears once

    def __init__(
        self,
        cfg: TelemetryConfig,
        *,
        client_version: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.client_version = client_version
        self.cfg = cfg

        # Resolve enabled flag: env-var > YAML config > default(True).
        env_off = _env_disabled(self.ENV_VAR_MASTER)
        if env_off is None:
            self._enabled = bool(cfg.enabled)
        else:
            self._enabled = not env_off

        # Allow an env override of the endpoint too, for sysadmins.
        env_ep = os.environ.get(self.ENV_VAR_ENDPOINT)
        if env_ep:
            self.endpoint = env_ep
        else:
            self.endpoint = cfg.endpoint

        # No endpoint → silently no-op. In practice every router resolves
        # to one: SemKNN derives `{server_url}/telemetry`, everyone else
        # falls back to the hardcoded public collector in `_endpoints.py`.
        if self._enabled and not self.endpoint:
            self._enabled = False

        self.session_id = str(uuid.uuid4())

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=cfg.queue_max)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._http = http_client     # tests inject a mock
        self._dropped = 0
        self._sent = 0
        self._failed_batches = 0
        self._atexit_registered = False
        self._lock = threading.Lock()

    # ----- public API ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def stats(self) -> dict[str, int]:
        return {
            "sent": self._sent,
            "dropped": self._dropped,
            "failed_batches": self._failed_batches,
            "queue_size": self._queue.qsize(),
        }

    def record(self, payload: dict[str, Any]) -> None:
        """Enqueue one event. No-op when disabled. Never blocks."""
        if not self._enabled:
            return
        self._notice.maybe_show(self.cfg, self.ENV_VAR_MASTER)

        envelope = {
            "schema_version": SCHEMA_VERSION,
            "client_version": self.client_version,
            "session_id": self.session_id,
            "ts_iso": _utc_now_iso(),
        }
        envelope.update(_scrub(payload))

        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            self._dropped += 1
            return
        self._ensure_worker()

    def flush(self, timeout_s: float | None = None) -> bool:
        """Block until the queue drains or timeout. Returns True if drained."""
        if not self._enabled or self._thread is None:
            return True
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else 10.0)
        while self._queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        return self._queue.qsize() == 0

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """Stop the worker, drain remaining events, close the client."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=timeout_s)
            self._thread = None
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None

    # ----- internals -----------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            t = threading.Thread(
                target=self._worker,
                name="agentcodec-telemetry",
                daemon=True,
            )
            t.start()
            self._thread = t
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

    def _worker(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        timeout = self.cfg.flush_interval_s
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=timeout)
                batch.append(event)
            except queue.Empty:
                event = None

            should_flush = (
                len(batch) >= self.cfg.batch_max
                or (batch and (time.monotonic() - last_flush) >= self.cfg.flush_interval_s)
            )
            if should_flush:
                self._flush(batch)
                batch = []
                last_flush = time.monotonic()

        # Drain remaining queue + buffered batch on shutdown.
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        if not batch or not self.endpoint:
            return
        client = self._get_http()
        try:
            # Compact separators to match the server sink's wire format and
            # keep telemetry bytes minimal.
            body = json.dumps(
                {"events": batch}, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            r = client.post(
                self.endpoint,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=self.cfg.timeout_s,
            )
            if r.status_code >= 400:
                self._failed_batches += 1
                logger.debug(
                    "telemetry: server returned %d for %d events; dropping",
                    r.status_code, len(batch),
                )
                return
            self._sent += len(batch)
        except Exception as e:
            self._failed_batches += 1
            logger.debug(
                "telemetry: flush failed (%r); dropping %d events",
                e, len(batch),
            )

    def _get_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                timeout=self.cfg.timeout_s,
                headers={"User-Agent": f"agentcodec-telemetry/{self.client_version}"},
            )
        return self._http


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z",
    )


def build_event_from_result(
    *,
    result: Any,
    routing_extra: dict[str, Any] | None,
    router_type: str,
    user_config: dict[str, Any] | None,
    lambda_: float | None,
    embedding: list[float] | None,
    bge_model: str | None,
    task_category: str | None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Turn a ReliabilityResult into a telemetry payload.

    Intentionally does NOT touch ``result.text``, ``result.reference``,
    ``result.task_id``, or any user-supplied metadata. Caller passes only
    the bits we want.
    """
    extra = routing_extra or {}
    payload: dict[str, Any] = {
        "router_type": router_type,
        "technique_used": getattr(result, "technique_used", None),
        "task_category": task_category,
        "latency_s": getattr(result, "latency_s", None),
        "wall_clock_s": getattr(result, "wall_clock_s", None)
            or getattr(result, "latency_s", None),
        "cumulative_latency_s": getattr(result, "cumulative_latency_s", None),
        "input_tokens": getattr(result, "input_tokens", None),
        "output_tokens": getattr(result, "output_tokens", None),
        "thinking_tokens": getattr(result, "thinking_tokens", None),
        "rounds": getattr(result, "rounds", None),
        "num_llm_calls": getattr(result, "num_llm_calls", None),
        "thinking_used": getattr(result, "thinking_used", None),
        # `observed_quality` is the user's judge's score for the FINAL answer.
        # This is the retraining signal we pair with `predicted_quality`
        # (set by SemKNN at /route time) to compute regret.
        "observed_quality": getattr(result, "final_quality", None),
        "best_individual_quality": getattr(result, "best_individual_quality", None),
        "diversity_gain": getattr(result, "diversity_gain", None),
        "observed_cost_usd": getattr(result, "cost_usd", None),
        "judge_cost_usd": getattr(result, "judge_cost_usd", None),
        "cost_source": getattr(result, "cost_source", None),
        "lambda": lambda_,
        # Embedding is REQUIRED for the event to be useful — without it
        # there's no way to attach the (quality, cost) observation to a
        # row of the SemKNN q-matrix. The caller (api._record_telemetry)
        # enforces this by skipping the event when the encoder can't
        # produce an embedding.
        "embedding": embedding,
        "embedding_bge_model": bge_model if embedding else None,
        "user_config": user_config,
        "profile_used": extra.get("profile_used"),
        "match_quality": extra.get("match_quality"),
        "match_similarity": extra.get("match_similarity"),
        "estimate": extra.get("estimate"),
        "predicted_quality": extra.get("predicted_quality_for_chosen"),
        "predicted_cost_usd": extra.get("predicted_cost_for_chosen"),
        "k": extra.get("k"),
        "error_type": error_type,
    }
    return payload
