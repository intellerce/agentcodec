"""
Library deployment configuration.

The `LibraryConfig` schema describes everything a hosting application needs
to instantiate a `ReliabilityModule`: the LLM channels, the judge, the
optional critic, the strategy (fixed technique vs routed), and runtime
defaults. It's intentionally separate from `runner.ExperimentConfig`
(which is benchmark-shaped) so the deployment surface stays clean.

Loaded from YAML or a dict. Strict — unknown keys raise at load time, so
typos surface immediately rather than silently falling back to defaults.

See ``configs/lib/`` for ready-to-copy examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class CostPer1M(BaseModel):
    """Per-million-token rates in USD. Both fields required when present."""
    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ThinkingConfig(BaseModel):
    """Explicit thinking control. See README §thinking for the full table."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    budget_tokens: int | None = Field(default=None, ge=1)

    # Reject the seconds-budget gotcha at config-load time.
    @model_validator(mode="before")
    @classmethod
    def _reject_seconds(cls, values: Any) -> Any:
        if isinstance(values, dict) and "budget_seconds" in values:
            raise ValueError(
                "thinking.budget_seconds is not supported. The underlying "
                "APIs only support token budgets — use `budget_tokens` "
                "instead, or wrap the call in your own asyncio timeout."
            )
        return values


class ModelConfig(BaseModel):
    """A single LLM channel."""
    model_config = ConfigDict(extra="forbid")
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 32768
    # Per-call HTTP timeout (seconds) for this channel's LLM requests. None
    # keeps the built-in default (300 for Ollama endpoints, 240 otherwise).
    # Raise it for slow local / reasoning models that exceed the default.
    request_timeout_s: float | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] | None = None
    category_temperatures: dict[str, float] | None = None
    cost_per_1m: CostPer1M | None = None
    # Accepts: bool (True/False), "auto", or a {enabled, budget_tokens} dict.
    thinking: bool | Literal["auto"] | ThinkingConfig | None = None
    # Default channel-wide system prompt. Per-call values passed via
    # ChatRequest (or mod.run(system=...)) override this on a per-request
    # basis; the channel default is used only when the request didn't
    # bring its own. None falls back to the legacy "You are a helpful
    # assistant." default inside AgentChannel.
    system_prompt: str | None = None

    @field_validator("thinking", mode="before")
    @classmethod
    def _coerce_thinking(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool) or v == "auto":
            return v
        if isinstance(v, dict):
            return ThinkingConfig.model_validate(v)
        if isinstance(v, ThinkingConfig):
            return v
        raise ValueError(f"thinking must be bool | 'auto' | dict, got {v!r}")


class JudgeConfig(BaseModel):
    """Judge / quality scorer configuration."""
    model_config = ConfigDict(extra="forbid")
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    extra_body: dict[str, Any] | None = None
    cost_per_1m: CostPer1M | None = None
    thinking: bool | Literal["auto"] | ThinkingConfig | None = None
    # Optional system_prompt override for the judge channel. Rarely needed:
    # the QualityScorer builds its own task-shaped prompts that already
    # carry detailed evaluation instructions. Provided for parity with
    # ModelConfig and to support drop-in compatibility shims that pass a
    # global system message via the reliability layer.
    system_prompt: str | None = None


class CriticConfig(BaseModel):
    """Critic for iterative techniques (HARQ-IR, Turbo)."""
    model_config = ConfigDict(extra="forbid")
    # `same: true` reuses the channel model (default).
    # `model: "..."` uses an explicit critic model.
    same: bool = True
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    cost_per_1m: CostPer1M | None = None
    thinking: bool | Literal["auto"] | ThinkingConfig | None = None

    @model_validator(mode="after")
    def _check_exclusive(self) -> CriticConfig:
        if not self.same and not self.model:
            raise ValueError("critic.same=false requires critic.model to be set")
        if self.same and self.model:
            raise ValueError("critic.same=true and critic.model set — pick one")
        return self


# --- Strategy variants -----------------------------------------------------


class FixedStrategy(BaseModel):
    """Always run the same technique (e.g. harq_ir)."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["fixed"] = "fixed"
    technique: str
    params: dict[str, Any] | None = None  # technique-specific knobs


class RouterConfig(BaseModel):
    """Per-router-type config block, embedded in RoutedStrategy.

    SemKNN is a remote service in this release. `server_url` defaults to
    the public hosted backend at ``https://agentcodec.intellerce.com``;
    `lambda` must be set explicitly because the quality/cost trade-off
    is deployment-specific. ACM-table and ACM-linear remain fully local.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["semknn", "acm_table", "acm_linear"]

    # --- SemKNN (remote) -------------------------------------------------
    # Default points at the public hosted endpoint. Override in YAML or
    # via the AGENTCODEC_SEMKNN_SERVER_URL env var (env wins, see
    # routing/factory.py).
    server_url: str | None = None
    lambda_: float | None = Field(default=None, ge=0, alias="lambda")
    api_key: str | None = None              # else falls back to $AGENTCODEC_API_KEY
    timeout_s: float = Field(default=10.0, gt=0)
    knn_k_override: int | None = Field(default=None, ge=1)
    # Per-request override of the server's deployment-wide policy.
    # None → use whatever the server is configured for (default `warn`).
    strict_match: bool | None = None
    # Offline degradation. "none" (default) → fail loudly on backend errors.
    fallback: Literal["none", "linear", "acm_table"] = "none"
    fallback_cache: str | None = None       # required when fallback == "linear"

    # --- ACM-linear (local) ----------------------------------------------
    cache: str | None = None                # acm_linear only now

    # --- ACM-table (local) -----------------------------------------------
    table: list[dict[str, Any]] | None = None
    category_tables: dict[str, list[dict[str, Any]]] | None = None

    @model_validator(mode="after")
    def _validate(self) -> RouterConfig:
        if self.type == "semknn":
            if self.cache is not None:
                raise ValueError(
                    "router.cache is not supported for type=semknn in this "
                    "release. SemKNN is now a remote service: set "
                    "`server_url` and `lambda` instead. Run your own backend "
                    "with the agentcodec-semknn-server image, or contact the "
                    "authors for a hosted endpoint."
                )
            # `server_url` is OPTIONAL — defaults to the public hosted
            # backend below. Users self-hosting or running a dev backend
            # override via YAML or the AGENTCODEC_SEMKNN_SERVER_URL env var.
            if not self.server_url:
                from ._endpoints import AGENTCODEC_SERVER_URL
                self.server_url = AGENTCODEC_SERVER_URL
            if self.lambda_ is None:
                raise ValueError(
                    "router.lambda is required for type=semknn "
                    "(use `lambda: <float>` in YAML)"
                )
            if self.fallback == "linear" and not self.fallback_cache:
                raise ValueError(
                    "router.fallback=linear requires fallback_cache to point "
                    "at a trained linear-router JSON"
                )
            if self.fallback == "acm_table" and not (self.table or self.category_tables):
                raise ValueError(
                    "router.fallback=acm_table requires `table` or "
                    "`category_tables` to be set on the same RouterConfig"
                )
        elif self.type == "acm_linear":
            if not self.cache:
                raise ValueError("router.cache is required for type=acm_linear")
            for field in ("server_url", "lambda_", "strict_match"):
                if getattr(self, field) is not None:
                    raise ValueError(f"router.{field} is only valid for type=semknn")
        elif self.type == "acm_table":
            if not (self.table or self.category_tables):
                raise ValueError(
                    "router.table or router.category_tables required for "
                    "type=acm_table"
                )
            for field in ("server_url", "lambda_", "strict_match", "cache"):
                if getattr(self, field) is not None:
                    raise ValueError(f"router.{field} is not valid for type=acm_table")
        return self


class RoutedStrategy(BaseModel):
    """Pick a technique per call via a router."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["routed"] = "routed"
    router: RouterConfig
    dispatch: dict[str, dict[str, Any]] | None = None  # per-technique knob overrides


Strategy = FixedStrategy | RoutedStrategy


# --- Runtime defaults ------------------------------------------------------


class StreamingDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # What events the stream emits by default. Hosts can filter at the
    # consumer side; this just controls what's produced.
    events: Literal["all", "tokens", "progress"] = "all"
    chunk_format: Literal["delta", "cumulative"] = "delta"
    emit_thinking_tokens: bool = False  # off by default — host opts in


class SoftNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    T_logprob: float = 0.1
    T_judge: float = 0.5
    T_verbal_100: float = 8.0


class CISCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csi_source: Literal["verbal_100", "response_probability"] = "verbal_100"
    softmax_temperature: float | None = None
    num_samples: int = Field(default=5, ge=1)


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["auto", "qa", "reasoning", "creative", "code"] = "auto"
    on_error: Literal["raise", "fallback_baseline"] = "raise"
    task_timeout_s: int = Field(default=300, ge=0)
    early_exit: bool = False
    soft_normalization: SoftNormalization = Field(default_factory=SoftNormalization)
    cisc: CISCConfig = Field(default_factory=CISCConfig)
    streaming: StreamingDefaults = Field(default_factory=StreamingDefaults)


class TelemetryYAMLConfig(BaseModel):
    """Optional telemetry block on LibraryConfig.

    Anonymous usage telemetry — see README §'Anonymous telemetry' for the
    full payload list. Master kill switch is the env var
    ``AGENTCODEC_TELEMETRY=0``; this YAML block is for per-deployment
    overrides (custom endpoint, batch size) only.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # Where to POST. When unset and `router.type == semknn`, defaults to
    # `{server_url}/telemetry`. Otherwise telemetry is disabled even when
    # `enabled: true`.
    endpoint: str | None = None
    quiet_notice: bool = False
    flush_interval_s: float = Field(default=30.0, gt=0)
    queue_max: int = Field(default=1000, ge=1)
    batch_max: int = Field(default=32, ge=1)
    timeout_s: float = Field(default=5.0, gt=0)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class LibraryConfig(BaseModel):
    """Top-level deployment configuration. Strict — rejects unknown keys."""
    model_config = ConfigDict(extra="forbid")
    models: list[ModelConfig] = Field(min_length=1)
    judge: JudgeConfig
    critic: CriticConfig | None = None
    strategy: Strategy
    defaults: Defaults = Field(default_factory=Defaults)
    # How QualityScorer combines deterministic checks with the LLM judge for
    # tasks that carry a score_mode. See README §Tiered scoring. Set to
    # "judge" to reproduce paper numbers exactly.
    score_strategy: Literal["blended", "exact", "judge"] = "blended"
    # Optional global cost overrides keyed by model name. Wins over MODEL_COSTS
    # but loses to per-model `cost_per_1m`. Useful when many models share
    # custom pricing.
    cost_overrides: dict[str, CostPer1M] | None = None
    # Optional anonymous-telemetry block. See agentcodec.telemetry. Master
    # kill switch is AGENTCODEC_TELEMETRY=0.
    telemetry: TelemetryYAMLConfig = Field(default_factory=TelemetryYAMLConfig)

    # ----- loaders -----

    @classmethod
    def from_yaml(cls, path: str | Path) -> LibraryConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LibraryConfig:
        return cls.model_validate(data)

    # ----- helpers used by ReliabilityModule -----

    def resolved_cost_for(self, model_name: str) -> tuple[float, float] | None:
        """Per-model cost_per_1m wins; else falls back to top-level
        cost_overrides; else None (channel will use MODEL_COSTS)."""
        for m in self.models:
            if m.model == model_name and m.cost_per_1m is not None:
                return (m.cost_per_1m.input, m.cost_per_1m.output)
        if self.cost_overrides and model_name in self.cost_overrides:
            cp = self.cost_overrides[model_name]
            return (cp.input, cp.output)
        return None
