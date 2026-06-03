"""
ReliabilityModule — the public library facade.

Construct from a YAML config or a dict, then call `.run()`, `.arun()`,
`.stream()`, or `.astream()` to get reliability-enhanced answers.

    from agentcodec import ReliabilityModule

    mod = ReliabilityModule.from_yaml("configs/lib/routed_semknn.yaml")
    out = mod.run("What's the capital of France?", category="qa")
    print(out.text)                  # the answer
    print(out.cost_usd)              # estimated cost (with source tier)
    print(out.technique_used)        # e.g. "harq_ir" or "diversity_mrc"

For production observability, request the full trace:

    out = mod.run(prompt, return_trace=True)
    json.dump(out.to_dict(), sys.stdout, indent=2)

For streaming UIs (FastAPI, Starlette, etc.):

    async for event in mod.astream(prompt, category="qa"):
        match event:
            case TokenEvent(text=t):       await ws.send_text(t)
            case ProgressEvent(stage=s):   ...
            case FinalEvent(result=r):     ...

Streaming is *event-based*, not token-only. For diversity-class techniques
the visible "tokens" only come from the final synthesizer call; before that
the host receives `ProgressEvent`s describing branch/score/synthesis
progress. See the README §streaming for the per-technique table.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .channel import AgentChannel, QualityScorer, _is_thinking_model
from .config import (
    CriticConfig,
    FixedStrategy,
    JudgeConfig,
    LibraryConfig,
    ModelConfig,
    RoutedStrategy,
    ThinkingConfig,
)
from .cost import summarize_pricing
from .dispatch import DispatchContext, dispatch
from .messages import ChatRequest, Message
from .models import TaskCategory, TaskItem
from .results import (
    Event,
    FinalEvent,
    ProgressEvent,
    ReliabilityResult,
    TokenEvent,
    WarningEvent,
    build_result_from_run,
)
from .routing import AutoCategoryClassifier, build_router
from .routing.acm_table import ACMTableRouter
from .routing.linear import LinearRouter
from .routing.remote import (
    DEFAULT_BGE_MODEL,
    RemoteSemKNNRouter,
    _derive_user_config,
    _load_encoder,
)
from .telemetry import Telemetry, TelemetryConfig, build_event_from_result

logger = logging.getLogger(__name__)


# Lazy-encode the prompt for telemetry on non-SemKNN routes. Uses the same
# process-wide encoder cache as the SemKNN router, so first call pays the
# ~130 MB BGE-small download / load and subsequent calls are ~10 ms on CPU.
#
# The encoder backs onto fastembed (core dep) and falls back to
# sentence-transformers if the user installed [remote-semknn]. If BOTH are
# missing, that's a broken install — we log loudly the first time and
# return (None, None) so the caller skips the event rather than shipping
# a partial payload that can't be used for SemKNN retraining.
_ENCODER_WARNED_ONCE = False


def _encode_for_telemetry(prompt: str) -> tuple[list[float] | None, str | None]:
    global _ENCODER_WARNED_ONCE
    try:
        enc = _load_encoder(DEFAULT_BGE_MODEL)
    except ImportError as e:
        if not _ENCODER_WARNED_ONCE:
            _ENCODER_WARNED_ONCE = True
            logger.warning(
                "telemetry: encoder unavailable (%s) — telemetry events "
                "will be SKIPPED because they would carry no embedding, "
                "which is the SemKNN retraining signal. Reinstall agentcodec "
                "to pick up the fastembed core dep, or set "
                "AGENTCODEC_TELEMETRY=0 to silence this.",
                e,
            )
        return None, None
    except Exception as e:
        logger.debug("telemetry: encoder load failed: %r", e)
        return None, None
    try:
        return enc.encode(prompt), DEFAULT_BGE_MODEL
    except Exception as e:
        logger.debug("telemetry: prompt encode failed: %r", e)
        return None, None


# ---------------------------------------------------------------------------
# Construction-time helpers
# ---------------------------------------------------------------------------

def _resolve_thinking_for_channel(thinking) -> bool | dict | str | None:
    """Coerce a ModelConfig.thinking value into the form AgentChannel accepts."""
    if thinking is None:
        return None
    if isinstance(thinking, bool) or thinking == "auto":
        return thinking
    if isinstance(thinking, ThinkingConfig):
        return {"enabled": thinking.enabled, "budget_tokens": thinking.budget_tokens}
    return thinking  # already a dict


def _model_to_channel_kwargs(m: ModelConfig) -> dict[str, Any]:
    """Translate a ModelConfig block into AgentChannel kwargs."""
    kwargs: dict[str, Any] = {
        "model": m.model,
        "temperature": m.temperature,
        "max_tokens": m.max_tokens,
        "base_url": m.base_url,
        "api_key": m.api_key,
        "extra_body": dict(m.extra_body) if m.extra_body else None,
        "category_temperatures": dict(m.category_temperatures) if m.category_temperatures else None,
        "thinking": _resolve_thinking_for_channel(m.thinking),
    }
    if m.request_timeout_s is not None:
        kwargs["timeout_s"] = m.request_timeout_s
    if m.system_prompt is not None:
        kwargs["system_prompt"] = m.system_prompt
    if m.cost_per_1m is not None:
        kwargs["cost_per_1m"] = (m.cost_per_1m.input, m.cost_per_1m.output)
    return kwargs


def _emit_startup_summary(cfg: LibraryConfig) -> list[dict[str, Any]]:
    """Log the per-channel pricing + thinking summary at construction time.

    Returns the list of warnings, also stored on the module for later
    inspection.
    """
    # Pricing summary
    pricing_input = []
    for m in cfg.models:
        entry = {"model": m.model, "cost_per_1m": None}
        if m.cost_per_1m is not None:
            entry["cost_per_1m"] = (m.cost_per_1m.input, m.cost_per_1m.output)
        elif cfg.cost_overrides and m.model in cfg.cost_overrides:
            cp = cfg.cost_overrides[m.model]
            entry["cost_per_1m"] = (cp.input, cp.output)
        pricing_input.append(entry)
    rows, warnings = summarize_pricing(pricing_input)

    logger.info("=" * 60)
    logger.info("[agentcodec] Channel pricing tiers")
    logger.info("=" * 60)
    for r in rows:
        logger.info(
            f"  {r['model']:<40s} ${r['rate_input_per_1m']:>6.2f}/${r['rate_output_per_1m']:>6.2f} per 1M  "
            f"({r['tier']})"
        )

    # Thinking summary
    logger.info("[agentcodec] Thinking-capable models in this config:")
    thinking_capable_disabled: list[str] = []
    for m in cfg.models:
        capable = _is_thinking_model(m.model)
        if not capable:
            continue
        explicitly_set = m.thinking is not None and m.thinking != "auto"
        if isinstance(m.thinking, ThinkingConfig):
            enabled = m.thinking.enabled
            budget = m.thinking.budget_tokens
        elif m.thinking is True:
            enabled = True
            budget = None
        else:
            enabled = False
            budget = None
        budget_str = f" budget_tokens={budget}" if budget else ""
        state = "ENABLED" if enabled else "DISABLED (default)"
        logger.info(f"  - {m.model}: thinking-capable=YES → {state}{budget_str}")
        if not enabled and not explicitly_set:
            thinking_capable_disabled.append(m.model)

    if thinking_capable_disabled:
        # No "[agentcodec]" prefix here — the emit loop below adds exactly one.
        notice = (
            f"Note: {len(thinking_capable_disabled)} thinking-capable "
            f"model(s) running with thinking DISABLED (the default). Set "
            f"`thinking: true` per model to enable. Disabled: "
            f"{', '.join(thinking_capable_disabled)}"
        )
        warnings.append(notice)

    # Single emission point so every warning is logged once with one prefix.
    for w in warnings:
        logger.warning(f"[agentcodec] {w}")

    return [{"message": w, "code": "startup_pricing", "severity": "warn"} for w in warnings]


# ---------------------------------------------------------------------------
# ReliabilityModule
# ---------------------------------------------------------------------------

class ReliabilityModule:
    """The library entry point. One instance per deployment configuration."""

    def __init__(self, config: LibraryConfig) -> None:
        self.config = config
        self._startup_warnings = _emit_startup_summary(config)

        # Build channels.
        self.channels: dict[str, AgentChannel] = {}
        for m in config.models:
            ch = AgentChannel(**_model_to_channel_kwargs(m))
            self.channels[m.model] = ch

        # Build judge.
        self.scorer = self._build_scorer(config.judge)

        # Build optional critic channel.
        self.critic_channel: AgentChannel | None = None
        if config.critic and not config.critic.same:
            self.critic_channel = self._build_critic(config.critic)
        # When critic.same is True, the critic falls back to the primary
        # channel at dispatch time inside HARQService / TurboDecoder.

        # Build the router. The factory needs the full config so the remote
        # SemKNN router can derive its user_config fingerprint from the
        # configured channels.
        self.router = build_router(config)

        # Auto-category classifier (cheap, no LLM call).
        self._classifier = AutoCategoryClassifier()

        # Pre-build the dispatch context that's reused across calls.
        self._ctx = self._build_dispatch_context()

        # Anonymous telemetry. Master kill switch is env AGENTCODEC_TELEMETRY=0.
        self.telemetry = self._build_telemetry()

    # ----- Construction helpers -----

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReliabilityModule:
        """Load from a YAML file. Eager-validates the config."""
        cfg = LibraryConfig.from_yaml(path)
        return cls(cfg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReliabilityModule:
        cfg = LibraryConfig.from_dict(data)
        return cls(cfg)

    @classmethod
    def from_preset(
        cls,
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
    ) -> ReliabilityModule:
        """One-line constructor for a fixed/routed strategy on a single model.

        The compat shims (``agentcodec.openai.OpenAI(reliability="harq_ir")``,
        etc.) call this under the hood. See :func:`agentcodec.presets.build_preset_config`
        for the full kwarg matrix.

        Example::

            mod = ReliabilityModule.from_preset(
                "harq_ir", model="gpt-4o-mini", api_key="sk-...",
                max_rounds=6,
            )
        """
        from .presets import build_preset_config
        data = build_preset_config(
            name,
            model=model, models=models,
            api_key=api_key, base_url=base_url,
            judge=judge, critic=critic,
            extras=extras, temperature=temperature,
            **technique_params,
        )
        return cls.from_dict(data)

    def _build_scorer(self, jcfg: JudgeConfig) -> QualityScorer:
        eb = dict(jcfg.extra_body) if jcfg.extra_body else None
        # Translate judge thinking via the same helper used for channels
        # by going through a transient AgentChannel-style construction
        # of the extra_body dict.
        if jcfg.thinking is not None:
            tcfg = AgentChannel._normalize_thinking_config(
                _resolve_thinking_for_channel(jcfg.thinking)
            )
            eb = AgentChannel._translate_thinking_to_extra_body(
                tcfg, jcfg.model, eb,
            )
        scorer = QualityScorer(
            judge_model=jcfg.model,
            base_url=jcfg.base_url,
            api_key=jcfg.api_key,
            extra_body=eb,
            score_strategy=self.config.score_strategy,
        )
        # Per-judge cost override: stash on the underlying judge channel.
        if jcfg.cost_per_1m is not None:
            scorer.judge.cost_per_1m = (jcfg.cost_per_1m.input, jcfg.cost_per_1m.output)
        # Power-user knob: replace the built-in scoring system prompt. Rarely
        # needed because QualityScorer ships with a tuned evaluator prompt
        # already; primarily here so the compat shims can carry a global
        # `system=` value uniformly across the channel pool + judge.
        if jcfg.system_prompt is not None:
            scorer.judge.system_prompt = jcfg.system_prompt
        return scorer

    def _build_critic(self, ccfg: CriticConfig) -> AgentChannel:
        thinking_arg = _resolve_thinking_for_channel(ccfg.thinking)
        kwargs: dict[str, Any] = {
            "model": ccfg.model,
            "temperature": 0.2,
            "base_url": ccfg.base_url,
            "api_key": ccfg.api_key,
            "thinking": thinking_arg,
        }
        if ccfg.cost_per_1m is not None:
            kwargs["cost_per_1m"] = (ccfg.cost_per_1m.input, ccfg.cost_per_1m.output)
        return AgentChannel(**kwargs)

    def _build_dispatch_context(self) -> DispatchContext:
        d = self.config.defaults
        ctx = DispatchContext(
            channels=self.channels,
            scorer=self.scorer,
            critic_channel=self.critic_channel,
            soft_normalization={
                "enabled": d.soft_normalization.enabled,
                "T_logprob": d.soft_normalization.T_logprob,
                "T_judge": d.soft_normalization.T_judge,
                "T_verbal_100": d.soft_normalization.T_verbal_100,
            },
            cisc={
                "csi_source": d.cisc.csi_source,
                "softmax_temperature": d.cisc.softmax_temperature,
                "num_samples": d.cisc.num_samples,
            },
            early_exit=d.early_exit,
        )

        # Wire any inline ACM tables / SemKNN router into the dispatch ctx,
        # so techniques="acm" / "acm_learned" pick them up.
        if isinstance(self.router, ACMTableRouter):
            ctx.acm_table = self._build_acm_profiles(self.router.table)
            if self.router.category_tables:
                ctx.acm_category_tables = {
                    cat: self._build_acm_profiles(entries)
                    for cat, entries in self.router.category_tables.items()
                }
        if isinstance(self.router, LinearRouter):
            # Pre-load the underlying RouterWeights so the dispatcher's
            # "acm_learned" technique can reuse it without re-reading the JSON.
            # The remote SemKNN router has no local q-matrix to share —
            # it always routes to a leaf technique, never to "acm_learned".
            ctx.acm_learned_router = self.router._weights
            # Allow per-technique knob overrides through `strategy.dispatch`.
            if isinstance(self.config.strategy, RoutedStrategy):
                ctx.acm_learned_dispatch_defaults = self.config.strategy.dispatch
                ctx.dispatch_overrides = self.config.strategy.dispatch or {}
        elif isinstance(self.config.strategy, RoutedStrategy):
            ctx.dispatch_overrides = self.config.strategy.dispatch or {}
        elif isinstance(self.config.strategy, FixedStrategy):
            if self.config.strategy.params:
                ctx.dispatch_overrides = {
                    self.config.strategy.technique: self.config.strategy.params
                }
        return ctx

    def _build_telemetry(self) -> Telemetry:
        """Construct the per-module telemetry client.

        Resolution order for the endpoint:
            1. YAML  `telemetry.endpoint`             (explicit)
            2. SemKNN backend URL + `/telemetry`       (when routing remote)
            3. Hardcoded public collector              (default for all other routers)
        """
        from ._endpoints import DEFAULT_TELEMETRY_ENDPOINT
        tb = self.config.telemetry
        if isinstance(self.router, RemoteSemKNNRouter):
            fallback = self.router.server_url.rstrip("/") + "/telemetry"
        else:
            fallback = DEFAULT_TELEMETRY_ENDPOINT
        tcfg = TelemetryConfig(
            enabled=tb.enabled,
            endpoint=tb.endpoint or fallback,
            quiet_notice=tb.quiet_notice,
            flush_interval_s=tb.flush_interval_s,
            queue_max=tb.queue_max,
            batch_max=tb.batch_max,
            timeout_s=tb.timeout_s,
        )
        from . import __version__
        return Telemetry(tcfg, client_version=__version__)

    def _record_telemetry(
        self,
        *,
        result: ReliabilityResult,
        decision: Any,
        task: TaskItem,
        error_type: str | None = None,
    ) -> None:
        """Enqueue one telemetry event after a .run/.stream call.

        SemKNN routes are skipped here: the /route call already shipped the
        embedding + lambda + user_config + chosen technique to the backend,
        which records its own request-side event. A separate /telemetry POST
        with the same data would be redundant.

        For non-SemKNN routers (`fixed`, `acm_table`, `acm_linear`), we
        lazy-encode the prompt client-side with BGE-small via the same
        encoder used by SemKNN (fastembed by default, ~10 ms CPU). If
        telemetry is disabled the encoder is never touched.

        Catches every exception so a buggy event can never break a request.
        """
        try:
            if not self.telemetry.enabled:
                # Disabled → no embedding work for non-server routers.
                return
            if isinstance(self.router, RemoteSemKNNRouter):
                # Recorded server-side from /route; nothing to send here.
                return

            embedding, bge_model = _encode_for_telemetry(task.prompt)
            if embedding is None:
                return

            extra = getattr(decision, "extra", None) or {}
            # Cache the channel-pool fingerprint; config is immutable post-init.
            if not hasattr(self, "_user_cfg_cache"):
                try:
                    self._user_cfg_cache = _derive_user_config(self.config)
                except Exception:
                    self._user_cfg_cache = None

            payload = build_event_from_result(
                result=result,
                routing_extra=extra,
                router_type=getattr(decision, "router_type", "unknown"),
                user_config=self._user_cfg_cache,
                lambda_=extra.get("lambda"),
                embedding=embedding,
                bge_model=bge_model,
                task_category=task.category.value
                    if hasattr(task.category, "value")
                    else str(task.category),
                error_type=error_type,
            )
            self.telemetry.record(payload)
        except Exception as e:
            # Never let telemetry bookkeeping affect the caller.
            logger.debug("telemetry: record failed: %r", e)

    @staticmethod
    def _build_acm_profiles(entries: list[dict[str, Any]] | None) -> list[Any]:
        if not entries:
            return []
        from .techniques.acm import ACMProfile
        out: list[Any] = []
        for e in entries:
            out.append(ACMProfile(
                name=e.get("name", ""),
                difficulty_range=tuple(e.get("difficulty_range", [0.0, 1.0])),
                model=e.get("model", ""),
                technique=e["technique"],
                code_rate=e.get("code_rate", 1.0),
                num_branches=e.get("num_branches", 1),
                max_rounds=e.get("max_rounds", 1),
                estimated_cost_multiplier=e.get("estimated_cost_multiplier", 1.0),
            ))
        return out

    # ----- Public API -----

    def run(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | TaskCategory | None = None,
        reference: str | None = None,
        return_trace: bool = False,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        score_mode: str | None = None,
    ) -> ReliabilityResult:
        """Run the configured strategy. Synchronous.

        Two input shapes are supported, exactly one of which must be given:

        * ``prompt="..."`` — plain string (back-compat path). Optional
          ``system="..."`` adds a system message; ``tools=``, ``stop=``,
          etc. apply as usual.
        * ``messages=[{"role": ..., "content": ...}, ...]`` — full OpenAI-shaped
          conversation including system / prior turns / tool results.

        Args:
            prompt:           the user task text (mutually exclusive with messages=)
            messages:         OpenAI-shaped chat history (mutually exclusive with prompt=)
            system:           optional system-message override
            tools:            OpenAI-shaped tool definitions
            tool_choice:      "auto" | "none" | "required" | {"type":"function","function":{"name":...}}
            response_format:  e.g. {"type": "json_object"} or a JSON-schema dict
            stop:             stop sequence(s)
            seed:             deterministic-decoding seed (where supported)
            top_p:            nucleus sampling cutoff
            temperature:      per-call override for sampling temperature
            max_tokens:       per-call override for the response length budget
            category:         task category. Pass None for auto-classify.
            reference:        optional ground-truth answer for the judge.
            return_trace:     when True, fully populate ``result.trace``.
            task_id:          optional opaque ID for log correlation.
            metadata:         optional dict carried through to the trace.
            score_mode:       "exact_match" / "exact_letter" / "yes_no" / etc.
        """
        prompt_for_task, request = self._build_request_and_prompt(
            prompt, messages, system, tools, tool_choice, response_format,
            stop, seed, top_p, temperature, max_tokens,
        )
        # Building a TaskItem and walking the same dispatcher the benchmark
        # uses keeps logic in one place — every technique gets free re-use.
        task = self._build_task(
            prompt_for_task, category, reference, task_id, metadata, score_mode,
            request=request,
        )

        wall_t0 = time.time()
        decision = self.router.choose(task)

        try:
            run = dispatch(decision.chosen, task, self._ctx)
            error = None
        except Exception as e:
            if self.config.defaults.on_error == "fallback_baseline":
                logger.error(
                    f"Technique {decision.chosen!r} failed: {e!r} — "
                    f"falling back to baseline as configured."
                )
                run = dispatch("baseline", task, self._ctx)
                error = f"primary technique failed: {e!r}"
            else:
                raise

        # Collect the judge LLM-call outputs onto the run. Without this,
        # `return_trace=True` produces a trace with role="channel" entries
        # but no role="judge" entries, even though the judge clearly ran
        # (per-channel `quality_score` is populated). The benchmark runner
        # does this at runner.py; the public API path needs it too.
        run.judge_outputs = self.scorer.collect_judge_outputs()

        wall_clock = time.time() - wall_t0

        result = build_result_from_run(
            run,
            technique_used=decision.chosen if not error else "baseline",
            wall_clock_s=wall_clock,
            return_trace=return_trace,
            error=error,
            routing_info=decision.to_dict(),
            category_info={
                "value": task.category.value,
                "source": "user" if category is not None else (
                    "auto" if self.config.defaults.category == "auto" else "default"
                ),
            },
            extra_warnings=self._startup_warnings if return_trace else None,
        )
        self._record_telemetry(
            result=result, decision=decision, task=task,
            error_type="primary_failed_fallback_baseline" if error else None,
        )
        return result

    async def arun(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | TaskCategory | None = None,
        reference: str | None = None,
        return_trace: bool = False,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        score_mode: str | None = None,
        executor: Any | None = None,
    ) -> ReliabilityResult:
        """Async wrapper around `run()`.

        Accepts the same kwargs as :meth:`run`; offloads the (currently
        synchronous) techniques to a thread pool. Pass a bounded
        ``ThreadPoolExecutor`` for high-concurrency hosts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: self.run(
                prompt,
                messages=messages,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                stop=stop,
                seed=seed,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                category=category,
                reference=reference,
                return_trace=return_trace,
                task_id=task_id,
                metadata=metadata,
                score_mode=score_mode,
            ),
        )

    def stream(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | TaskCategory | None = None,
        reference: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        score_mode: str | None = None,
    ) -> Iterator[Event]:
        """Sync streaming iterator.

        Default: yields ProgressEvents at each major stage (route, dispatch,
        complete) and a single terminal FinalEvent with the result. Token-
        level streaming is not yet wired through every technique — see
        README §streaming for the per-technique table and roadmap.

        Accepts the same input shapes as :meth:`run` (``prompt=`` or
        ``messages=``, with optional ``system=``, ``tools=``, etc.).
        """
        prompt_for_task, request = self._build_request_and_prompt(
            prompt, messages, system, tools, tool_choice, response_format,
            stop, seed, top_p, temperature, max_tokens,
        )
        task = self._build_task(
            prompt_for_task, category, reference, task_id, metadata, score_mode,
            request=request,
        )
        wall_t0 = time.time()

        # Stage 1: routing
        decision = self.router.choose(task)
        yield ProgressEvent(
            stage="route",
            detail={
                "chosen": decision.chosen,
                "router_type": decision.router_type,
                "confidence": decision.confidence,
                "candidates_score": decision.candidates_score,
            },
            elapsed_s=time.time() - wall_t0,
        )

        # Stage 2: dispatch (synchronous; emits no mid-stream events yet —
        # that requires per-technique stream() variants which is roadmap)
        yield ProgressEvent(
            stage="dispatch_start",
            detail={"technique": decision.chosen},
            elapsed_s=time.time() - wall_t0,
        )

        try:
            run = dispatch(decision.chosen, task, self._ctx)
            error = None
        except Exception as e:
            if self.config.defaults.on_error == "fallback_baseline":
                yield WarningEvent(
                    message=f"primary technique failed: {e!r}",
                    code="fallback_to_baseline",
                    severity="error",
                )
                run = dispatch("baseline", task, self._ctx)
                error = f"primary technique failed: {e!r}"
            else:
                raise

        # See ``ReliabilityModule.run``: the per-call scorer outputs need to
        # be explicitly collected onto the run so that role="judge" entries
        # appear in `result.trace["calls"]` when `return_trace=True`.
        run.judge_outputs = self.scorer.collect_judge_outputs()

        yield ProgressEvent(
            stage="dispatch_complete",
            detail={
                "technique": decision.chosen,
                "rounds": getattr(run, "rounds", 0),
                "final_quality": getattr(run, "final_quality", None),
            },
            elapsed_s=time.time() - wall_t0,
        )

        # Final: emit the full text as a single TokenEvent (chunk_format=delta
        # semantics: this is one-shot since we don't have mid-stream tokens
        # yet) plus the FinalEvent.
        text = getattr(run, "combined_output", "") or (
            run.individual_outputs[0].text if run.individual_outputs else ""
        )
        if text:
            yield TokenEvent(
                text=text, role="answer",
                model=run.individual_outputs[0].model if run.individual_outputs else "",
                call_id="final",
                cumulative=False,
            )

        wall_clock = time.time() - wall_t0
        result = build_result_from_run(
            run,
            technique_used=decision.chosen if not error else "baseline",
            wall_clock_s=wall_clock,
            return_trace=True,  # streaming consumers usually want the trace
            error=error,
            routing_info=decision.to_dict(),
            category_info={
                "value": task.category.value,
                "source": "user" if category is not None else (
                    "auto" if self.config.defaults.category == "auto" else "default"
                ),
            },
        )
        self._record_telemetry(
            result=result, decision=decision, task=task,
            error_type="primary_failed_fallback_baseline" if error else None,
        )
        yield FinalEvent(result=result)

    async def astream(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        system: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: str | Sequence[str] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        category: str | TaskCategory | None = None,
        reference: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        score_mode: str | None = None,
        executor: Any | None = None,
    ) -> AsyncIterator[Event]:
        """Async streaming iterator.

        Drives :func:`agentcodec.dispatch.adispatch` natively in the running
        event loop. For techniques with a native ``_<name>_astream`` impl
        (currently ``baseline`` and ``harq_ir``), per-call token deltas and
        round-by-round progress events flow through unbuffered. For other
        techniques, the sync ``dispatch()`` runs in an executor and only the
        terminal ``FinalEvent`` is emitted.

        Accepts the same input kwargs as :meth:`run` / :meth:`stream`.

        ``executor`` is accepted for backward compatibility but is no longer
        used on the native path — left in the signature so existing callers
        don't break.
        """
        from .dispatch import ReliabilityRun, adispatch

        prompt_for_task, request = self._build_request_and_prompt(
            prompt, messages, system, tools, tool_choice, response_format,
            stop, seed, top_p, temperature, max_tokens,
        )
        task = self._build_task(
            prompt_for_task, category, reference, task_id, metadata, score_mode,
            request=request,
        )
        wall_t0 = time.time()

        # Stage 1: routing (still sync; very fast)
        decision = self.router.choose(task)
        yield ProgressEvent(
            stage="route",
            detail={
                "chosen": decision.chosen,
                "router_type": decision.router_type,
                "confidence": decision.confidence,
                "candidates_score": decision.candidates_score,
            },
            elapsed_s=time.time() - wall_t0,
        )

        yield ProgressEvent(
            stage="dispatch_start",
            detail={"technique": decision.chosen},
            elapsed_s=time.time() - wall_t0,
        )

        run = None
        error: str | None = None
        try:
            async for frame in adispatch(decision.chosen, task, self._ctx):
                if isinstance(frame, ReliabilityRun):
                    run = frame
                else:
                    yield frame
        except Exception as e:
            if self.config.defaults.on_error == "fallback_baseline":
                yield WarningEvent(
                    message=f"primary technique failed: {e!r}",
                    code="fallback_to_baseline",
                    severity="error",
                )
                async for frame in adispatch("baseline", task, self._ctx):
                    if isinstance(frame, ReliabilityRun):
                        run = frame
                    else:
                        yield frame
                error = f"primary technique failed: {e!r}"
            else:
                raise

        if run is None:
            raise RuntimeError(
                "adispatch produced no ReliabilityRun — technique impl is broken"
            )

        # See ``ReliabilityModule.run``: the per-call scorer outputs need to
        # be explicitly collected onto the run so that role="judge" entries
        # appear in `result.trace["calls"]` when `return_trace=True`.
        run.judge_outputs = self.scorer.collect_judge_outputs()

        yield ProgressEvent(
            stage="dispatch_complete",
            detail={
                "technique": decision.chosen,
                "rounds": getattr(run, "rounds", 0),
                "final_quality": getattr(run, "final_quality", None),
            },
            elapsed_s=time.time() - wall_t0,
        )

        wall_clock = time.time() - wall_t0
        result = build_result_from_run(
            run,
            technique_used=decision.chosen if not error else "baseline",
            wall_clock_s=wall_clock,
            return_trace=True,
            error=error,
            routing_info=decision.to_dict(),
            category_info={
                "value": task.category.value,
                "source": "user" if category is not None else (
                    "auto" if self.config.defaults.category == "auto" else "default"
                ),
            },
        )
        self._record_telemetry(
            result=result, decision=decision, task=task,
            error_type="primary_failed_fallback_baseline" if error else None,
        )
        yield FinalEvent(result=result)

    # ----- Lifecycle -----

    def close(self) -> None:
        """Flush in-flight telemetry events and shut down the worker.

        Idempotent. The Telemetry worker is also drained at process exit
        via atexit, so this is only needed when you want a hard guarantee
        — e.g. in a test, or before forking.
        """
        try:
            self.telemetry.flush(timeout_s=2.0)
        finally:
            self.telemetry.shutdown()

    def __enter__(self) -> ReliabilityModule:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- Internals -----

    def _build_request_and_prompt(
        self,
        prompt: str | None,
        messages: Sequence[Mapping[str, Any] | Message] | None,
        system: str | None,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        response_format: Mapping[str, Any] | None,
        stop: str | Sequence[str] | None,
        seed: int | None,
        top_p: float | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ChatRequest]:
        """Normalize ``run(prompt=...)`` or ``run(messages=...)`` to a (prompt, request) pair.

        Exactly one of ``prompt`` or ``messages`` must be supplied. We return
        the back-compat string view of the user turn alongside the rich
        ChatRequest so legacy code paths (telemetry, score_mode inference,
        TaskItem.prompt) keep working unchanged.
        """
        if (prompt is None) == (messages is None):
            raise ValueError(
                "specify exactly one of prompt= or messages= when calling "
                "ReliabilityModule.run()/arun()/stream()/astream()"
            )
        if messages is not None:
            request = ChatRequest.from_openai_messages(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                stop=stop,
                seed=seed,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if system is not None:
                request = request.with_system(system)
            try:
                prompt_for_task = request.last_user_text
            except ValueError:
                raise ValueError(
                    "messages= must include at least one user turn"
                ) from None
        else:
            request = ChatRequest.from_prompt(
                prompt,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                stop=stop,
                seed=seed,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prompt_for_task = prompt
        return prompt_for_task, request

    def _build_task(
        self,
        prompt: str,
        category: str | TaskCategory | None,
        reference: str | None,
        task_id: str | None,
        metadata: dict[str, Any] | None,
        score_mode: str | None = None,
        request: ChatRequest | None = None,
    ) -> TaskItem:
        cat = self._resolve_category(prompt, category)
        task = TaskItem(
            id=task_id or f"adhoc-{int(time.time() * 1000)}",
            category=cat,
            prompt=prompt,
            request=request,
            reference=reference,
            metadata=dict(metadata or {}),
        )
        # Explicit score_mode wins over metadata-based inference done in
        # TaskItem.__post_init__. Validated against the registry so a typo
        # fails loudly instead of silently falling back to the judge.
        if score_mode is not None:
            from .scoring import _DISPATCH as _SCORERS
            valid = set(_SCORERS) | {"judge"}
            if score_mode not in valid:
                raise ValueError(
                    f"Unknown score_mode {score_mode!r}. Valid values: {sorted(valid)}"
                )
            task.score_mode = score_mode
        return task

    def _resolve_category(
        self,
        prompt: str,
        category: str | TaskCategory | None,
    ) -> TaskCategory:
        if isinstance(category, TaskCategory):
            return category
        if isinstance(category, str):
            return TaskCategory(category)
        # category=None — fall back to defaults
        default = self.config.defaults.category
        if default == "auto":
            return self._classifier.classify(prompt)
        return TaskCategory(default)
