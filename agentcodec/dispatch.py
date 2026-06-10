"""
Technique dispatcher — single source of truth for "given a technique name +
a TaskItem, run it and return a ReliabilityRun."

Mirrors the dispatch logic in `runner.BenchmarkRunner._run_technique` but
with dependencies passed in explicitly (no `self`). Both the benchmark
runner and the library facade (`agentcodec.api.ReliabilityModule`) use this.

Keeping the logic in one place means the library inherits every technique
the benchmark supports, including the soft-output variants and the
prior-method baselines.

This module also hosts the **async-streaming** counterparts (``adispatch``
and per-technique ``_<name>_astream`` generators). Each ``astream``
generator yields :class:`Event` instances (TokenEvent / ProgressEvent /
WarningEvent) as the technique progresses, and the LAST frame it yields
is the completed :class:`ReliabilityRun` itself (NOT an Event). Callers
distinguish by ``isinstance(frame, ReliabilityRun)``. Techniques without
a native async impl fall back to running sync ``dispatch()`` in an
executor — the loop stays unblocked but no mid-stream events are emitted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .channel import AgentChannel, QualityScorer
from .messages import ChannelChunk, ChannelDone
from .models import (
    AgentOutput,
    CombiningStrategy,
    HARQMode,
    ReliabilityRun,
    TaskItem,
)
from .results import ProgressEvent, TokenEvent, WarningEvent
from .techniques import (
    ACMLearnedRouter,
    ACMRouter,
    BestOfNBaseline,
    ChainOfVerificationBaseline,
    CISCBaseline,
    DiversityEnsemble,
    DiversityMRCDiscreteN,
    FECService,
    FountainDecoder,
    HARQService,
    MixtureOfAgentsBaseline,
    SelectionCombiningN,
    SelfConsistencyBaseline,
    SelfRefineBaseline,
    SoftACMRouter,
    SoftDiversityMRC,
    SoftDiversityMRCDiscreteN,
    SoftFountainDecoder,
    TurboDecoder,
    WeightedBoNBaseline,
)

logger = logging.getLogger(__name__)


@dataclass
class DispatchContext:
    """All the dependencies a technique needs.

    Built once at module construction; passed into every dispatch call.
    """
    channels: dict[str, AgentChannel]      # name -> channel
    scorer: QualityScorer
    critic_channel: AgentChannel | None = None
    soft_normalization: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "T_logprob": 0.1, "T_judge": 0.5, "T_verbal_100": 8.0,
    })
    cisc: dict[str, Any] = field(default_factory=dict)
    early_exit: bool = False
    # Optional inline ACM tables (when not loading from a cache file).
    acm_table: list[Any] | None = None
    acm_category_tables: dict[str, list[Any]] | None = None
    # Optional pre-loaded learned/SemKNN router weights or path.
    acm_learned_router: Any | None = None
    acm_learned_dispatch_defaults: dict[str, dict[str, Any]] | None = None
    # Per-technique knob overrides (max_rounds, num_branches, code_rate, etc.).
    dispatch_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def _knob(ctx: DispatchContext, technique: str, key: str, default: Any) -> Any:
    """Look up a per-technique knob override or fall back to the default."""
    overrides = ctx.dispatch_overrides.get(technique, {})
    return overrides.get(key, default)


def dispatch(
    technique: str,
    task: TaskItem,
    ctx: DispatchContext,
) -> ReliabilityRun:
    """Run `technique` on `task` using the dependencies in `ctx`.

    This is the single source of truth for technique routing. The benchmark
    runner and the library facade both call it.
    """
    channels_list = list(ctx.channels.values())
    if not channels_list:
        raise ValueError("DispatchContext.channels is empty")
    primary = channels_list[0]

    if technique == "baseline":
        return _baseline(task, primary, ctx.scorer)

    # ---- Prior-method baselines ----

    if technique == "self_consistency":
        return SelfConsistencyBaseline(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
        ).run(task)

    if technique == "self_refine":
        return SelfRefineBaseline(
            channel=primary,
            scorer=ctx.scorer,
            max_rounds=_knob(ctx, technique, "max_rounds", 3),
        ).run(task)

    if technique == "chain_of_verification":
        return ChainOfVerificationBaseline(
            channel=primary,
            scorer=ctx.scorer,
            num_verification_questions=_knob(ctx, technique, "num_verification_questions", 3),
        ).run(task)

    if technique == "diversity_sc_N":
        return SelectionCombiningN(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
        ).run(task)

    if technique == "diversity_mrc_discrete_N":
        sn = ctx.soft_normalization
        return DiversityMRCDiscreteN(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
            softmax_normalize=sn["enabled"],
            softmax_temperature=sn["T_judge"],
        ).run(task)

    if technique == "best_of_n":
        return BestOfNBaseline(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
        ).run(task)

    if technique == "weighted_bon":
        return WeightedBoNBaseline(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
        ).run(task)

    if technique == "cisc":
        cisc_cfg = ctx.cisc or {}
        return CISCBaseline(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=cisc_cfg.get("num_samples", 5),
            voter=primary,
            csi_source=cisc_cfg.get("csi_source", "verbal_100"),
            softmax_temperature=cisc_cfg.get("softmax_temperature"),
        ).run(task)

    if technique == "mixture_of_agents":
        return MixtureOfAgentsBaseline(
            channels=channels_list,
            scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            aggregator=primary,
        ).run(task)

    # ---- Soft-output techniques (must be checked before generic diversity_*) ----

    if technique == "diversity_mrc_soft":
        sn = ctx.soft_normalization
        return SoftDiversityMRC(
            channels=channels_list, scorer=ctx.scorer,
            softmax_normalize=sn["enabled"],
            softmax_temperature=sn["T_logprob"],
        ).run(task, synthesizer=primary)

    if technique == "diversity_mrc_discrete_N_soft":
        sn = ctx.soft_normalization
        return SoftDiversityMRCDiscreteN(
            channels=channels_list, scorer=ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
            softmax_normalize=sn["enabled"],
            softmax_temperature=sn["T_logprob"],
        ).run(task)

    if technique == "fountain_soft":
        return SoftFountainDecoder(
            channels=channels_list, scorer=ctx.scorer,
            max_samples=_knob(ctx, technique, "max_samples", 8),
        ).run(task)

    if technique == "acm_soft":
        return SoftACMRouter(channels=ctx.channels, scorer=ctx.scorer).run(task)

    # ---- Generic diversity_* ----

    if technique.startswith("diversity_"):
        if technique == "diversity_spatial":
            svc = DiversityEnsemble(
                channels=channels_list, scorer=ctx.scorer,
                combining=CombiningStrategy.MRC,
            )
            return svc.run(task, synthesizer=primary)
        if technique == "diversity_frequency":
            from .techniques.diversity import DEFAULT_PROMPT_VARIANTS
            svc = DiversityEnsemble(
                channels=[primary], scorer=ctx.scorer,
                combining=CombiningStrategy.MRC,
                prompt_variants=DEFAULT_PROMPT_VARIANTS,
            )
            return svc.run(task, synthesizer=primary)
        if technique == "diversity_time":
            svc = DiversityEnsemble(
                channels=[primary], scorer=ctx.scorer,
                combining=CombiningStrategy.MRC,
                temperature_spread=[0.3, 0.5, 0.7, 0.9],
            )
            return svc.run(task, synthesizer=primary)
        # diversity_sc / diversity_mrc / diversity_egc
        strategy = CombiningStrategy(technique.split("_", 1)[1])
        return DiversityEnsemble(
            channels=channels_list, scorer=ctx.scorer, combining=strategy,
        ).run(task, synthesizer=primary)

    # ---- HARQ ----

    if technique.startswith("harq_"):
        mode = HARQMode(technique.split("_", 1)[1])
        return HARQService(
            channel=primary, scorer=ctx.scorer, mode=mode,
            max_rounds=_knob(ctx, technique, "max_rounds", 5),
            critic_channel=ctx.critic_channel,
            early_exit=ctx.early_exit,
        ).run(task)

    # ---- Turbo ----

    if technique == "turbo":
        return TurboDecoder(
            generator=primary,
            critic=ctx.critic_channel,
            scorer=ctx.scorer,
            max_iterations=_knob(ctx, technique, "max_iterations", 5),
            quality_threshold=0.85,
            early_exit=ctx.early_exit,
        ).run(task)

    # ---- Fountain ----

    if technique == "fountain":
        sn = ctx.soft_normalization
        return FountainDecoder(
            channels=channels_list, scorer=ctx.scorer,
            max_samples=_knob(ctx, technique, "max_samples", 8),
            softmax_normalize=sn["enabled"],
            softmax_temperature=sn["T_judge"],
        ).run(task)

    # ---- FEC ----

    if technique.startswith("fec_"):
        rate = float(_knob(ctx, technique, "code_rate", technique.split("_", 1)[1]))
        return FECService(channel=primary, scorer=ctx.scorer, code_rate=rate).run(task)

    # ---- ACM (hand-coded table) ----

    if technique == "acm":
        return ACMRouter(
            channels=ctx.channels, scorer=ctx.scorer,
            acm_table=ctx.acm_table,
            critic_channel=ctx.critic_channel,
            category_tables=ctx.acm_category_tables,
        ).run(task)

    # ---- ACM-learned (linear or SemKNN) ----

    if technique == "acm_learned":
        if ctx.acm_learned_router is None:
            raise ValueError(
                "Technique 'acm_learned' requires `acm_learned_router` "
                "to be loaded into DispatchContext (a RouterWeights instance "
                "or a path to a linear router weights JSON)."
            )
        svc = ACMLearnedRouter(
            channels=ctx.channels, scorer=ctx.scorer,
            router_weights=ctx.acm_learned_router,
            dispatch_defaults=ctx.acm_learned_dispatch_defaults,
            critic_channel=ctx.critic_channel,
        )
        run = svc.run(task)
        run.config["routed_tagged_technique"] = run.technique
        run.technique = "acm_learned"
        return run

    raise ValueError(f"Unknown technique: {technique}")


def _baseline(task: TaskItem, channel: AgentChannel, scorer: QualityScorer) -> ReliabilityRun:
    """Single-channel uncoded transmission. Same logic as
    BenchmarkRunner._run_baseline.
    """
    run = ReliabilityRun(
        task_id=task.id,
        task_category=task.category.value if hasattr(task.category, "value") else str(task.category),
        technique="baseline",
        config={"model": channel.model},
    )
    out = channel.transmit(task.request)
    out.quality_score = scorer.score(task.prompt, out.text, reference=task.reference, task=task)
    run.individual_outputs = [out]
    run.combined_output = out.text
    run.final_quality = out.quality_score
    run.compute_metrics()
    return run


# Public list of every technique name `dispatch()` accepts. Used by the
# config validator to flag typos in `strategy.technique`.
KNOWN_TECHNIQUES: tuple[str, ...] = (
    "baseline",
    "self_consistency", "self_refine", "chain_of_verification",
    "best_of_n", "weighted_bon", "cisc", "mixture_of_agents",
    "diversity_sc", "diversity_mrc", "diversity_egc",
    "diversity_sc_N", "diversity_mrc_discrete_N",
    # NOTE: "diversity_spatial" is intentionally omitted — it's an alias for
    # diversity_mrc (same MRC combining; see dispatch branch above) and is not
    # counted as a distinct public technique. It still dispatches correctly.
    "diversity_frequency", "diversity_time",
    "diversity_mrc_soft", "diversity_mrc_discrete_N_soft",
    "harq_cc", "harq_ir",
    "turbo",
    "fountain", "fountain_soft",
    "fec_0.75", "fec_0.50", "fec_0.33", "fec_0.25",
    "acm", "acm_soft", "acm_learned",
)


# ---------------------------------------------------------------------------
# Async-streaming dispatcher
# ---------------------------------------------------------------------------
#
# Each technique that supports streaming exposes a top-level coroutine
# generator named ``_<name>_astream``. The protocol is:
#
#   * Yields :class:`Event` instances (``TokenEvent``, ``ProgressEvent``,
#     ``WarningEvent``) as the technique progresses.
#   * Yields exactly one :class:`ReliabilityRun` as the **final** frame —
#     callers detect this with ``isinstance(frame, ReliabilityRun)``.
#
# TokenEvent role taxonomy (see README §streaming for the canonical list):
#   "answer"        — text the model intends as the user-facing answer
#   "thinking"      — model-internal reasoning (provider channel)
#   "draft"         — intermediate answer in a refine/HARQ loop
#   "critique"      — critic's evaluation between rounds
#   "verification"  — verification Q/A in chain-of-verification
#   "candidate"     — one branch in a parallel-N technique
#   "synthesis"     — aggregator output
#   "judge"         — quality scorer output (rarely streamed)
#
# Native vs fallback: techniques without ``_<name>_astream`` fall through
# to executor-wrapped ``dispatch()``. Those produce no mid-stream events
# but still complete correctly.

# Techniques that have a native async-streaming implementation. Anything
# else falls back to the sync ``dispatch()`` running in an executor.
#
# Parallel-branch techniques (diversity_*, BoN family, MoA, self_consistency)
# execute their N branches concurrently via ``asyncio.gather`` — that gives
# the real async speedup. Their synthesizer/aggregator/vote call currently
# emits a SINGLE TokenEvent rather than per-token deltas — extracting the
# synthesis-prompt builders so the synth call can use ``atransmit_stream``
# is a v0.5 follow-up (see roadmap).
_ASTREAM_TECHNIQUES: frozenset[str] = frozenset({
    "baseline",
    "harq_ir",
    "harq_cc",
    "turbo",
    "self_refine",
    "chain_of_verification",
    "self_consistency",
    "best_of_n",
    "weighted_bon",
    "mixture_of_agents",
    "diversity_sc",
    "diversity_mrc",
    "diversity_egc",
    "diversity_spatial",
    "diversity_frequency",
    "diversity_time",
    "diversity_sc_N",
    "diversity_mrc_discrete_N",
    "fountain",
    "fountain_soft",
    "cisc",
    "fec_0.75",
    "fec_0.50",
    "fec_0.33",
    "fec_0.25",
    "diversity_mrc_soft",
    "diversity_mrc_discrete_N_soft",
    "acm",
})


async def _astream_one_call(
    channel: AgentChannel,
    prompt_or_request: Any,
    *,
    role: str,
    call_id: str,
    temperature: float | None = None,
) -> AsyncIterator[Any]:
    """Drive one ``channel.atransmit_stream`` call; forward each
    :class:`ChannelChunk` as a :class:`TokenEvent` and yield the final
    :class:`AgentOutput` as the LAST frame (NOT an Event).

    Used by per-technique astream impls to dry up the channel-call boilerplate.
    The protocol mirrors the technique-astream contract one level down:
    iterate, distinguish the terminal frame by ``isinstance(frame, AgentOutput)``.

    ``role`` is the TokenEvent role to use for "answer"-style chunks. Thinking
    and tool_call chunks always get their own canonical roles (``"thinking"``,
    forwarded as ``"answer"`` for tool_call since there's no dedicated role).
    """
    out: AgentOutput | None = None
    async for frame in channel.atransmit_stream(
        prompt_or_request, temperature=temperature,
    ):
        if isinstance(frame, ChannelChunk):
            if frame.role == "thinking":
                yield TokenEvent(
                    text=frame.text, role="thinking",
                    model=channel.model, call_id=call_id,
                )
            else:
                yield TokenEvent(
                    text=frame.text, role=role,
                    model=channel.model, call_id=call_id,
                )
        elif isinstance(frame, ChannelDone):
            out = frame.output
    if out is None:
        raise RuntimeError(
            f"channel atransmit_stream produced no ChannelDone (call_id={call_id})"
        )
    yield out


def _record_run_baseline(
    task: TaskItem, out: AgentOutput, model: str,
) -> ReliabilityRun:
    """Build the ReliabilityRun for a single-call baseline. Shared between
    sync ``_baseline`` and async ``_baseline_astream`` so the trace shape
    stays identical regardless of which path the caller used."""
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="baseline",
        config={"model": model},
    )
    run.individual_outputs = [out]
    run.combined_output = out.text
    run.final_quality = out.quality_score
    run.compute_metrics()
    return run


async def _baseline_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
) -> AsyncIterator[Any]:
    """Single-call baseline as an async stream.

    Forwards every channel chunk as a :class:`TokenEvent` (role ``"answer"``
    or ``"thinking"`` per the channel-level taxonomy), then scores the
    answer in an executor (the sync judge call would block the loop),
    then yields the completed :class:`ReliabilityRun` as the final frame.
    """
    yield ProgressEvent(
        stage="channel_start",
        detail={"model": channel.model, "technique": "baseline"},
    )
    out: AgentOutput | None = None
    call_id = "baseline:0"
    async for frame in channel.atransmit_stream(task.request):
        if isinstance(frame, ChannelChunk):
            yield TokenEvent(
                text=frame.text,
                role=frame.role if frame.role != "tool_call" else "answer",
                model=channel.model,
                call_id=call_id,
            )
        elif isinstance(frame, ChannelDone):
            out = frame.output
    if out is None:
        raise RuntimeError("baseline astream did not receive ChannelDone")

    loop = asyncio.get_running_loop()
    out.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, out.text, reference=task.reference, task=task,
        ),
    )
    yield ProgressEvent(
        stage="channel_complete",
        detail={"quality": out.quality_score, "model": channel.model},
    )
    yield _record_run_baseline(task, out, channel.model)


async def _harq_ir_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    max_rounds: int = 5,
    quality_threshold: float = 0.85,
    critic_channel: AgentChannel | None = None,
    early_exit: bool = False,
) -> AsyncIterator[Any]:
    """HARQ-IR with per-round streaming.

    Round 1 streams the initial attempt as ``role="answer"`` (it might be
    the final answer if quality threshold is hit). Subsequent rounds
    stream the critic's structured critique as ``role="critique"`` and the
    refined attempt as ``role="draft"``. The final accepted draft becomes
    the run's ``combined_output``.

    Communication-faithful loop: independent attempt scoring on round 1,
    comparative scoring (round vs prior best) on rounds ≥ 2. Regression
    protection — only advance ``best_output`` when the refinement score
    is ≥ the prior best. Mirrors :class:`HARQService._run_incremental_redundancy`
    without re-importing the existing service (we re-use its helpers).
    """
    # Late-import the service so its helper methods are reusable here
    # without circular-importing techniques. The class has stateless
    # ``_get_structured_critique`` / ``_build_correction_prompt`` /
    # ``_score_plateau`` so we just instantiate it as a helper container.
    from .techniques.harq import HARQService, _parse_structured_critique

    helper = HARQService(
        channel=channel, scorer=scorer, mode=HARQMode.IR,
        max_rounds=max_rounds, quality_threshold=quality_threshold,
        critic_channel=critic_channel, early_exit=early_exit,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="harq_ir",
        config={
            "mode": "ir", "max_rounds": max_rounds,
            "quality_threshold": quality_threshold,
            "model": channel.model,
            "critic_model": helper.critic.model,
            "early_exit": early_exit,
        },
    )

    # ---- Round 1 ----
    yield ProgressEvent(
        stage="round_start",
        detail={"round": 1, "technique": "harq_ir", "model": channel.model},
    )
    initial: AgentOutput | None = None
    async for frame in channel.atransmit_stream(task.request):
        if isinstance(frame, ChannelChunk):
            yield TokenEvent(
                text=frame.text,
                role=frame.role if frame.role != "tool_call" else "answer",
                model=channel.model,
                call_id="harq_ir:round1:gen",
            )
        elif isinstance(frame, ChannelDone):
            initial = frame.output
    if initial is None:
        raise RuntimeError("HARQ-IR round 1 produced no ChannelDone")

    initial.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, initial.text, reference=task.reference, task=task,
        ),
    )
    outputs: list[AgentOutput] = [initial]
    run.rounds = 1
    best_output = initial
    current_output = initial
    score_history = [initial.quality_score]
    yield ProgressEvent(
        stage="round_complete",
        detail={"round": 1, "quality": initial.quality_score},
    )

    if initial.quality_score >= quality_threshold:
        run.individual_outputs = outputs
        run.combined_output = initial.text
        run.final_quality = initial.quality_score
        run.compute_metrics()
        yield run
        return

    overhead: list[AgentOutput] = []
    accumulated_corrections: list[dict] = []

    for round_num in range(2, max_rounds + 1):
        # Critic pass (sync helper running in executor — keeps the loop unblocked)
        yield ProgressEvent(
            stage="critic_start",
            detail={"round": round_num, "model": helper.critic.model},
        )
        critique_text, critique_output = await loop.run_in_executor(
            None,
            lambda: helper._get_structured_critique(
                task.prompt, current_output.text, task.reference,
                current_score=current_output.quality_score,
                prior_corrections=accumulated_corrections,
            ),
        )
        overhead.append(critique_output)
        # Surface the critique text as a TokenEvent for hosts that want to
        # render "the critic said X" in their UI. Single emission per critic
        # call (sync helper doesn't stream); future native-async critique
        # would yield deltas here.
        yield TokenEvent(
            text=critique_text,
            role="critique",
            model=helper.critic.model,
            call_id=f"harq_ir:round{round_num}:critic",
        )

        critique_data = _parse_structured_critique(critique_text)
        new_issues = critique_data["issues"]
        no_new_issues = (
            len(new_issues) == 0
            or ((len(new_issues) == 1 and not new_issues[0].get("raw"))
            and not new_issues[0].get("quote"))
        )
        if no_new_issues:
            if early_exit and current_output.quality_score >= quality_threshold * 0.9:
                yield WarningEvent(
                    code="harq_ir_converged",
                    message=f"round {round_num}: critic found no issues, quality near threshold",
                    severity="info",
                )
                break
            if early_exit:
                yield WarningEvent(
                    code="harq_ir_early_exit_low_quality",
                    message=f"round {round_num}: critic found no issues but quality is low",
                    severity="warn",
                )
                break
            if not new_issues:
                new_issues = [{"raw": critique_data["raw_text"]}]

        if early_exit and helper._score_plateau(score_history):
            yield WarningEvent(
                code="harq_ir_plateau",
                message=f"round {round_num}: score plateau detected",
                severity="info",
            )
            break

        structured_issues = [c for c in new_issues if "quote" in c]
        accumulated_corrections.extend(structured_issues)
        refinement_prompt = helper._build_correction_prompt(
            task.prompt, best_output.text, new_issues,
        )

        # Generator pass — stream the refinement as a "draft"
        yield ProgressEvent(
            stage="refine_start",
            detail={"round": round_num, "model": channel.model},
        )
        refined: AgentOutput | None = None
        async for frame in channel.atransmit_stream(refinement_prompt):
            if isinstance(frame, ChannelChunk):
                yield TokenEvent(
                    text=frame.text,
                    role="draft" if frame.role == "answer" else frame.role,
                    model=channel.model,
                    call_id=f"harq_ir:round{round_num}:gen",
                )
            elif isinstance(frame, ChannelDone):
                refined = frame.output
        if refined is None:
            raise RuntimeError(f"HARQ-IR round {round_num} produced no ChannelDone")

        # Comparative scoring
        refined.quality_score = await loop.run_in_executor(
            None,
            lambda: scorer.score_comparative(
                task.prompt,
                candidate=refined.text,
                baseline=current_output.text,
                baseline_score=current_output.quality_score,
                reference=task.reference,
            ),
        )
        outputs.append(refined)
        run.rounds = round_num
        score_history.append(refined.quality_score)
        yield ProgressEvent(
            stage="round_complete",
            detail={"round": round_num, "quality": refined.quality_score},
        )

        if refined.quality_score >= best_output.quality_score:
            current_output = refined
            best_output = refined
        else:
            current_output = best_output

        if best_output.quality_score >= quality_threshold:
            break

    run.individual_outputs = outputs
    run.overhead_outputs = overhead
    run.combined_output = best_output.text
    run.final_quality = best_output.quality_score
    run.compute_metrics()
    yield run


async def _harq_cc_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    max_rounds: int = 5,
    quality_threshold: float = 0.85,
    critic_channel: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """HARQ-CC: N independent attempts, soft-combine if no early exit.

    Each round streams the attempt as ``role="candidate"`` (they're
    independent retransmissions, not refinements). If any attempt clears
    the quality threshold, that one becomes the answer. Otherwise the
    critic synthesizes them via chase-combining — streamed as
    ``role="synthesis"``.
    """
    from .techniques.harq import HARQService

    helper = HARQService(
        channel=channel, scorer=scorer, mode=HARQMode.CC,
        max_rounds=max_rounds, quality_threshold=quality_threshold,
        critic_channel=critic_channel,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="harq_cc",
        config={
            "mode": "cc", "max_rounds": max_rounds,
            "quality_threshold": quality_threshold,
            "model": channel.model,
            "critic_model": helper.critic.model,
        },
    )

    outputs: list[AgentOutput] = []
    for round_num in range(1, max_rounds + 1):
        yield ProgressEvent(
            stage="round_start",
            detail={"round": round_num, "technique": "harq_cc", "model": channel.model},
        )
        round_out: AgentOutput | None = None
        async for frame in _astream_one_call(
            channel, task.request,
            role="candidate", call_id=f"harq_cc:round{round_num}",
        ):
            if isinstance(frame, AgentOutput):
                round_out = frame
            else:
                yield frame
        if round_out is None:
            raise RuntimeError(f"HARQ-CC round {round_num} produced no AgentOutput")
        round_out.quality_score = await loop.run_in_executor(
            None,
            lambda o=round_out: scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            ),
        )
        outputs.append(round_out)
        run.rounds = round_num
        yield ProgressEvent(
            stage="round_complete",
            detail={"round": round_num, "quality": round_out.quality_score},
        )
        if round_out.quality_score >= quality_threshold:
            run.individual_outputs = outputs
            run.combined_output = round_out.text
            run.final_quality = round_out.quality_score
            run.compute_metrics()
            yield run
            return

    # All rounds done without threshold → chase-combine via critic.
    yield ProgressEvent(
        stage="combine_start",
        detail={"model": helper.critic.model, "n_attempts": len(outputs)},
    )
    combined_text, combine_output = await loop.run_in_executor(
        None, lambda: helper._combine_cc(outputs, task.prompt),
    )
    # Single synthesis emission (sync combine helper doesn't stream yet).
    yield TokenEvent(
        text=combined_text, role="synthesis",
        model=helper.critic.model, call_id="harq_cc:combine",
    )

    best_output = max(outputs, key=lambda o: o.quality_score)
    final_quality = await loop.run_in_executor(
        None,
        lambda: scorer.score_comparative(
            task.prompt,
            candidate=combined_text,
            baseline=best_output.text,
            baseline_score=best_output.quality_score,
            reference=task.reference,
        ),
    )

    run.individual_outputs = outputs
    run.overhead_outputs = [combine_output]
    run.combined_output = combined_text
    run.final_quality = final_quality
    run.compute_metrics()
    yield run


async def _turbo_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    max_iterations: int = 5,
    quality_threshold: float = 0.85,
    critic_channel: AgentChannel | None = None,
    early_exit: bool = False,
) -> AsyncIterator[Any]:
    """Turbo: generator/critic exchange with extrinsic scaling.

    Iteration 0 streams the generator's draft as ``role="answer"`` (might
    become the final answer if it hits threshold). Subsequent iterations
    emit ``role="critique"`` for the critic and ``role="draft"`` for the
    generator's refinement. Mirrors :class:`TurboDecoder.run`.
    """
    from .techniques.turbo import TurboDecoder, _parse_structured_critique

    helper = TurboDecoder(
        generator=channel, scorer=scorer,
        critic=critic_channel or channel,
        max_iterations=max_iterations, quality_threshold=quality_threshold,
        early_exit=early_exit,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="turbo",
        config={
            "generator": channel.model,
            "critic": helper.critic.model,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "early_exit": early_exit,
            "extrinsic_scale": helper.extrinsic_scale,
        },
    )

    # ---- Iteration 0: initial generation ----
    yield ProgressEvent(
        stage="iter_start",
        detail={"iteration": 0, "technique": "turbo", "model": channel.model},
    )
    gen_out: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, task.request,
        role="answer", call_id="turbo:iter0:gen",
    ):
        if isinstance(frame, AgentOutput):
            gen_out = frame
        else:
            yield frame
    if gen_out is None:
        raise RuntimeError("Turbo iteration 0 produced no AgentOutput")
    gen_out.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, gen_out.text, reference=task.reference, task=task,
        ),
    )
    all_outputs: list[AgentOutput] = [gen_out]
    score_history = [gen_out.quality_score]
    best_text = gen_out.text
    best_score = gen_out.quality_score
    run.rounds = 1
    yield ProgressEvent(
        stage="iter_complete",
        detail={"iteration": 0, "quality": best_score},
    )

    if best_score >= quality_threshold:
        run.individual_outputs = all_outputs
        run.combined_output = best_text
        run.final_quality = best_score
        run.compute_metrics()
        yield run
        return

    overhead: list[AgentOutput] = []
    accumulated_corrections: list[dict] = []
    current_alpha = helper.extrinsic_scale
    consecutive_regressions = 0

    for iteration in range(1, max_iterations):
        # --- Critic pass (sync) ---
        yield ProgressEvent(
            stage="critic_start",
            detail={"iteration": iteration, "model": helper.critic.model},
        )
        extrinsic_text, critic_output = await loop.run_in_executor(
            None,
            lambda: helper._critic_pass(
                task, best_text, best_score, accumulated_corrections,
                iteration=iteration,
            ),
        )
        overhead.append(critic_output)
        yield TokenEvent(
            text=extrinsic_text, role="critique",
            model=helper.critic.model,
            call_id=f"turbo:iter{iteration}:critic",
        )

        critique_data = _parse_structured_critique(extrinsic_text)
        new_issues = critique_data["issues"]
        no_new_issues = (
            len(new_issues) == 0
            or (len(new_issues) == 1
                and not new_issues[0].get("raw")
                and not new_issues[0].get("quote"))
        )
        if no_new_issues:
            if best_score >= quality_threshold * 0.9:
                yield WarningEvent(
                    code="turbo_converged",
                    message=f"iter {iteration}: critic found no issues, quality near threshold",
                    severity="info",
                )
                break
            if early_exit:
                yield WarningEvent(
                    code="turbo_early_exit_low_quality",
                    message=f"iter {iteration}: critic found no issues but quality is low",
                    severity="warn",
                )
                break
            if not new_issues:
                new_issues = [{"raw": critique_data["raw_text"]}]

        if early_exit and helper._score_plateau(score_history):
            yield WarningEvent(
                code="turbo_plateau",
                message=f"iter {iteration}: score plateau detected",
                severity="info",
            )
            break

        scaled_issues = helper._scale_extrinsic(new_issues, alpha=current_alpha)

        # --- Generator pass (stream as draft) ---
        yield ProgressEvent(
            stage="refine_start",
            detail={"iteration": iteration, "model": channel.model,
                    "n_corrections": len(scaled_issues)},
        )
        # The generator_pass helper builds a prompt + calls transmit. We
        # replicate the prompt building inline so we can stream the call.
        refinement_prompt = helper._build_correction_prompt(
            task.prompt, best_text, scaled_issues,
        )
        refined: AgentOutput | None = None
        async for frame in _astream_one_call(
            channel, refinement_prompt,
            role="draft", call_id=f"turbo:iter{iteration}:gen",
        ):
            if isinstance(frame, AgentOutput):
                refined = frame
            else:
                yield frame
        if refined is None:
            raise RuntimeError(f"Turbo iter {iteration} gen produced no AgentOutput")

        refined.quality_score = await loop.run_in_executor(
            None,
            lambda: scorer.score_comparative(
                task.prompt,
                candidate=refined.text,
                baseline=best_text,
                baseline_score=best_score,
                reference=task.reference,
            ),
        )
        all_outputs.append(refined)
        run.rounds = iteration + 1
        score_history.append(refined.quality_score)
        yield ProgressEvent(
            stage="iter_complete",
            detail={"iteration": iteration, "quality": refined.quality_score},
        )

        if refined.quality_score >= best_score:
            best_text = refined.text
            best_score = refined.quality_score
            accumulated_corrections.extend(
                c for c in scaled_issues if "quote" in c
            )
            consecutive_regressions = 0
            current_alpha = min(helper.extrinsic_scale, current_alpha * 1.2)
        else:
            consecutive_regressions += 1
            current_alpha = max(0.1, current_alpha * 0.5)

        if best_score >= quality_threshold:
            break
        if consecutive_regressions >= 2:
            yield WarningEvent(
                code="turbo_diverging",
                message=f"iter {iteration}: 2 consecutive regressions — stopping",
                severity="warn",
            )
            break

    run.individual_outputs = all_outputs
    run.overhead_outputs = overhead
    run.combined_output = best_text
    run.final_quality = best_score
    run.compute_metrics()
    yield run


async def _self_refine_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    max_rounds: int = 3,
    early_stop: bool = True,
) -> AsyncIterator[Any]:
    """Self-Refine (Madaan et al. 2023): draft → critique → refine loop.

    Streams round 1's draft as ``role="answer"`` (becomes final if loop
    terminates early), each critique as ``role="critique"``, and each
    refined draft as ``role="draft"``. The paper's STOP|CONTINUE prefix
    on the critique exits the loop when the critic signals no further
    improvement (when ``early_stop=True``).
    """
    from .techniques.baselines import SelfRefineBaseline

    helper = SelfRefineBaseline(
        channel=channel, scorer=scorer,
        max_rounds=max_rounds, early_stop=early_stop,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="self_refine",
        config={
            "max_rounds": max_rounds, "model": channel.model,
            "early_stop": early_stop,
        },
    )

    yield ProgressEvent(
        stage="draft_start",
        detail={"round": 0, "model": channel.model},
    )
    current: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, task.request, role="answer", call_id="self_refine:draft0",
        temperature=0.7,
    ):
        if isinstance(frame, AgentOutput):
            current = frame
        else:
            yield frame
    if current is None:
        raise RuntimeError("Self-Refine initial draft produced no AgentOutput")
    current.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, current.text, reference=task.reference, task=task,
        ),
    )
    history: list[AgentOutput] = [current]

    stop_reason = "max_rounds"
    for k in range(1, max_rounds + 1):
        # --- Critique pass ---
        if early_stop:
            critique_prompt = (
                f"## Original Task\n{task.prompt}\n\n"
                f"## Current Answer\n{current.text}\n\n"
                f"Critique this answer. On the FIRST line of your "
                f"response, write exactly one word: 'STOP' if the "
                f"answer is already good and no further improvement "
                f"is needed, or 'CONTINUE' if there is room for "
                f"improvement. Then on the following lines, point "
                f"out any problems, errors, inaccuracies, missing "
                f"details, or unclear reasoning. Be specific and "
                f"constructive."
            )
        else:
            critique_prompt = (
                f"## Original Task\n{task.prompt}\n\n"
                f"## Current Answer\n{current.text}\n\n"
                f"Critique this answer. Point out any problems, errors, "
                f"inaccuracies, missing details, or unclear reasoning. "
                f"Be specific and constructive."
            )
        yield ProgressEvent(
            stage="critic_start",
            detail={"round": k, "model": channel.model},
        )
        critique: AgentOutput | None = None
        async for frame in _astream_one_call(
            channel, critique_prompt,
            role="critique", call_id=f"self_refine:critique{k}",
            temperature=0.5,
        ):
            if isinstance(frame, AgentOutput):
                critique = frame
            else:
                yield frame
        if critique is None:
            raise RuntimeError(f"Self-Refine critique {k} produced no AgentOutput")
        history.append(critique)
        run.rounds = k

        if early_stop and helper._is_stop_signal(critique.text):
            stop_reason = "model_signal"
            break

        # --- Revise pass ---
        revise_prompt = (
            f"## Original Task\n{task.prompt}\n\n"
            f"## Previous Answer\n{current.text}\n\n"
            f"## Critique\n{critique.text}\n\n"
            f"Produce an improved answer that addresses the critique. "
            f"Return only the improved answer."
        )
        yield ProgressEvent(
            stage="refine_start",
            detail={"round": k, "model": channel.model},
        )
        revised: AgentOutput | None = None
        async for frame in _astream_one_call(
            channel, revise_prompt,
            role="draft", call_id=f"self_refine:revise{k}",
            temperature=0.7,
        ):
            if isinstance(frame, AgentOutput):
                revised = frame
            else:
                yield frame
        if revised is None:
            raise RuntimeError(f"Self-Refine revise {k} produced no AgentOutput")
        revised.quality_score = await loop.run_in_executor(
            None,
            lambda: scorer.score(
                task.prompt, revised.text, reference=task.reference, task=task,
            ),
        )
        history.append(revised)
        current = revised
        yield ProgressEvent(
            stage="round_complete",
            detail={"round": k, "quality": current.quality_score},
        )

    run.config["stop_reason"] = stop_reason
    run.individual_outputs = history
    run.combined_output = current.text
    run.final_quality = current.quality_score or 0.0
    run.compute_metrics()
    yield run


async def _chain_of_verification_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    num_verification_questions: int = 3,
) -> AsyncIterator[Any]:
    """Chain-of-Verification (CoVe, Factored variant): draft → plan → N×verify → revise.

    Streams the baseline as ``role="answer"`` (early draft), the plan as
    ``role="verification"`` (the questions are the start of the
    verification chain), each verification answer as ``role="verification"``,
    and the final revised answer as ``role="answer"``. Mirrors
    :class:`ChainOfVerificationBaseline.run`.
    """
    import re as _re

    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="chain_of_verification",
        config={
            "num_verification_questions": num_verification_questions,
            "model": channel.model,
        },
    )

    # Step 1: baseline answer
    yield ProgressEvent(
        stage="baseline_start",
        detail={"step": 1, "model": channel.model},
    )
    baseline: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, task.request,
        role="answer", call_id="cove:baseline", temperature=0.7,
    ):
        if isinstance(frame, AgentOutput):
            baseline = frame
        else:
            yield frame
    if baseline is None:
        raise RuntimeError("CoVe baseline produced no AgentOutput")
    baseline.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, baseline.text, reference=task.reference, task=task,
        ),
    )

    # Step 2: plan verification questions
    plan_prompt = (
        f"## Task\n{task.prompt}\n\n"
        f"## Draft Answer\n{baseline.text}\n\n"
        f"Generate exactly {num_verification_questions} short verification "
        f"questions that, if answered independently, would test the "
        f"correctness of the draft answer's factual claims. Number them "
        f"1), 2), 3). Produce only the numbered questions, one per line."
    )
    yield ProgressEvent(
        stage="plan_start",
        detail={"step": 2, "model": channel.model,
                "n_questions": num_verification_questions},
    )
    plan: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, plan_prompt,
        role="verification", call_id="cove:plan", temperature=0.3,
    ):
        if isinstance(frame, AgentOutput):
            plan = frame
        else:
            yield frame
    if plan is None:
        raise RuntimeError("CoVe plan produced no AgentOutput")

    # Parse questions
    questions: list[str] = []
    for line in plan.text.splitlines():
        line = line.strip()
        m = _re.match(r"^\s*\d+[\.\)\:]\s*(.+)$", line)
        if m:
            questions.append(m.group(1).strip())
    questions = questions[:num_verification_questions]
    if not questions:
        questions = [
            ln.strip() for ln in plan.text.splitlines() if ln.strip()
        ][:num_verification_questions]

    # Step 3: execute verifications independently
    verifications: list[AgentOutput] = []
    for i, q in enumerate(questions):
        yield ProgressEvent(
            stage="verify_start",
            detail={"step": 3, "question_index": i, "model": channel.model},
        )
        v: AgentOutput | None = None
        async for frame in _astream_one_call(
            channel,
            f"Answer this question as accurately as possible:\n\n{q}",
            role="verification", call_id=f"cove:verify{i}", temperature=0.3,
        ):
            if isinstance(frame, AgentOutput):
                v = frame
            else:
                yield frame
        if v is None:
            raise RuntimeError(f"CoVe verification {i} produced no AgentOutput")
        verifications.append(v)

    # Step 4: revise
    verifs_text = "\n\n".join(
        f"Q: {q}\nA: {v.text}" for q, v in zip(questions, verifications, strict=False)
    )
    final_prompt = (
        f"## Task\n{task.prompt}\n\n"
        f"## Draft Answer\n{baseline.text}\n\n"
        f"## Verifications\n{verifs_text}\n\n"
        f"Produce a revised answer that is consistent with the "
        f"verification answers above. If any verification contradicts "
        f"the draft, correct the draft. Return only the final answer."
    )
    yield ProgressEvent(
        stage="revise_start",
        detail={"step": 4, "model": channel.model},
    )
    final: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, final_prompt,
        role="answer", call_id="cove:revise", temperature=0.5,
    ):
        if isinstance(frame, AgentOutput):
            final = frame
        else:
            yield frame
    if final is None:
        raise RuntimeError("CoVe revise produced no AgentOutput")
    final.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, final.text, reference=task.reference, task=task,
        ),
    )

    run.individual_outputs = [baseline, final]
    run.overhead_outputs = [plan, *verifications]
    run.rounds = 1 + num_verification_questions
    run.combined_output = final.text
    run.final_quality = final.quality_score or 0.0
    run.compute_metrics()
    yield run


# ---------------------------------------------------------------------------
# Parallel-branch astream impls
# ---------------------------------------------------------------------------
#
# Branches run concurrently via ``asyncio.gather``. The branches themselves
# don't stream tokens to the user — emitting interleaved deltas from N
# concurrent calls would be unreadable. Instead we emit ``ProgressEvent``
# per branch completion. The synthesizer / aggregator / vote call still runs
# in an executor and lands as a single TokenEvent for now.


async def _gather_branches(
    branch_calls: list[tuple[AgentChannel, Any, str, float | None]],
) -> list[AgentOutput]:
    """Run multiple :meth:`AgentChannel.atransmit` calls concurrently.

    Each entry is ``(channel, prompt_or_request, prompt_variant, temperature)``.
    Returns the list of :class:`AgentOutput` in the SAME ORDER as the
    input — branch indices stay aligned with the call list.
    """
    async def _one(ch: AgentChannel, prompt: Any, v_name: str, temp: float | None):
        return await ch.atransmit(
            prompt, prompt_variant=v_name, temperature=temp,
        )
    return await asyncio.gather(*[
        _one(ch, p, v, t) for ch, p, v, t in branch_calls
    ])


async def _diversity_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    combining: CombiningStrategy,
    prompt_variants: dict[str, str] | None = None,
    temperature_spread: list[float] | None = None,
    synthesizer: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """Diversity ensemble (SC / MRC / EGC + spatial / frequency / time variants).

    Runs all (channel × variant × temperature) branches CONCURRENTLY via
    ``asyncio.gather``, then batch-scores them, then combines via the
    requested strategy. For SC the best branch's text becomes the answer;
    for MRC/EGC the judge (or fallback synthesizer) merges them.

    Branches don't stream tokens (interleaved deltas would be unreadable);
    they emit ``ProgressEvent`` per branch start/complete. The synthesis
    call lands as a single ``role="synthesis"`` TokenEvent.
    """
    from .techniques.diversity import DiversityEnsemble

    helper = DiversityEnsemble(
        channels=channels, scorer=scorer, combining=combining,
        prompt_variants=prompt_variants,
        temperature_spread=temperature_spread,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique=f"diversity_{combining.value}",
        config={
            "num_channels": len(channels),
            "combining": combining.value,
            "num_prompt_variants": len(helper.prompt_variants),
            "temperature_spread": temperature_spread,
        },
    )

    # Build the branch specs: one per (channel, variant, temperature).
    branch_specs: list[tuple[AgentChannel, str, str, float | None]] = []
    for ch in channels:
        for v_name, v_template in helper.prompt_variants.items():
            prompt_text = v_template.format(prompt=task.prompt)
            if temperature_spread:
                for t in temperature_spread:
                    branch_specs.append((ch, prompt_text, v_name, t))
            else:
                branch_specs.append((ch, prompt_text, v_name, None))

    yield ProgressEvent(
        stage="branches_start",
        detail={
            "n_branches": len(branch_specs),
            "combining": combining.value,
            "technique": run.technique,
        },
    )
    outputs = await _gather_branches(branch_specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model, "variant": out.prompt_variant},
        )

    # Score in batch (sync helper, run in executor).
    await loop.run_in_executor(
        None,
        lambda: scorer.score_batch(
            task.prompt, outputs, reference=task.reference, task=task,
        ),
    )
    yield ProgressEvent(
        stage="branches_scored",
        detail={"scores": [o.quality_score for o in outputs]},
    )

    run.individual_outputs = outputs

    # Combine. Judge as synthesizer (matches sync impl behavior).
    if combining.value != "sc" and hasattr(scorer, "judge"):
        synth_channel = scorer.judge
    else:
        synth_channel = synthesizer or channels[0]

    combined_text, synth_output = await loop.run_in_executor(
        None,
        lambda: helper._combine(outputs, task.prompt, synth_channel),
    )
    run.combined_output = combined_text
    if synth_output is not None:
        run.overhead_outputs = [synth_output]
        yield TokenEvent(
            text=combined_text, role="synthesis",
            model=synth_channel.model, call_id=f"{run.technique}:synth",
        )
    else:
        # SC fallback or SC dominance — best branch text IS the answer.
        yield TokenEvent(
            text=combined_text, role="answer",
            model=(
                max(outputs, key=lambda o: o.quality_score).model
                if outputs else ""
            ),
            call_id=f"{run.technique}:best",
        )

    # Final scoring (matches sync impl).
    best_output = max(outputs, key=lambda o: o.quality_score)
    best_ind = best_output.quality_score
    if combining.value == "sc":
        run.final_quality = best_ind
    elif combined_text == best_output.text:
        run.final_quality = best_ind
    else:
        synth_score = await loop.run_in_executor(
            None,
            lambda: scorer.score_comparative(
                task.prompt,
                candidate=combined_text,
                baseline=best_output.text,
                baseline_score=best_ind,
                reference=task.reference,
            ),
        )
        if synth_score >= best_ind:
            run.final_quality = synth_score
        else:
            # MRC guarantee: combining can't be worse than SC.
            run.combined_output = best_output.text
            run.final_quality = best_ind

    run.compute_metrics()
    yield run


async def _best_of_n_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
) -> AsyncIterator[Any]:
    """Best-of-N (Cobbe 2021 / Lightman 2024): N samples + pick best by judge.

    Samples run concurrently. The winner's text becomes the answer
    (no synthesizer call) and is emitted as ``role="answer"``.
    """
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="best_of_n",
        config={"num_samples": num_samples, "model": channel.model},
    )

    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "best_of_n"},
    )
    specs = [(channel, task.request, "default", 0.7)] * num_samples
    outputs = await _gather_branches(specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    # Score each candidate (matches sync impl's per-sample judge call).
    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)
    yield ProgressEvent(
        stage="branches_scored",
        detail={"scores": [o.quality_score for o in outputs]},
    )

    best = max(outputs, key=lambda o: o.quality_score or 0.0)
    yield TokenEvent(
        text=best.text, role="answer",
        model=best.model, call_id="best_of_n:winner",
    )

    run.individual_outputs = outputs
    run.rounds = num_samples
    run.combined_output = best.text
    run.final_quality = best.quality_score or 0.0
    run.compute_metrics()
    yield run


async def _weighted_bon_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
    voter: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """Weighted Best-of-N (Snell 2024) — free-form voter-LLM aggregation path.

    N samples run concurrently, then a single voter LLM call clusters by
    content and returns the highest-summed-judge-score cluster's top sample.
    The implementation here uses the voter-LLM (free-form) path; the
    canonical exact-match extractor path is omitted because the library
    doesn't ship task-specific extractors today.
    """
    # NB: we replicate the voter prompt inline (instead of calling
    # WeightedBoNBaseline.run()) so we can stream the voter call via
    # _astream_one_call rather than blocking on a sync transmit.
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="weighted_bon",
        config={
            "num_samples": num_samples, "model": channel.model,
            "voter": (voter or channel).model,
        },
    )

    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "weighted_bon"},
    )
    specs = [(channel, task.request, "default", 0.7)] * num_samples
    outputs = await _gather_branches(specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)
    yield ProgressEvent(
        stage="branches_scored",
        detail={"scores": [o.quality_score for o in outputs]},
    )

    # Voter call — single emission; ideally would stream in v0.5.
    yield ProgressEvent(
        stage="aggregate_start",
        detail={"voter_model": (voter or channel).model},
    )
    # Replicate the voter prompt from sync WeightedBoNBaseline so we can
    # stream it. The actual clustering logic is inside helper.run; we run
    # _just_ the voter portion in an executor for now since the sync
    # helper hard-codes its sample loop. Future v0.5: extract the
    # cluster-call so we can stream it via channel.atransmit_stream.
    sync_voter = voter or channel
    joined = "\n\n".join(
        f"### Sample {i+1} [Quality: {o.quality_score:.2f}]\n{o.text}"
        for i, o in enumerate(outputs)
    )
    vote_prompt = (
        f"## Task\n{task.prompt}\n\n"
        f"## Candidate Answers\n{joined}\n\n"
        f"Identify the answer (by content, not by wording) that is "
        f"supported by the majority of high-quality candidates. Return "
        f"ONLY that answer's text.\n\n"
        f"## Selected answer:"
    )
    vote = await sync_voter.atransmit(vote_prompt, temperature=0.0)
    combined_text = vote.text.strip() or max(
        outputs, key=lambda o: o.quality_score or 0.0,
    ).text

    yield TokenEvent(
        text=combined_text, role="synthesis",
        model=sync_voter.model, call_id="weighted_bon:vote",
    )

    run.individual_outputs = outputs
    run.overhead_outputs = [vote]
    run.rounds = num_samples
    run.combined_output = combined_text
    run.final_quality = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, combined_text, reference=task.reference, task=task,
        ),
    )
    run.compute_metrics()
    yield run


async def _self_consistency_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
    voter: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """Self-Consistency (Wang+2023) — Universal-SC free-form path.

    N samples run concurrently via ``asyncio.gather``, then a voter LLM
    identifies the most representative answer. The exact-match
    ``answer_extractor`` path of the sync impl is omitted here because
    the library doesn't ship task-specific extractors; users wanting
    that path should call the sync technique through ``mod.run()``.
    """
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="self_consistency_llm_voter",
        config={
            "num_samples": num_samples,
            "num_channels": len(channels),
            "aggregation": "llm_voter",
        },
    )

    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "self_consistency"},
    )
    specs = [
        (channels[i % len(channels)], task.request, "default", 0.7)
        for i in range(num_samples)
    ]
    outputs = await _gather_branches(specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)

    sync_voter = voter or channels[0]
    yield ProgressEvent(
        stage="aggregate_start",
        detail={"voter_model": sync_voter.model},
    )
    joined = "\n\n".join(
        f"### Sample {i+1}\n{o.text}" for i, o in enumerate(outputs)
    )
    vote_prompt = (
        f"Below are {num_samples} independent answers to the same task. "
        f"Identify the MOST FREQUENT final answer (majority vote by content, "
        f"not by wording). Return ONLY the majority answer's text, with no "
        f"commentary.\n\n"
        f"## Task\n{task.prompt}\n\n"
        f"## Candidate Answers\n{joined}\n\n"
        f"## Majority answer:"
    )
    vote = await sync_voter.atransmit(vote_prompt, temperature=0.0)
    combined_text = vote.text.strip() or outputs[0].text
    yield TokenEvent(
        text=combined_text, role="synthesis",
        model=sync_voter.model, call_id="self_consistency:vote",
    )

    run.individual_outputs = outputs
    run.overhead_outputs = [vote]
    run.rounds = num_samples
    run.combined_output = combined_text
    run.final_quality = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, combined_text, reference=task.reference, task=task,
        ),
    )
    run.compute_metrics()
    yield run


async def _mixture_of_agents_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
    aggregator: AgentChannel | None = None,
    num_layers: int = 2,
) -> AsyncIterator[Any]:
    """Mixture-of-Agents (Wang+2025): K proposers × L layers + aggregator.

    Layer 1 runs all K proposers concurrently from the original prompt.
    Subsequent layers each run K proposers concurrently on a prompt that
    aggregates the previous layer's outputs. A final aggregator call
    produces the answer.

    ``num_samples`` is interpreted as the proposers-per-layer width K when
    we have ``len(channels) >= num_samples``; otherwise we cycle through
    available channels. Matches the sync impl's K=len(channels) convention
    when channels are not under-provisioned.
    """
    from .techniques.baselines import MixtureOfAgentsBaseline

    helper = MixtureOfAgentsBaseline(
        channels=channels, scorer=scorer,
        num_samples=num_samples, aggregator=aggregator,
        num_layers=num_layers,
    )
    loop = asyncio.get_running_loop()
    K = len(channels)
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="mixture_of_agents",
        config={
            "num_samples": num_samples, "num_channels": K,
            "num_layers": num_layers,
            "aggregator": (aggregator or channels[0]).model,
        },
    )

    # Layer 1: K proposers from the task prompt.
    yield ProgressEvent(
        stage="layer_start",
        detail={"layer": 1, "n_proposers": K, "technique": "mixture_of_agents"},
    )
    specs = [(channels[i], task.request, "default", 0.7) for i in range(K)]
    prev_layer = await _gather_branches(specs)
    for i, out in enumerate(prev_layer):
        yield ProgressEvent(
            stage="proposer_complete",
            detail={"layer": 1, "proposer": i, "model": out.model},
        )

    def _score_layer(layer_outputs: list[AgentOutput]) -> None:
        for o in layer_outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_layer, prev_layer)
    all_proposals = list(prev_layer)

    # Layers 2..L
    for layer in range(2, num_layers + 1):
        agg_prompt = helper._aggregator_prompt(task.prompt, prev_layer)
        yield ProgressEvent(
            stage="layer_start",
            detail={"layer": layer, "n_proposers": K},
        )
        layer_specs = [(channels[i], agg_prompt, "default", 0.7) for i in range(K)]
        next_layer = await _gather_branches(layer_specs)
        for i, out in enumerate(next_layer):
            yield ProgressEvent(
                stage="proposer_complete",
                detail={"layer": layer, "proposer": i, "model": out.model},
            )
        await loop.run_in_executor(None, _score_layer, next_layer)
        all_proposals.extend(next_layer)
        prev_layer = next_layer

    # Final aggregator: stream it via atransmit_stream so the answer
    # actually streams to the user.
    agg_channel = aggregator or channels[0]
    final_prompt = helper._aggregator_prompt(task.prompt, prev_layer)
    yield ProgressEvent(
        stage="aggregate_start",
        detail={"aggregator_model": agg_channel.model},
    )
    synthesis: AgentOutput | None = None
    async for frame in _astream_one_call(
        agg_channel, final_prompt,
        role="answer", call_id="moa:aggregate", temperature=0.7,
    ):
        if isinstance(frame, AgentOutput):
            synthesis = frame
        else:
            yield frame
    if synthesis is None:
        raise RuntimeError("MoA aggregator produced no AgentOutput")

    combined_text = synthesis.text.strip() or prev_layer[0].text
    run.individual_outputs = all_proposals
    run.overhead_outputs = [synthesis]
    run.rounds = len(all_proposals)
    run.combined_output = combined_text
    run.final_quality = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, combined_text, reference=task.reference, task=task,
        ),
    )
    run.compute_metrics()
    yield run


async def _diversity_sc_N_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
) -> AsyncIterator[Any]:
    """Wider-pool Selection Combining: N samples cycled through all channels,
    score each, pick argmax. Like ``best_of_n`` but the candidates cycle
    through multiple models for spatial diversity.
    """
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="diversity_sc_N",
        config={"num_samples": num_samples, "num_channels": len(channels)},
    )

    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "diversity_sc_N"},
    )
    specs = [
        (channels[i % len(channels)], task.request, "default", 0.7)
        for i in range(num_samples)
    ]
    outputs = await _gather_branches(specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)
    yield ProgressEvent(
        stage="branches_scored",
        detail={"scores": [o.quality_score for o in outputs]},
    )

    best = max(outputs, key=lambda o: o.quality_score or 0.0)
    yield TokenEvent(
        text=best.text, role="answer",
        model=best.model, call_id="diversity_sc_N:winner",
    )

    run.individual_outputs = outputs
    run.rounds = num_samples
    run.combined_output = best.text
    run.final_quality = best.quality_score or 0.0
    run.compute_metrics()
    yield run


async def _diversity_mrc_discrete_N_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
    voter: AgentChannel | None = None,
    softmax_normalize: bool = True,
    softmax_temperature: float = 0.5,
) -> AsyncIterator[Any]:
    """Discrete MRC on a multi-model pool: cluster N samples semantically
    via one voter LLM call, then pick the highest-scored sample from the
    highest-summed-weight cluster. Mirrors :class:`DiversityMRCDiscreteN`.
    """
    import json as _json
    import re as _re

    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="diversity_mrc_discrete_N",
        config={
            "num_samples": num_samples,
            "num_channels": len(channels),
            "softmax_normalize": softmax_normalize,
            "softmax_temperature": (
                softmax_temperature if softmax_normalize else None
            ),
        },
    )

    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "diversity_mrc_discrete_N"},
    )
    specs = [
        (channels[i % len(channels)], task.request, "default", 0.7)
        for i in range(num_samples)
    ]
    outputs = await _gather_branches(specs)
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)
    yield ProgressEvent(
        stage="branches_scored",
        detail={"scores": [o.quality_score for o in outputs]},
    )

    sync_voter = voter or channels[0]
    joined = "\n\n".join(
        f"### Sample {i+1}\n{o.text}" for i, o in enumerate(outputs)
    )
    cluster_prompt = (
        f"Below are {num_samples} independent answers to the same task. "
        f"Group them into semantic equivalence classes: two answers belong "
        f"to the same class iff they convey the same final answer or "
        f"conclusion (phrasing differences are ignored).\n\n"
        f"## Task\n{task.prompt}\n\n"
        f"## Samples\n{joined}\n\n"
        f"Return ONLY a JSON array of {num_samples} integers, where "
        f"the i-th integer is the 0-indexed cluster ID of sample (i+1). "
        f"Use the smallest cluster IDs possible (0, 1, 2, ...).\n"
        f"Example for 5 samples: [0, 0, 1, 0, 2]\n\n"
        f"JSON:"
    )
    yield ProgressEvent(
        stage="cluster_start",
        detail={"voter_model": sync_voter.model},
    )
    vote = await sync_voter.atransmit(cluster_prompt, temperature=0.0)

    labels: list[int] | None = None
    try:
        m = _re.search(r"\[[\s\d,]*\]", vote.text)
        if m:
            parsed = _json.loads(m.group(0))
            if isinstance(parsed, list) and len(parsed) == num_samples:
                labels = [int(x) for x in parsed]
    except Exception:
        labels = None
    if labels is None:
        labels = list(range(num_samples))

    raw_weights = [(o.quality_score or 0.0) for o in outputs]
    if softmax_normalize:
        from .techniques.soft import softmax_with_temperature
        weights = softmax_with_temperature(raw_weights, softmax_temperature)
    else:
        weights = raw_weights
    totals: dict[int, float] = {}
    members: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        totals[lbl] = totals.get(lbl, 0.0) + weights[idx]
        members.setdefault(lbl, []).append(idx)
    winning_cluster = max(totals, key=lambda k: totals[k])
    winning_members = members[winning_cluster]
    best_idx = max(winning_members, key=lambda i: raw_weights[i])
    best = outputs[best_idx]

    yield TokenEvent(
        text=best.text, role="synthesis",
        model=best.model,
        call_id=f"diversity_mrc_discrete_N:cluster{winning_cluster}",
    )

    run.individual_outputs = outputs
    run.overhead_outputs = [vote]
    run.rounds = num_samples
    run.config["cluster_labels"] = labels
    run.config["winning_cluster_size"] = len(winning_members)
    run.config["raw_weights"] = [round(w, 4) for w in raw_weights]
    run.config["norm_weights"] = [round(w, 4) for w in weights]
    run.combined_output = best.text
    run.final_quality = best.quality_score or 0.0
    run.compute_metrics()
    yield run


async def _fountain_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    *,
    confidence_threshold: float = 0.8,
    max_samples: int = 8,
    min_samples: int = 3,
) -> AsyncIterator[Any]:
    """Rateless fountain: keep sampling until confidence threshold is hit.

    Sequential (each sample's existence depends on prior confidence), so
    branches can't be gathered concurrently. Streams each sample as
    ``role="candidate"`` and the decoder output as ``role="synthesis"``.
    """
    from .techniques.fountain import FountainDecoder

    helper = FountainDecoder(
        channels=channels, scorer=scorer,
        confidence_threshold=confidence_threshold,
        max_samples=max_samples, min_samples=min_samples,
    )
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="fountain",
        config={
            "confidence_threshold": confidence_threshold,
            "max_samples": max_samples,
            "min_samples": min_samples,
            "num_channels": len(channels),
        },
    )

    outputs: list[AgentOutput] = []
    for sample_num in range(1, max_samples + 1):
        ch = channels[(sample_num - 1) % len(channels)]
        temp = 0.5 + (sample_num % 5) * 0.1
        yield ProgressEvent(
            stage="sample_start",
            detail={"sample": sample_num, "model": ch.model, "temperature": temp},
        )
        out = await ch.atransmit(task.request, temperature=temp)
        out.quality_score = await loop.run_in_executor(
            None,
            lambda o=out: scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            ),
        )
        outputs.append(out)
        run.rounds = sample_num
        # Surface the candidate text as a single TokenEvent so consumers can
        # render incremental "draft i" updates.
        yield TokenEvent(
            text=out.text, role="candidate",
            model=ch.model, call_id=f"fountain:sample{sample_num}",
        )
        yield ProgressEvent(
            stage="sample_complete",
            detail={"sample": sample_num, "quality": out.quality_score},
        )

        if sample_num >= min_samples:
            confidence = helper._estimate_confidence(outputs)
            yield ProgressEvent(
                stage="confidence_check",
                detail={"sample": sample_num, "confidence": confidence,
                        "threshold": confidence_threshold},
            )
            if confidence >= confidence_threshold:
                break

    # Decode.
    yield ProgressEvent(
        stage="decode_start",
        detail={"n_samples": len(outputs)},
    )
    best_output = max(outputs, key=lambda o: o.quality_score)
    decoded_text, synth_output = await loop.run_in_executor(
        None, lambda: helper._decode(outputs, task.prompt),
    )

    run.individual_outputs = outputs
    if synth_output is not None:
        run.overhead_outputs = [synth_output]
        synth_score = await loop.run_in_executor(
            None,
            lambda: scorer.score(
                task.prompt, decoded_text,
                reference=task.reference, task=task,
            ),
        )
        if synth_score >= best_output.quality_score:
            yield TokenEvent(
                text=decoded_text, role="synthesis",
                model=channels[0].model,
                call_id="fountain:decode",
            )
            run.combined_output = decoded_text
            run.final_quality = synth_score
        else:
            # Regression guard: synthesis didn't help, keep the best individual.
            yield TokenEvent(
                text=best_output.text, role="answer",
                model=best_output.model, call_id="fountain:best_individual",
            )
            run.combined_output = best_output.text
            run.final_quality = best_output.quality_score
    else:
        yield TokenEvent(
            text=decoded_text, role="answer",
            model=best_output.model, call_id="fountain:single_survivor",
        )
        run.combined_output = decoded_text
        run.final_quality = best_output.quality_score

    run.compute_metrics()
    yield run


async def _cisc_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    num_samples: int = 5,
    csi_source: str = "verbal_100",
    softmax_temperature: float | None = None,
    voter: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """CISC (Confidence-Informed Self-Consistency).

    N concurrent samples, each carrying intrinsic CSI (verbal_100 score or
    response_probability from logprobs). Then cluster via voter LLM and
    pick the highest-confidence-weighted cluster's top sample.
    """
    import json as _json
    import math
    import re as _re

    from .techniques.soft import (
        append_verbal_confidence_prompt,
        parse_verbal_confidence,
    )

    T = softmax_temperature or (1.0 if csi_source == "response_probability" else 8.0)
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique="cisc",
        config={
            "num_samples": num_samples, "model": channel.model,
            "csi_source": csi_source, "softmax_temperature": T,
            "aggregation": "voter_cluster",
        },
    )

    prompt_text = task.prompt
    if csi_source == "verbal_100":
        prompt_text = append_verbal_confidence_prompt(task.prompt, scale=100)
    # Build a ChatRequest with the augmented prompt; ``atransmit_stream``
    # accepts strings, so just hand it the string when prompt was rewritten.
    # We use atransmit (not _stream) for concurrent branches — branches
    # don't stream tokens to the user.
    yield ProgressEvent(
        stage="branches_start",
        detail={"n_branches": num_samples, "technique": "cisc"},
    )

    async def _one() -> AgentOutput:
        return await channel.atransmit(prompt_text, temperature=0.7)
    outputs = await asyncio.gather(*[_one() for _ in range(num_samples)])
    for i, out in enumerate(outputs):
        yield ProgressEvent(
            stage="branch_complete",
            detail={"branch": i, "model": out.model},
        )

    def _score_all() -> None:
        for o in outputs:
            o.quality_score = scorer.score(
                task.prompt, o.text, reference=task.reference, task=task,
            )
    await loop.run_in_executor(None, _score_all)

    # Per-sample raw confidence c_i.
    if csi_source == "response_probability":
        missing = [o for o in outputs if o.mean_logprob is None]
        if missing:
            models = {o.model for o in missing}
            raise RuntimeError(
                f"CISC(csi_source='response_probability') requires logprobs but "
                f"backend returned none for {len(missing)}/{len(outputs)} "
                f"samples (models: {models}). Switch csi_source to 'verbal_100'."
            )
        raw_confidences = [math.exp(o.mean_logprob) for o in outputs]
    else:
        parsed = [parse_verbal_confidence(o.text, scale=100) for o in outputs]
        unparsed = sum(1 for p in parsed if p is None)
        if unparsed == len(outputs):
            raise RuntimeError(
                f"CISC(csi_source='verbal_100') could not parse any of the "
                f"{len(outputs)} samples for `Confidence: <int>`."
            )
        raw_confidences = [p if p is not None else 0.0 for p in parsed]
        run.config["verbal_unparsed"] = unparsed

    # Confidence normalization (softmax with T).
    scaled = [c / T for c in raw_confidences]
    shift = max(scaled)
    exps = [math.exp(s - shift) for s in scaled]
    Z = sum(exps)
    norm_confidences = [e / Z for e in exps]

    # Voter clustering.
    sync_voter = voter or channel
    joined = "\n\n".join(
        f"### Sample {i+1}\n{o.text}" for i, o in enumerate(outputs)
    )
    cluster_prompt = (
        f"Below are {num_samples} independent answers to the same task. "
        f"Group them into semantic equivalence classes.\n\n"
        f"## Task\n{task.prompt}\n\n"
        f"## Samples\n{joined}\n\n"
        f"Return ONLY a JSON array of {num_samples} integers (smallest "
        f"cluster IDs possible).\n\nJSON:"
    )
    yield ProgressEvent(
        stage="cluster_start",
        detail={"voter_model": sync_voter.model},
    )
    vote = await sync_voter.atransmit(cluster_prompt, temperature=0.0)
    labels: list[int] | None = None
    try:
        m = _re.search(r"\[[\s\d,]*\]", vote.text)
        if m:
            p = _json.loads(m.group(0))
            if isinstance(p, list) and len(p) == num_samples:
                labels = [int(x) for x in p]
    except Exception:
        labels = None
    if labels is None:
        labels = list(range(num_samples))

    totals: dict[int, float] = {}
    members: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        totals[lbl] = totals.get(lbl, 0.0) + norm_confidences[idx]
        members.setdefault(lbl, []).append(idx)
    winning_cluster = max(totals, key=lambda k: totals[k])
    winning_members = members[winning_cluster]
    best_idx = max(winning_members, key=lambda i: raw_confidences[i])
    best = outputs[best_idx]

    yield TokenEvent(
        text=best.text, role="synthesis",
        model=best.model,
        call_id=f"cisc:cluster{winning_cluster}",
    )

    run.individual_outputs = outputs
    run.overhead_outputs = [vote]
    run.rounds = num_samples
    run.config["cluster_labels"] = labels
    run.config["winning_cluster_size"] = len(winning_members)
    run.config["raw_confidences"] = [round(c, 4) for c in raw_confidences]
    run.config["norm_confidences"] = [round(c, 4) for c in norm_confidences]
    run.combined_output = best.text
    run.final_quality = best.quality_score or 0.0
    run.compute_metrics()
    yield run


async def _fec_astream(
    task: TaskItem,
    channel: AgentChannel,
    scorer: QualityScorer,
    *,
    code_rate: float,
) -> AsyncIterator[Any]:
    """FEC (Forward Error Correction): main answer + concurrent parity calls + decode.

    Streams the main answer as ``role="answer"``, runs parity sections
    concurrently (no token streaming for parity, just ProgressEvents), and
    emits the decoded result as ``role="synthesis"``.
    """
    from .techniques.fec import FECService

    helper = FECService(channel=channel, scorer=scorer, code_rate=code_rate)
    loop = asyncio.get_running_loop()
    run = ReliabilityRun(
        task_id=task.id,
        task_category=(
            task.category.value if hasattr(task.category, "value")
            else str(task.category)
        ),
        technique=f"fec_{helper.effective_rate}",
        config={
            "code_rate": code_rate,
            "effective_rate": helper.effective_rate,
            "parity_sections": helper.parity_sections,
            "num_parity_calls": len(helper.parity_sections),
            "model": channel.model,
        },
    )

    # Phase 1: stream main answer.
    yield ProgressEvent(
        stage="main_start",
        detail={"model": channel.model},
    )
    main_output: AgentOutput | None = None
    async for frame in _astream_one_call(
        channel, task.request,
        role="answer", call_id="fec:main",
    ):
        if isinstance(frame, AgentOutput):
            main_output = frame
        else:
            yield frame
    if main_output is None:
        raise RuntimeError("FEC main produced no AgentOutput")
    main_output.quality_score = await loop.run_in_executor(
        None,
        lambda: scorer.score(
            task.prompt, main_output.text, reference=task.reference, task=task,
        ),
    )
    run.individual_outputs.append(main_output)

    if not helper.parity_sections:
        # Rate 1.0 — uncoded.
        run.combined_output = main_output.text
        run.final_quality = main_output.quality_score
        run.compute_metrics()
        yield run
        return

    # Phase 2: concurrent parity calls (no token stream).
    yield ProgressEvent(
        stage="parity_start",
        detail={"n_sections": len(helper.parity_sections),
                "sections": list(helper.parity_sections)},
    )
    from .techniques.fec import SECTION_PROMPTS

    async def _one_parity(section_name: str) -> tuple[str, AgentOutput]:
        prompt = SECTION_PROMPTS[section_name].format(
            task=task.prompt, answer=main_output.text,
        )
        out = await channel.atransmit(prompt)
        return section_name, out

    parity_results = await asyncio.gather(*[
        _one_parity(s) for s in helper.parity_sections
    ])
    parity_outputs: list[AgentOutput] = []
    parity_texts: dict[str, str] = {}
    for _i, (section_name, out) in enumerate(parity_results):
        parity_outputs.append(out)
        parity_texts[section_name] = out.text
        yield ProgressEvent(
            stage="parity_complete",
            detail={"section": section_name, "model": out.model},
        )
    run.individual_outputs.extend(parity_outputs)

    # Phase 3: decode.
    yield ProgressEvent(
        stage="decode_start",
        detail={"n_sections": len(parity_texts)},
    )
    decoded_text, decode_output = await loop.run_in_executor(
        None,
        lambda: helper._decode(task.prompt, main_output.text, parity_texts),
    )
    run.overhead_outputs = [decode_output]
    run.combined_output = decoded_text
    yield TokenEvent(
        text=decoded_text, role="synthesis",
        model=channel.model, call_id="fec:decode",
    )

    # Final scoring (comparative).
    run.final_quality = await loop.run_in_executor(
        None,
        lambda: scorer.score_comparative(
            task.prompt,
            candidate=run.combined_output,
            baseline=main_output.text,
            baseline_score=main_output.quality_score,
            reference=task.reference,
        ),
    )
    run.compute_metrics()
    yield run


_ACM_PROFILE_TO_TECHNIQUE: dict[str, str] = {
    "uncoded": "baseline",
    "harq_ir": "harq_ir",
    "harq_cc": "harq_cc",
    "turbo": "turbo",
    "fountain": "fountain",
    "diversity_mrc": "diversity_mrc",
    "diversity_egc": "diversity_egc",
    "diversity_sc": "diversity_sc",
}


async def _acm_astream(
    task: TaskItem,
    channels: list[AgentChannel],
    scorer: QualityScorer,
    ctx: DispatchContext,
    *,
    acm_table: list[Any] | None = None,
    category_tables: dict[str, list[Any]] | None = None,
    difficulty_estimator: AgentChannel | None = None,
) -> AsyncIterator[Any]:
    """ACM router: estimate difficulty, select profile from table, then
    delegate to the chosen sub-technique's astream so the user sees real
    mid-stream events for whatever technique gets picked.

    Emits ``acm_estimate_start`` / ``acm_route_decision`` ProgressEvents
    around the routing step, then forwards the sub-technique's full event
    stream. The terminal :class:`ReliabilityRun` is re-tagged with
    ``technique="acm_<sub>"`` and gets the difficulty estimation appended
    to ``overhead_outputs`` so cost accounting stays accurate.
    """
    from .techniques.acm import ACMRouter

    helper = ACMRouter(
        channels={ch.model: ch for ch in channels},
        scorer=scorer,
        acm_table=acm_table or [],
        category_tables=category_tables or {},
        difficulty_estimator=difficulty_estimator or channels[0],
    )
    loop = asyncio.get_running_loop()

    # Phase 1: difficulty estimation — single channel call. Run via sync
    # helper in an executor since it has a fallback path that's a bit
    # involved (logprobs probe → self-rating). Future improvement: stream
    # the probe and surface mean_logprob as a confidence indicator.
    yield ProgressEvent(
        stage="acm_estimate_start",
        detail={"model": helper.difficulty_estimator.model},
    )
    difficulty, diff_output = await loop.run_in_executor(
        None, lambda: helper._estimate_difficulty(task),
    )

    # Phase 2: pick profile.
    cat = (
        task.category.value if hasattr(task.category, "value")
        else str(task.category)
    )
    table = (category_tables or {}).get(cat, acm_table or [])
    profile = helper._select_profile(difficulty, table=table)
    sub = _ACM_PROFILE_TO_TECHNIQUE.get(profile.technique)
    if sub is None and profile.technique == "fec":
        sub = f"fec_{profile.code_rate}"
    if sub is None:
        raise ValueError(
            f"ACM profile.technique={profile.technique!r} is not routable to "
            f"a known astream technique"
        )
    yield ProgressEvent(
        stage="acm_route_decision",
        detail={
            "difficulty": difficulty, "profile": profile.name,
            "selected_technique": sub, "selected_model": profile.model,
            "category": cat,
        },
    )

    # Phase 3: delegate. Build a sub-context with profile-specific knob
    # overrides without mutating the caller's ctx.
    sub_overrides: dict[str, Any] = {}
    if sub in ("harq_ir", "harq_cc", "turbo") and profile.max_rounds:
        sub_overrides[sub] = {"max_rounds": profile.max_rounds}
    elif sub == "fountain" and profile.num_branches:
        sub_overrides[sub] = {"max_samples": max(profile.num_branches, 3)}

    sub_ctx = DispatchContext(
        channels=ctx.channels, scorer=ctx.scorer,
        critic_channel=ctx.critic_channel,
        soft_normalization=ctx.soft_normalization, cisc=ctx.cisc,
        early_exit=ctx.early_exit,
        dispatch_overrides={**ctx.dispatch_overrides, **sub_overrides},
    )

    sub_run: ReliabilityRun | None = None
    async for frame in adispatch(sub, task, sub_ctx):
        if isinstance(frame, ReliabilityRun):
            sub_run = frame
        else:
            yield frame
    if sub_run is None:
        raise RuntimeError(f"ACM sub-dispatch ({sub}) yielded no ReliabilityRun")

    # Re-tag run + append difficulty overhead so cost/trace stay consistent
    # with the sync ACM impl.
    sub_run.technique = f"acm_{profile.technique}"
    sub_run.overhead_outputs.append(diff_output)
    sub_run.config["estimated_difficulty"] = difficulty
    sub_run.config["selected_profile"] = profile.name
    sub_run.config["routing_category"] = cat
    sub_run.config["routing_mode"] = (
        "category" if cat in (category_tables or {}) else "global"
    )
    if diff_output.mean_logprob is not None:
        sub_run.config["difficulty_source"] = "pilot_logprob"
        sub_run.config["difficulty_logprob"] = diff_output.mean_logprob
    else:
        sub_run.config["difficulty_source"] = "self_rating"
    sub_run.compute_metrics()
    yield sub_run


async def adispatch(
    technique: str,
    task: TaskItem,
    ctx: DispatchContext,
) -> AsyncIterator[Any]:
    """Async-streaming counterpart to :func:`dispatch`.

    Yields :class:`Event` instances as the technique progresses, and the
    LAST yield is the completed :class:`ReliabilityRun`. For techniques
    listed in :data:`_ASTREAM_TECHNIQUES`, routes to the native async
    implementation. For all others, falls back to executor-wrapped sync
    :func:`dispatch` — no mid-stream events, but the loop stays unblocked.
    """
    channels_list = list(ctx.channels.values())
    if not channels_list:
        raise ValueError("DispatchContext.channels is empty")
    primary = channels_list[0]

    if technique == "baseline":
        async for frame in _baseline_astream(task, primary, ctx.scorer):
            yield frame
        return

    if technique == "harq_ir":
        async for frame in _harq_ir_astream(
            task, primary, ctx.scorer,
            max_rounds=_knob(ctx, technique, "max_rounds", 5),
            quality_threshold=_knob(ctx, technique, "quality_threshold", 0.85),
            critic_channel=ctx.critic_channel,
            early_exit=ctx.early_exit or _knob(ctx, technique, "early_exit", False),
        ):
            yield frame
        return

    if technique == "harq_cc":
        async for frame in _harq_cc_astream(
            task, primary, ctx.scorer,
            max_rounds=_knob(ctx, technique, "max_rounds", 5),
            quality_threshold=_knob(ctx, technique, "quality_threshold", 0.85),
            critic_channel=ctx.critic_channel,
        ):
            yield frame
        return

    if technique == "turbo":
        async for frame in _turbo_astream(
            task, primary, ctx.scorer,
            max_iterations=_knob(ctx, technique, "max_iterations", 5),
            quality_threshold=_knob(ctx, technique, "quality_threshold", 0.85),
            critic_channel=ctx.critic_channel,
            early_exit=ctx.early_exit or _knob(ctx, technique, "early_exit", False),
        ):
            yield frame
        return

    if technique == "self_refine":
        async for frame in _self_refine_astream(
            task, primary, ctx.scorer,
            max_rounds=_knob(ctx, technique, "max_rounds", 3),
            early_stop=_knob(ctx, technique, "early_stop", True),
        ):
            yield frame
        return

    if technique == "chain_of_verification":
        async for frame in _chain_of_verification_astream(
            task, primary, ctx.scorer,
            num_verification_questions=_knob(
                ctx, technique, "num_verification_questions", 3,
            ),
        ):
            yield frame
        return

    if technique == "self_consistency":
        async for frame in _self_consistency_astream(
            task, channels_list, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
        ):
            yield frame
        return

    if technique == "best_of_n":
        async for frame in _best_of_n_astream(
            task, primary, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
        ):
            yield frame
        return

    if technique == "weighted_bon":
        async for frame in _weighted_bon_astream(
            task, primary, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
        ):
            yield frame
        return

    if technique == "mixture_of_agents":
        async for frame in _mixture_of_agents_astream(
            task, channels_list, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            aggregator=primary,
            num_layers=_knob(ctx, technique, "num_layers", 2),
        ):
            yield frame
        return

    if technique == "diversity_sc_N":
        async for frame in _diversity_sc_N_astream(
            task, channels_list, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
        ):
            yield frame
        return

    if technique in ("diversity_mrc_discrete_N", "diversity_mrc_discrete_N_soft"):
        sn = ctx.soft_normalization
        # Soft variant: force softmax-on; non-soft honors ctx.soft_normalization.
        soft_on = (
            True if technique == "diversity_mrc_discrete_N_soft"
            else sn["enabled"]
        )
        soft_T = sn["T_logprob"] if technique.endswith("_soft") else sn["T_judge"]
        async for frame in _diversity_mrc_discrete_N_astream(
            task, channels_list, ctx.scorer,
            num_samples=_knob(ctx, technique, "num_samples", 5),
            voter=primary,
            softmax_normalize=soft_on,
            softmax_temperature=soft_T,
        ):
            yield frame
        return

    if technique == "diversity_mrc_soft":
        # Soft MRC reuses non-soft astream — the softmax-weighted variant is
        # an aggregation-layer detail that lands in the synthesizer prompt;
        # the visible event stream is identical to diversity_mrc.
        async for frame in _diversity_astream(
            task, channels_list, ctx.scorer,
            combining=CombiningStrategy.MRC,
            synthesizer=primary,
        ):
            yield frame
        return

    if technique in ("fountain", "fountain_soft"):
        # fountain_soft uses logprob-weighted confidence in the synthesizer;
        # the visible event stream and decoder shape are the same.
        async for frame in _fountain_astream(
            task, channels_list, ctx.scorer,
            confidence_threshold=_knob(ctx, technique, "confidence_threshold", 0.8),
            max_samples=_knob(ctx, technique, "max_samples", 8),
            min_samples=_knob(ctx, technique, "min_samples", 3),
        ):
            yield frame
        return

    if technique == "cisc":
        cisc_cfg = ctx.cisc or {}
        async for frame in _cisc_astream(
            task, primary, ctx.scorer,
            num_samples=cisc_cfg.get("num_samples", 5),
            csi_source=cisc_cfg.get("csi_source", "verbal_100"),
            softmax_temperature=cisc_cfg.get("softmax_temperature"),
            voter=primary,
        ):
            yield frame
        return

    if technique.startswith("fec_"):
        code_rate = float(technique.split("_", 1)[1])
        async for frame in _fec_astream(
            task, primary, ctx.scorer, code_rate=code_rate,
        ):
            yield frame
        return

    if technique in {
        "diversity_sc", "diversity_mrc", "diversity_egc",
        "diversity_spatial", "diversity_frequency", "diversity_time",
    }:
        from .techniques.diversity import DEFAULT_PROMPT_VARIANTS

        # Map each variant to its config knobs (matches the sync dispatcher).
        if technique == "diversity_spatial":
            div_channels = channels_list
            div_combining = CombiningStrategy.MRC
            div_variants: dict[str, str] | None = None
            div_temps: list[float] | None = None
        elif technique == "diversity_frequency":
            div_channels = [primary]
            div_combining = CombiningStrategy.MRC
            div_variants = DEFAULT_PROMPT_VARIANTS
            div_temps = None
        elif technique == "diversity_time":
            div_channels = [primary]
            div_combining = CombiningStrategy.MRC
            div_variants = None
            div_temps = [0.3, 0.5, 0.7, 0.9]
        else:
            div_channels = channels_list
            div_combining = CombiningStrategy(technique.split("_", 1)[1])
            div_variants = None
            div_temps = None

        async for frame in _diversity_astream(
            task, div_channels, ctx.scorer,
            combining=div_combining,
            prompt_variants=div_variants,
            temperature_spread=div_temps,
            synthesizer=primary,
        ):
            yield frame
        return

    if technique == "acm":
        # Inline ACM tables may be carried on the ctx by the user's setup.
        async for frame in _acm_astream(
            task, channels_list, ctx.scorer, ctx,
            acm_table=ctx.acm_table,
            category_tables=ctx.acm_category_tables,
            difficulty_estimator=primary,
        ):
            yield frame
        return

    # Fallback for un-converted techniques: run sync dispatch in executor
    # so the event loop stays responsive. Yields only the final run; the
    # mid-stream events story for these techniques is Phase 3 future work.
    loop = asyncio.get_running_loop()
    run = await loop.run_in_executor(None, lambda: dispatch(technique, task, ctx))
    yield run


def is_streamable(technique: str) -> bool:
    """Return True iff this technique has a native astream implementation.

    Native-streaming techniques emit per-call ``TokenEvent`` and per-stage
    ``ProgressEvent`` instances mid-run when driven through
    :func:`adispatch` (or ``ReliabilityModule.astream``). Non-streaming
    techniques still complete correctly through ``adispatch`` — they just
    fall back to executor-wrapped sync ``dispatch()`` and emit only the
    terminal :class:`ReliabilityRun`.

    Use this for UI / progress-display logic that needs to know whether a
    given technique will produce meaningful mid-stream events. The full
    set is enumerated as :data:`_ASTREAM_TECHNIQUES`.
    """
    return technique in _ASTREAM_TECHNIQUES
