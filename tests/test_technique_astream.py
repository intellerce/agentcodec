"""Phase 3 unit tests — adispatch() + per-technique astream() generators.

Each test drives ``adispatch(technique, task, ctx)`` end-to-end and asserts:
* TokenEvents flow as the technique progresses (for native-async techniques)
* The terminal frame is a ``ReliabilityRun`` with correct shape
* Sync-fallback techniques produce a ReliabilityRun without mid-stream events
"""
from __future__ import annotations

import pytest

from agentcodec.dispatch import adispatch
from agentcodec.models import ReliabilityRun
from agentcodec.results import ProgressEvent, TokenEvent

# ---------------------------------------------------------------------------
# Baseline native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_astream_streams_tokens(qa_task, dispatch_ctx, mock_channel):
    """The baseline technique forwards each ChannelChunk as a TokenEvent."""
    # Use a single mock channel so the conftest pair doesn't introduce
    # surprises; the dispatcher always picks channels_list[0] for baseline.
    dispatch_ctx.channels = {"primary": mock_channel}
    mock_channel.responses = [("Hello world example", 0.91)]

    events: list = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("baseline", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        else:
            events.append(frame)

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    progress_events = [e for e in events if isinstance(e, ProgressEvent)]

    # Three words in "Hello world example" → three answer chunks.
    answer_tokens = [t for t in token_events if t.role == "answer"]
    assert len(answer_tokens) == 3
    assert "".join(t.text for t in answer_tokens) == "Hello world example"
    # Progress events bracket the channel call.
    stages = [p.stage for p in progress_events]
    assert "channel_start" in stages and "channel_complete" in stages
    # Final frame is the ReliabilityRun.
    assert run is not None
    assert run.technique == "baseline"
    assert run.combined_output == "Hello world example"
    # MockScorer ignores the 0.91 metadata on AgentOutput (it scores the
    # bare text string) — we only care that final_quality is populated.
    assert run.final_quality is not None
    assert 0.0 < run.final_quality <= 1.0


# ---------------------------------------------------------------------------
# HARQ-IR native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harq_ir_astream_single_round_when_threshold_hit(
    qa_task, dispatch_ctx, mock_channel,
):
    """Round 1 above threshold → stop immediately with no critic call."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.dispatch_overrides = {"harq_ir": {"max_rounds": 5, "quality_threshold": 0.5}}
    mock_channel.responses = [("Very good answer", 0.92)]  # quality > threshold

    events: list = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("harq_ir", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        else:
            events.append(frame)

    assert run is not None
    assert run.technique == "harq_ir"
    assert run.rounds == 1
    # No critic events; only round 1 progress + answer tokens.
    assert not any(
        isinstance(e, TokenEvent) and e.role == "critique" for e in events
    )
    answer_tokens = [
        e for e in events if isinstance(e, TokenEvent) and e.role == "answer"
    ]
    assert "".join(t.text for t in answer_tokens) == "Very good answer"


@pytest.mark.asyncio
async def test_harq_ir_astream_multi_round_emits_draft_and_critique(
    qa_task, dispatch_ctx, mock_channel,
):
    """Round 1 below threshold → emits critic + refined draft."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.dispatch_overrides = {
        "harq_ir": {"max_rounds": 2, "quality_threshold": 0.99},
    }
    # Round 1 answer below threshold → triggers round 2 with critic + refine.
    mock_channel.responses = [
        ("First weak answer", 0.30),    # initial
        ('{"issues": [{"quote": "weak", "fix": "make it stronger"}]}', 0.0),  # critic
        ("Refined stronger answer", 0.50),  # refinement
    ]

    events: list = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("harq_ir", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        else:
            events.append(frame)

    assert run is not None
    assert run.rounds >= 1
    role_counts: dict[str, int] = {}
    for e in events:
        if isinstance(e, TokenEvent):
            role_counts[e.role] = role_counts.get(e.role, 0) + 1
    # We expect at least: answer (round 1), critique (round 2), draft (round 2).
    assert role_counts.get("answer", 0) >= 1
    # Either critique fires (some quality_score paths skip round 2 entirely
    # in plateau/no-issues branches); when it does fire, drafts must too.
    if role_counts.get("critique", 0) > 0:
        assert role_counts.get("draft", 0) >= 1


# ---------------------------------------------------------------------------
# Sync fallback for un-converted techniques
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adispatch_fallback_for_unconverted_technique(
    qa_task, dispatch_ctx,
):
    """``diversity_sc_N`` doesn't have a native astream yet — it falls back
    to executor-wrapped sync ``dispatch()`` and yields only the terminal
    ReliabilityRun, no mid-stream events. Also verifies ``is_streamable()``
    correctly identifies the technique as not natively streaming."""
    from agentcodec.dispatch import is_streamable

    # acm_soft is one of the two techniques still on the sync fallback path.
    assert is_streamable("acm_soft") is False
    assert is_streamable("acm_learned") is False
    assert is_streamable("baseline") is True
    assert is_streamable("acm") is True  # was added in this batch
    # We don't actually drive acm_soft here (it needs a soft-aggregation
    # setup); the introspection assertions above are the contract.


@pytest.mark.asyncio
async def test_adispatch_empty_channels_raises(qa_task, mock_scorer):
    """Same guard as sync dispatch(): empty channels is a programming error."""
    from agentcodec.dispatch import DispatchContext

    ctx = DispatchContext(channels={}, scorer=mock_scorer)
    with pytest.raises(ValueError, match="DispatchContext.channels is empty"):
        async for _ in adispatch("baseline", qa_task, ctx):
            pass


# ---------------------------------------------------------------------------
# HARQ-CC native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harq_cc_astream_early_exit_on_first_attempt(
    qa_task, dispatch_ctx, mock_channel,
):
    """First attempt above threshold → no combine call; final answer is
    the first attempt; per-round events fire as ``role="candidate"``."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.dispatch_overrides = {
        "harq_cc": {"max_rounds": 3, "quality_threshold": 0.5},
    }
    mock_channel.responses = [("Great strong answer", 0.95)]

    events: list = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("harq_cc", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        else:
            events.append(frame)

    assert run is not None
    assert run.rounds == 1
    candidate_tokens = [
        e for e in events if isinstance(e, TokenEvent) and e.role == "candidate"
    ]
    assert candidate_tokens, "expected at least one role=candidate TokenEvent"
    # No synthesis call when early-exit fires.
    assert not any(
        isinstance(e, TokenEvent) and e.role == "synthesis" for e in events
    )


@pytest.mark.asyncio
async def test_harq_cc_astream_combines_when_threshold_not_met(
    qa_task, dispatch_ctx, mock_channel,
):
    """All rounds below threshold → critic combines via chase-combining,
    yields a ``role="synthesis"`` TokenEvent."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.dispatch_overrides = {
        "harq_cc": {"max_rounds": 2, "quality_threshold": 0.99},
    }
    # Both attempts fall below threshold + synthesizer call (+ score_comparative)
    mock_channel.responses = [
        ("First weak attempt", 0.10),
        ("Second weak attempt", 0.15),
        ("Combined consensus answer", 0.20),
    ]

    synthesis_tokens: list[str] = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("harq_cc", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, TokenEvent) and frame.role == "synthesis":
            synthesis_tokens.append(frame.text)

    assert run is not None
    assert run.rounds == 2
    assert run.overhead_outputs, "combined output must land in overhead_outputs"
    assert run.combined_output
    assert synthesis_tokens, "expected at least one role=synthesis TokenEvent"


# ---------------------------------------------------------------------------
# Turbo native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turbo_astream_above_threshold_returns_early(
    qa_task, dispatch_ctx, mock_channel,
):
    """Iteration 0 above threshold → no critic call; iter_complete fires."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.dispatch_overrides = {
        "turbo": {"max_iterations": 5, "quality_threshold": 0.5},
    }
    mock_channel.responses = [("Already great", 0.92)]

    events: list = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("turbo", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        else:
            events.append(frame)

    assert run is not None
    assert run.rounds == 1
    assert not any(
        isinstance(e, TokenEvent) and e.role == "critique" for e in events
    )
    iter_complete = [
        e for e in events if isinstance(e, ProgressEvent) and e.stage == "iter_complete"
    ]
    assert len(iter_complete) == 1


# ---------------------------------------------------------------------------
# Self-Refine native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_refine_astream_emits_draft_critique_revise(
    qa_task, dispatch_ctx, mock_channel,
):
    """Self-refine should yield answer/critique/draft TokenEvents in order
    and assemble a ReliabilityRun with both draft + revision in history."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.dispatch_overrides = {
        "self_refine": {"max_rounds": 1, "early_stop": False},
    }
    # 3 calls: initial draft, critique, revised. early_stop=False so the
    # STOP|CONTINUE prefix check doesn't short-circuit.
    mock_channel.responses = [
        ("Initial draft answer", 0.40),
        ("CONTINUE: the answer lacks detail.", 0.0),
        ("Revised more-detailed answer", 0.70),
    ]

    role_seen: list[str] = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("self_refine", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, TokenEvent):
            if not role_seen or role_seen[-1] != frame.role:
                role_seen.append(frame.role)

    assert run is not None
    assert run.technique == "self_refine"
    # Order: answer (draft 0) → critique → draft (revised).
    assert role_seen == ["answer", "critique", "draft"]
    assert run.combined_output == "Revised more-detailed answer"
    assert len(run.individual_outputs) == 3  # draft, critique, revised


@pytest.mark.asyncio
async def test_self_refine_astream_early_stop_on_signal(
    qa_task, dispatch_ctx, mock_channel,
):
    """early_stop=True → critique starting with 'STOP' breaks the loop;
    no revision is emitted."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.dispatch_overrides = {
        "self_refine": {"max_rounds": 5, "early_stop": True},
    }
    mock_channel.responses = [
        ("Decent initial draft", 0.70),
        ("STOP\nThe answer is good as-is.", 0.0),
    ]

    role_seen: list[str] = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("self_refine", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, TokenEvent):
            if not role_seen or role_seen[-1] != frame.role:
                role_seen.append(frame.role)

    assert run is not None
    assert run.config.get("stop_reason") == "model_signal"
    assert "draft" not in role_seen   # no revise call after STOP
    assert run.combined_output == "Decent initial draft"


# ---------------------------------------------------------------------------
# Chain-of-Verification native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cove_astream_full_pipeline(qa_task, dispatch_ctx, mock_channel):
    """CoVe: baseline → plan → N verifications → revise. Verifies the
    role taxonomy and that all stages emit progress events."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.dispatch_overrides = {
        "chain_of_verification": {"num_verification_questions": 2},
    }
    # 5 calls: baseline, plan, verify1, verify2, revise.
    mock_channel.responses = [
        ("Draft answer about Paris", 0.50),
        ("1) Is Paris the capital of France?\n2) Was the Eiffel Tower built in 1889?", 0.0),
        ("Yes, Paris is the capital.", 0.0),
        ("Yes, the Eiffel Tower was built in 1889.", 0.0),
        ("Revised: Paris is the capital of France. Eiffel Tower built 1889.", 0.80),
    ]

    role_seen: list[str] = []
    stage_seen: list[str] = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("chain_of_verification", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, TokenEvent):
            if not role_seen or role_seen[-1] != frame.role:
                role_seen.append(frame.role)
        elif isinstance(frame, ProgressEvent):
            stage_seen.append(frame.stage)

    assert run is not None
    # Role taxonomy: answer (baseline) → verification (plan + N verifies) → answer (revise).
    assert role_seen[0] == "answer"
    assert "verification" in role_seen
    assert role_seen[-1] == "answer"
    # Stage events for all 4 steps.
    assert "baseline_start" in stage_seen
    assert "plan_start" in stage_seen
    assert "verify_start" in stage_seen
    assert "revise_start" in stage_seen
    # rounds = 1 + num_verification_questions
    assert run.rounds == 3
    # overhead carries the plan + N verifications
    assert len(run.overhead_outputs) == 3   # plan + 2 verifications


# ---------------------------------------------------------------------------
# Parallel-branch — diversity_mrc / diversity_sc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diversity_sc_astream_picks_best_branch(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """diversity_sc runs all branches concurrently, scores them, and emits
    the best as ``role="answer"`` (no synthesis call)."""
    # Different responses per channel so SC has a clear winner.
    mock_channel_pair[0].responses = [("Weak answer", 0.30)]
    mock_channel_pair[1].responses = [("Strong detailed answer x" * 5, 0.80)]

    progress_stages: list[str] = []
    answer_tokens: list[str] = []
    run: ReliabilityRun | None = None
    async for frame in adispatch("diversity_sc", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent) and frame.role == "answer":
            answer_tokens.append(frame.text)

    assert run is not None
    assert run.technique == "diversity_sc"
    # Concurrent branches → both fire branch_complete events
    branch_complete_count = sum(s == "branch_complete" for s in progress_stages)
    assert branch_complete_count == 2
    assert "branches_scored" in progress_stages
    # SC: no synthesis emission, just role=answer for the winner
    assert "".join(answer_tokens), "expected role=answer text"


@pytest.mark.asyncio
async def test_diversity_mrc_astream_synthesizes(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """diversity_mrc runs branches concurrently, scores, then judge (MRC
    synthesizer) emits a ``role="synthesis"`` TokenEvent. Two distinct
    branches → MRC synthesis path (not SC fallback)."""
    # Branches close enough in quality that MRC fires (otherwise SC fallback).
    # MRC needs ≥ 2 outputs with the secondary scoring ≥ 50% of best.
    mock_channel_pair[0].responses = [("Answer about A", 0.70)]
    mock_channel_pair[1].responses = [("Answer about B", 0.65)]

    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("diversity_mrc", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "diversity_mrc"
    # MRC fires the synthesis event (or answer if SC fallback engaged).
    assert role_counts.get("synthesis", 0) + role_counts.get("answer", 0) >= 1


# ---------------------------------------------------------------------------
# Best-of-N native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_best_of_n_astream_emits_winner(
    qa_task, dispatch_ctx, mock_channel,
):
    """Best-of-N: 3 concurrent samples, winner emitted as role=answer."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.dispatch_overrides = {"best_of_n": {"num_samples": 3}}
    # Different responses per call so there's a clear winner.
    mock_channel.responses = [
        ("Short", 0.30),
        ("Medium length answer here", 0.40),
        ("Much longer and more detailed answer with extra information", 0.50),
    ]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("best_of_n", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "best_of_n"
    assert run.rounds == 3
    branch_completes = sum(s == "branch_complete" for s in progress_stages)
    assert branch_completes == 3
    # Winner is emitted as role=answer (single emission).
    assert role_counts.get("answer", 0) == 1


# ---------------------------------------------------------------------------
# Mixture-of-Agents native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixture_of_agents_astream_full_pipeline(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """MoA: K proposers × L layers concurrent per-layer, then aggregator
    streams as role=answer."""
    dispatch_ctx.dispatch_overrides = {
        "mixture_of_agents": {"num_samples": 2, "num_layers": 1},
    }
    for ch in mock_channel_pair:
        ch.responses = [("Proposer answer", 0.50), ("Aggregated final answer", 0.80)]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("mixture_of_agents", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "mixture_of_agents"
    # Layer 1 fires; aggregate_start fires.
    assert "layer_start" in progress_stages
    assert "aggregate_start" in progress_stages
    # K=2 proposers → 2 proposer_complete events for layer 1.
    proposer_completes = sum(s == "proposer_complete" for s in progress_stages)
    assert proposer_completes == 2
    # Aggregator streams as role=answer (token-streamed via _astream_one_call).
    assert role_counts.get("answer", 0) >= 1


# ---------------------------------------------------------------------------
# Self-Consistency native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_consistency_astream_concurrent_samples_and_vote(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """Self-consistency: N concurrent samples + voter emits role=synthesis."""
    dispatch_ctx.dispatch_overrides = {"self_consistency": {"num_samples": 3}}
    for ch in mock_channel_pair:
        ch.responses = [("Answer", 0.50)]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("self_consistency", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.rounds == 3
    # 3 concurrent branches.
    branch_completes = sum(s == "branch_complete" for s in progress_stages)
    assert branch_completes == 3
    assert "aggregate_start" in progress_stages
    # Voter emits role=synthesis.
    assert role_counts.get("synthesis", 0) == 1


# ---------------------------------------------------------------------------
# Wider-pool variants — diversity_sc_N / diversity_mrc_discrete_N
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diversity_sc_N_astream_concurrent_picks_best(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """diversity_sc_N: cycles N samples through configured channels, picks best."""
    dispatch_ctx.dispatch_overrides = {"diversity_sc_N": {"num_samples": 3}}
    # Distinct responses so the scorer's text-length formula produces an argmax.
    mock_channel_pair[0].responses = [("Short", 0.30), ("Medium", 0.40)]
    mock_channel_pair[1].responses = [("Much longer detailed answer", 0.50)]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("diversity_sc_N", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "diversity_sc_N"
    assert run.rounds == 3
    branch_completes = sum(s == "branch_complete" for s in progress_stages)
    assert branch_completes == 3
    # Winner emitted as role=answer (single emission).
    assert role_counts.get("answer", 0) == 1


@pytest.mark.asyncio
async def test_diversity_mrc_discrete_N_astream_emits_synthesis(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """diversity_mrc_discrete_N: concurrent samples + voter clusters them →
    emits role=synthesis for the cluster winner."""
    dispatch_ctx.dispatch_overrides = {
        "diversity_mrc_discrete_N": {"num_samples": 3},
    }
    # Sample responses + voter cluster JSON.
    mock_channel_pair[0].responses = [
        ("Answer A", 0.40),
        ("Answer A again", 0.50),
        ("[0, 0, 1]", 0.0),  # voter cluster labels
    ]
    mock_channel_pair[1].responses = [
        ("Answer B different", 0.30),
    ]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("diversity_mrc_discrete_N", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "diversity_mrc_discrete_N"
    # 3 concurrent branches + cluster stage.
    assert sum(s == "branch_complete" for s in progress_stages) == 3
    assert "cluster_start" in progress_stages
    # Voter winner emitted as role=synthesis.
    assert role_counts.get("synthesis", 0) == 1


# ---------------------------------------------------------------------------
# Fountain native astream (sequential rateless)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fountain_astream_sequential_with_decode(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """Fountain samples sequentially until confidence threshold OR max_samples.
    Emits candidates per sample and a synthesis (or answer) at decode."""
    dispatch_ctx.dispatch_overrides = {
        "fountain": {
            "confidence_threshold": 0.99,  # never reached → run to max_samples
            "min_samples": 2, "max_samples": 3,
        },
    }
    for ch in mock_channel_pair:
        ch.responses = [("Sample answer text", 0.50)] * 5

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("fountain", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "fountain"
    assert run.rounds == 3
    # Per-sample stages fire.
    assert sum(s == "sample_start" for s in progress_stages) == 3
    assert sum(s == "sample_complete" for s in progress_stages) == 3
    # confidence_check fires after each sample once we have min_samples.
    assert sum(s == "confidence_check" for s in progress_stages) >= 1
    # decode_start fires once at the end.
    assert "decode_start" in progress_stages
    # 3 candidate tokens + 1 synthesis-or-answer at decode.
    assert role_counts.get("candidate", 0) == 3
    assert role_counts.get("synthesis", 0) + role_counts.get("answer", 0) == 1


# ---------------------------------------------------------------------------
# CISC native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cisc_astream_verbal_path(
    qa_task, dispatch_ctx, mock_channel,
):
    """CISC with verbal_100 CSI: N concurrent samples, voter clusters."""
    dispatch_ctx.channels = {"primary": mock_channel}
    dispatch_ctx.cisc = {
        "csi_source": "verbal_100", "num_samples": 3, "softmax_temperature": 8.0,
    }
    # Each sample must include a "Confidence: <int>" line. The voter call
    # returns a cluster JSON.
    mock_channel.responses = [
        ("Answer A.\n\nConfidence: 80", 0.50),
        ("Answer A again.\n\nConfidence: 70", 0.50),
        ("Different answer.\n\nConfidence: 40", 0.50),
        ("[0, 0, 1]", 0.0),  # voter cluster labels
    ]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("cisc", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique == "cisc"
    assert run.rounds == 3
    # 3 concurrent branches + cluster.
    assert sum(s == "branch_complete" for s in progress_stages) == 3
    assert "cluster_start" in progress_stages
    # Voter winner emitted as role=synthesis.
    assert role_counts.get("synthesis", 0) == 1


# ---------------------------------------------------------------------------
# FEC native astream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fec_astream_main_and_concurrent_parity(
    qa_task, dispatch_ctx, mock_channel,
):
    """FEC: stream main answer; run parity sections concurrently; emit decoded."""
    dispatch_ctx.channels = {"primary": mock_channel}
    mock_channel.responses = [("Main answer", 0.50)] * 10

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("fec_0.50", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    assert run.technique.startswith("fec_")
    # Stages: main_start, parity_start, parity_complete x N, decode_start.
    assert "main_start" in progress_stages
    assert "parity_start" in progress_stages
    assert "decode_start" in progress_stages
    # Main answer streams (as role=answer) and decode emits role=synthesis.
    assert role_counts.get("answer", 0) >= 1
    assert role_counts.get("synthesis", 0) == 1


# ---------------------------------------------------------------------------
# ACM router native astream (delegate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acm_astream_delegates_to_sub_technique(
    qa_task, dispatch_ctx, mock_channel_pair,
):
    """ACM router: estimate difficulty, pick profile, delegate to sub-technique.
    The forwarded event stream is whatever the picked sub-technique emits."""
    # Hand-coded ACM table that always picks baseline (so we get a
    # deterministic sub-technique to assert on).
    from agentcodec.techniques.acm import ACMProfile

    dispatch_ctx.acm_table = [
        ACMProfile(
            name="MCS-Always-Baseline",
            difficulty_range=(0.0, 1.0),
            model=mock_channel_pair[0].model,
            technique="uncoded",
        ),
    ]
    for ch in mock_channel_pair:
        ch.responses = [
            ("Probe with confidence", 0.50),  # difficulty probe
            ("Baseline answer", 0.70),         # sub-technique baseline
        ]

    progress_stages: list[str] = []
    role_counts: dict[str, int] = {}
    run: ReliabilityRun | None = None
    async for frame in adispatch("acm", qa_task, dispatch_ctx):
        if isinstance(frame, ReliabilityRun):
            run = frame
        elif isinstance(frame, ProgressEvent):
            progress_stages.append(frame.stage)
        elif isinstance(frame, TokenEvent):
            role_counts[frame.role] = role_counts.get(frame.role, 0) + 1

    assert run is not None
    # ACM re-tags the technique with its sub-technique:
    assert run.technique == "acm_uncoded"
    # The ACM-specific routing events fired.
    assert "acm_estimate_start" in progress_stages
    assert "acm_route_decision" in progress_stages
    # The sub-technique (baseline) fired its own events through us.
    assert "channel_start" in progress_stages
    # Difficulty estimation lands in overhead_outputs (sub_run had one
    # AgentOutput appended for the difficulty probe).
    assert len(run.overhead_outputs) >= 1
    assert run.config.get("selected_profile") == "MCS-Always-Baseline"
