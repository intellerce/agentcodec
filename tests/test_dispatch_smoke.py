"""Smoke tests for `agentcodec.dispatch.dispatch()`.

Goal: make sure the dispatch wiring is sound. Each test runs a single
representative technique through the dispatcher with MockChannel and
MockScorer, then checks that the returned `ReliabilityRun` is shaped
correctly. We don't try to exercise *every* technique — many require
specific judge behaviors (CISC verbal-100, fountain thresholds) that
make them integration-level rather than unit tests. The representatives
below cover the wiring for every family.

If any of these break after a refactor, the dispatcher's contract with
the techniques is the regression candidate.
"""

from __future__ import annotations

import pytest

from agentcodec.dispatch import KNOWN_TECHNIQUES, dispatch
from agentcodec.models import ReliabilityRun


def test_known_techniques_count() -> None:
    """KNOWN_TECHNIQUES is part of the public API. The README claims 29."""
    assert len(KNOWN_TECHNIQUES) == 29, (
        f"KNOWN_TECHNIQUES drifted to {len(KNOWN_TECHNIQUES)} entries; "
        "update the README count if this is intentional."
    )
    # No duplicates.
    assert len(set(KNOWN_TECHNIQUES)) == len(KNOWN_TECHNIQUES)


@pytest.mark.parametrize("technique", [
    "baseline",
    "self_refine",
    "diversity_sc",
    "diversity_mrc",
    "diversity_egc",
])
def test_dispatch_smoke_no_critic(technique, qa_task, dispatch_ctx) -> None:
    """Techniques that don't need a critic channel."""
    run = dispatch(technique, qa_task, dispatch_ctx)
    assert isinstance(run, ReliabilityRun)
    assert run.individual_outputs, f"{technique!r} produced no individual outputs"
    assert run.combined_output, f"{technique!r} produced no combined output"


def test_dispatch_smoke_harq_ir(qa_task, dispatch_ctx, mock_channel) -> None:
    """HARQ-IR uses the critic channel. Wire one explicitly."""
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.early_exit = True
    dispatch_ctx.dispatch_overrides = {"harq_ir": {"max_rounds": 2}}
    run = dispatch("harq_ir", qa_task, dispatch_ctx)
    assert isinstance(run, ReliabilityRun)
    assert run.combined_output
    assert run.rounds >= 1


def test_dispatch_smoke_turbo(qa_task, dispatch_ctx, mock_channel) -> None:
    dispatch_ctx.critic_channel = mock_channel
    dispatch_ctx.early_exit = True
    dispatch_ctx.dispatch_overrides = {"turbo": {"max_iterations": 2}}
    run = dispatch("turbo", qa_task, dispatch_ctx)
    assert isinstance(run, ReliabilityRun)
    assert run.combined_output


def test_dispatch_smoke_fec(qa_task, dispatch_ctx) -> None:
    """FEC dispatch keys on the rate suffix in the technique name."""
    run = dispatch("fec_0.50", qa_task, dispatch_ctx)
    assert isinstance(run, ReliabilityRun)


def test_dispatch_unknown_technique_raises(qa_task, dispatch_ctx) -> None:
    with pytest.raises(ValueError, match="Unknown technique"):
        dispatch("nonexistent_technique", qa_task, dispatch_ctx)


def test_dispatch_empty_channels_raises(qa_task, mock_scorer) -> None:
    """A DispatchContext with no channels is a programming error,
    not a runtime fallback."""
    from agentcodec.dispatch import DispatchContext
    ctx = DispatchContext(channels={}, scorer=mock_scorer)
    with pytest.raises(ValueError, match="DispatchContext.channels is empty"):
        dispatch("baseline", qa_task, ctx)


def test_acm_learned_without_router_raises(qa_task, dispatch_ctx) -> None:
    """acm_learned needs preloaded router weights; that's an explicit
    contract — fail loudly rather than silently doing nothing."""
    assert dispatch_ctx.acm_learned_router is None
    with pytest.raises(ValueError, match="acm_learned"):
        dispatch("acm_learned", qa_task, dispatch_ctx)
