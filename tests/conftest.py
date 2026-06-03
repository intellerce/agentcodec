"""Shared test fixtures.

The expensive moving parts of `agentcodec` are the LLM calls. Every test
in this directory should be a unit test — never hit the network. We
build a `MockChannel` / `MockScorer` pair that returns canned outputs so
the dispatcher and the rest of the library can be exercised under
pytest without an Ollama / API key in sight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentcodec.messages import ChatRequest
from agentcodec.models import AgentOutput, TaskCategory, TaskItem


@pytest.fixture(autouse=True)
def _silence_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the test process ship a real telemetry event."""
    monkeypatch.setenv("AGENTCODEC_TELEMETRY", "0")
    monkeypatch.setenv("AGENTCODEC_TELEMETRY_QUIET", "1")


# ---------------------------------------------------------------------------
# MockChannel — drop-in for AgentChannel in dispatch-level tests
# ---------------------------------------------------------------------------


@dataclass
class MockChannel:
    """Stand-in for `agentcodec.channel.AgentChannel` in dispatch tests.

    The dispatcher and every technique only call ``transmit()`` on a
    channel — we implement just that, plus the public fields the
    techniques read. The shape mirrors :class:`AgentOutput` exactly so
    downstream code that introspects `.token_count`, `.cost_usd`, etc.
    keeps working.

    Notes:
      * ``responses``: optional list of (text, quality) pairs cycled per
        call. If exhausted (or empty) we fall back to the default
        ``(answer, 0.8)`` pair.
      * ``raise_after``: if set, raises ``RuntimeError`` after N calls —
        used to test the ``fallback_baseline`` path.
    """
    model: str = "mock/gpt"
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = "mock"
    system_prompt: str = "You are a helpful assistant."
    is_anthropic: bool = False
    cost_per_1m: tuple[float, float] = (0.10, 0.10)
    responses: list[tuple[str, float]] = field(default_factory=list)
    raise_after: int | None = None
    _call_count: int = 0
    # Each entry records {prompt, request, temperature, prompt_variant}
    # so tests can assert on the shape of what the technique sent.
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transmit(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
        request_logprobs: bool = False,
    ) -> AgentOutput:
        self._call_count += 1
        if self.raise_after is not None and self._call_count > self.raise_after:
            raise RuntimeError(f"MockChannel raise_after={self.raise_after} hit")
        # Normalize like the real AgentChannel does so tests can inspect either
        # the string view (`call["prompt"]`) or the structured view
        # (`call["request"]`) uniformly.
        if isinstance(prompt_or_request, ChatRequest):
            request = prompt_or_request
            prompt_text = request.last_user_text
        else:
            prompt_text = prompt_or_request
            request = ChatRequest.from_prompt(
                prompt_or_request, system=self.system_prompt,
            )
        self.calls.append({
            "prompt": prompt_text,
            "request": request,
            "temperature": temperature,
            "prompt_variant": prompt_variant,
            "request_logprobs": request_logprobs,
        })

        if self.responses:
            text, quality = self.responses[(self._call_count - 1) % len(self.responses)]
        else:
            text, quality = "MOCK ANSWER: " + prompt_text[:40], 0.8

        t = temperature if temperature is not None else self.temperature
        out = AgentOutput(
            text=text,
            model=self.model,
            temperature=t,
            prompt_variant=prompt_variant,
            quality_score=quality,
            latency_s=0.001,
            token_count=42,
            input_tokens=20,
            output_tokens=22,
            cost_usd=0.000042,
            cost_source="exact_user_rate",
            rate_input_per_1m=self.cost_per_1m[0],
            rate_output_per_1m=self.cost_per_1m[1],
            answer_tokens=22,
            answer_cost_usd=0.000042,
            finish_reason="stop",
        )
        if request_logprobs:
            out.token_logprobs = [-0.1] * out.output_tokens
            out.mean_logprob = -0.1
        return out

    async def atransmit(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
    ) -> AgentOutput:
        """Async one-shot — mirrors :meth:`AgentChannel.atransmit`. Wraps
        the sync transmit for deterministic test behavior."""
        return self.transmit(
            prompt_or_request, temperature=temperature,
            prompt_variant=prompt_variant,
        )

    async def atransmit_stream(
        self,
        prompt_or_request: str | ChatRequest,
        temperature: float | None = None,
        prompt_variant: str = "default",
    ):
        """Async-stream counterpart used by per-technique astream tests.

        Builds the same :class:`AgentOutput` ``transmit()`` would, then
        emits it as a handful of ``ChannelChunk(role="answer")`` frames
        (word-split for deterministic chunking) plus a terminal
        ``ChannelDone``. Records the call the same way ``transmit()``
        does so existing assertions keep working.
        """
        from agentcodec.messages import ChannelChunk, ChannelDone

        out = self.transmit(
            prompt_or_request, temperature=temperature,
            prompt_variant=prompt_variant,
        )
        # Stream the answer word-by-word for deterministic chunk counts.
        words = out.text.split(" ")
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            yield ChannelChunk(role="answer", text=piece)
        yield ChannelDone(output=out)


# ---------------------------------------------------------------------------
# MockScorer — drop-in for QualityScorer
# ---------------------------------------------------------------------------


class MockScorer:
    """Drop-in QualityScorer that returns canned quality scores.

    Techniques call ``score(...)`` per candidate and then aggregate. We
    return the candidate's ``quality_score`` field if present, else a
    deterministic value derived from the text length so two distinct
    outputs don't accidentally tie.
    """

    def __init__(self, judge_model: str = "mock/judge"):
        self.judge_model = judge_model
        self.judge = MockChannel(model=judge_model, cost_per_1m=(0.15, 0.15))
        self.score_strategy = "judge"
        self.cost_so_far = 0.0
        self.call_count = 0
        # Mirror QualityScorer's judge-output buffer so the dispatcher's
        # post-dispatch `run.judge_outputs = self.scorer.collect_judge_outputs()`
        # works against the mock. The mock records no real judge AgentOutputs,
        # so this drains empty — the correct shape for trace assembly.
        self._judge_outputs: list[AgentOutput] = []

    def score(self, prompt: str, output: Any,
              reference: str | None = None, task: Any = None,
              **kwargs: Any) -> float:
        """Match the real QualityScorer.score signature. The dispatcher
        passes the bare text string; techniques sometimes pass the full
        AgentOutput. Handle both."""
        self.call_count += 1
        if isinstance(output, str):
            text, prebaked = output, 0.0
        else:
            text = getattr(output, "text", "") or ""
            prebaked = getattr(output, "quality_score", 0.0) or 0.0
        if prebaked:
            return float(prebaked)
        # Stable but distinct: 0.5 + (len(text) mod 50) / 100
        return 0.5 + (len(text) % 50) / 100.0

    def score_comparative(
        self,
        prompt: str,
        candidate: Any,
        baseline: Any = None,
        baseline_score: float | None = None,
        reference: str | None = None,
        task: Any = None,
        **kwargs: Any,
    ) -> float:
        """Mirror QualityScorer.score_comparative: differential scoring
        of a single candidate against a baseline. Tests don't exercise
        the noise-cancellation arithmetic; we just return the candidate's
        independent score so the technique can make progress."""
        return self.score(prompt, candidate, reference=reference, task=task)

    def score_batch(
        self, prompt: str, outputs: list[AgentOutput],
        reference: str | None = None, task: Any = None,
    ) -> list[AgentOutput]:
        for o in outputs:
            o.quality_score = self.score(prompt, o.text, reference, task)
        return outputs

    def collect_judge_outputs(self) -> list[AgentOutput]:
        """Drain recorded judge outputs — mirrors QualityScorer.collect_judge_outputs."""
        out = list(self._judge_outputs)
        self._judge_outputs.clear()
        return out


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_channel() -> MockChannel:
    return MockChannel()


@pytest.fixture
def mock_channel_pair() -> list[MockChannel]:
    """Two channels — needed by diversity / spatial techniques."""
    return [
        MockChannel(model="mock/a", responses=[("answer from A", 0.78)]),
        MockChannel(model="mock/b", responses=[("answer from B", 0.83)]),
    ]


@pytest.fixture
def mock_scorer() -> MockScorer:
    return MockScorer()


@pytest.fixture
def qa_task() -> TaskItem:
    return TaskItem(
        id="t-1",
        category=TaskCategory.QA,
        prompt="What is the capital of France?",
        reference="Paris",
    )


@pytest.fixture
def dispatch_ctx(mock_channel_pair, mock_scorer):
    """A DispatchContext wired with two mock channels and a mock scorer."""
    from agentcodec.dispatch import DispatchContext
    channels = {ch.model: ch for ch in mock_channel_pair}
    return DispatchContext(
        channels=channels,
        scorer=mock_scorer,
        critic_channel=None,
    )
