"""
Technique 4: Fountain / Rateless Decoder

Communication analog: Fountain codes (LT, Raptor) generate an unlimited stream
of encoded symbols. The receiver collects symbols until it can decode — no fixed
code rate, adapts to channel conditions automatically.

Agent analog: Generate unlimited output samples and accumulate them until a
confidence threshold is met. The "decoding" is synthesis of accumulated samples.
This is rateless — we don't commit to a fixed number of attempts upfront.

Key property: Cost adapts to task difficulty. Easy tasks need few samples,
hard tasks automatically get more.
"""

from __future__ import annotations

import logging

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, ReliabilityRun, TaskItem

logger = logging.getLogger(__name__)


class FountainDecoder:
    """
    Rateless code: keep generating samples until confidence threshold is met.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        confidence_threshold: float = 0.85,
        max_samples: int = 10,
        min_samples: int = 2,
        agreement_threshold: float = 0.7,
        softmax_normalize: bool = True,
        softmax_temperature: float = 0.5,
    ):
        # softmax_normalize=True (default) applies CISC's Def 3.1 step 2 to
        # the per-sample judge scores used as ML-decoding weights in the
        # synthesis prompt (`_decode`). The threshold-gating logic
        # (`confidence_threshold`, `dominance_gap`, `quality_band`) is
        # *not* an aggregation and is unaffected. T = 0.5 matches the
        # judge-score default used by `DiversityMRCDiscreteN`.
        self.channels = channels
        self.scorer = scorer
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.agreement_threshold = agreement_threshold
        self.softmax_normalize = softmax_normalize
        self.softmax_temperature = softmax_temperature

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="fountain",
            config={
                "confidence_threshold": self.confidence_threshold,
                "max_samples": self.max_samples,
                "min_samples": self.min_samples,
                "num_channels": len(self.channels),
            },
        )

        outputs: list[AgentOutput] = []
        channel_idx = 0

        for sample_num in range(1, self.max_samples + 1):
            # Round-robin across channels (different "encoding symbols")
            channel = self.channels[channel_idx % len(self.channels)]
            channel_idx += 1

            # Vary temperature slightly for each sample (stochastic encoding)
            temp = 0.5 + (sample_num % 5) * 0.1
            out = channel.transmit(task.request, temperature=temp)
            out.quality_score = self.scorer.score(task.prompt, out.text, reference=task.reference, task=task)
            outputs.append(out)
            run.rounds = sample_num

            # Check if we can "decode" — enough samples with sufficient quality
            if sample_num >= self.min_samples:
                confidence = self._estimate_confidence(outputs)
                logger.info(
                    f"Fountain sample {sample_num}: confidence={confidence:.3f}, "
                    f"latest_quality={out.quality_score:.3f}"
                )
                if confidence >= self.confidence_threshold:
                    break

        # "Decode" — synthesize collected samples via ML decoding.
        run.individual_outputs = outputs
        best_output = max(outputs, key=lambda o: o.quality_score)
        decoded_text, synth_output = self._decode(outputs, task.prompt)

        if synth_output is not None:
            run.overhead_outputs = [synth_output]
            # Regression guard: score the synthesis, fall back to best
            # individual if synthesis didn't improve. This prevents
            # transcoding loss — LLM rewrites can lose checklist matches
            # even when semantically equivalent.
            synth_score = self.scorer.score(
                task.prompt, decoded_text,
                reference=task.reference, task=task,
            )
            if synth_score >= best_output.quality_score:
                run.combined_output = decoded_text
                run.final_quality = synth_score
                logger.info(
                    f"Fountain decode: synthesis accepted "
                    f"({synth_score:.3f} >= best {best_output.quality_score:.3f})"
                )
            else:
                run.combined_output = best_output.text
                run.final_quality = best_output.quality_score
                logger.info(
                    f"Fountain decode: synthesis rejected "
                    f"({synth_score:.3f} < best {best_output.quality_score:.3f}), "
                    f"keeping best individual"
                )
        else:
            # Fast path / single survivor — no synthesis call
            run.combined_output = decoded_text
            run.final_quality = best_output.quality_score
        run.compute_metrics()
        return run

    def _estimate_confidence(self, outputs: list[AgentOutput]) -> float:
        """
        Estimate decoding confidence based on:
        1. Mean quality of collected samples
        2. Agreement between top samples (consistency)
        """
        scores = [o.quality_score for o in outputs]
        mean_quality = sum(scores) / len(scores)

        # Agreement: how close are the top scores to each other?
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) >= 2:
            top_scores = sorted_scores[:max(2, len(sorted_scores) // 2)]
            score_range = max(top_scores) - min(top_scores)
            agreement = 1.0 - score_range  # higher agreement = scores are similar
        else:
            agreement = 0.5

        # Confidence is a blend of quality and agreement
        confidence = 0.6 * mean_quality + 0.4 * agreement
        return confidence

    def _decode(self, outputs: list[AgentOutput], prompt: str) -> tuple[str, AgentOutput | None]:
        """
        ML decoder over collected fountain symbols.

        Comm-theory analog: real fountain decoding is maximum-likelihood —
        high-confidence symbols drive the decoded word, erasure-marked /
        low-LLR symbols are DISCARDED, not blended. Earlier versions merged
        every top-half sample which pulled wrong facts from weak samples into
        the answer (net -0.024 vs best individual across 69 tasks).

        New behavior:
        1. Quality-gate: keep only samples within `quality_band` of the best
           (erasure marking — weak samples are dropped, not averaged in).
        2. Single-best fast path: if one sample dominates (gap > `dominance_gap`),
           return it directly — no synthesis call, no corruption risk.
        3. ML-style synthesis prompt: primary answer = best sample; only
           incorporate corroborated detail from surviving samples; on conflict,
           trust the higher-weighted sample unconditionally.
        """
        quality_band = 0.10     # keep samples within 0.10 of best
        dominance_gap = 0.20    # if best - 2nd > this, skip synthesis entirely

        sorted_outputs = sorted(outputs, key=lambda o: o.quality_score, reverse=True)
        best_score = sorted_outputs[0].quality_score

        # Single-best fast path: one sample dominates → ML decode = pick it.
        if len(sorted_outputs) >= 2:
            gap = best_score - sorted_outputs[1].quality_score
            if gap > dominance_gap:
                logger.info(
                    f"Fountain ML-decode fast path: best={best_score:.3f} "
                    f"dominates by {gap:.3f} → return best, no synthesis"
                )
                return sorted_outputs[0].text, None

        # Erasure-marking: drop samples more than quality_band below best.
        surviving = [o for o in sorted_outputs if o.quality_score >= best_score - quality_band]
        if len(surviving) == 1:
            return surviving[0].text, None

        raw_qualities = [o.quality_score for o in surviving]
        if self.softmax_normalize:
            from .soft import softmax_with_temperature
            display_weights = softmax_with_temperature(
                raw_qualities, self.softmax_temperature
            )
        else:
            total_weight = sum(raw_qualities) or 1.0
            display_weights = [q / total_weight for q in raw_qualities]
        weighted_parts = []
        for i, (o, w) in enumerate(zip(surviving, display_weights, strict=False)):
            weighted_parts.append(
                f"### Sample {i+1} [Weight: {w:.2f}, Quality: {o.quality_score:.2f}, Model: {o.model}]\n{o.text}"
            )

        synth_prompt = (
            f"## Original Task\n{prompt}\n\n"
            f"## Collected Samples (sorted by quality — Sample 1 is the strongest)\n\n"
            + "\n\n---\n\n".join(weighted_parts)
            + "\n\n## Maximum-Likelihood Decoding Instructions\n"
            "Sample 1 is your PRIMARY answer. Build the final answer FROM Sample 1.\n\n"
            "You may incorporate content from lower-weighted samples ONLY when:\n"
            "- It adds a detail that is clearly missing from Sample 1 AND\n"
            "- It does not contradict Sample 1\n\n"
            "**When samples conflict, trust the higher-weighted sample unconditionally. "
            "Do NOT average, blend, or hedge between conflicting claims.**\n\n"
            "If Sample 1 is already complete and correct, output it as-is — do NOT "
            "rewrite it just to show effort. Do NOT invent content not present in any sample.\n\n"
            "Output ONLY the final decoded answer."
        )

        synth_channel = self.channels[0]
        result = synth_channel.transmit(synth_prompt, temperature=0.2)
        return result.text, result
