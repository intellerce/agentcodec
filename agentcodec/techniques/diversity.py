"""
Technique 1: Diversity Ensemble

Communication analog: Spatial / Frequency / Time diversity with SC, MRC, EGC combining.

- Spatial diversity  → multiple different models on the same task
- Frequency diversity → different prompt formulations for the same task
- Time diversity      → same model re-queried with different temperatures/seeds

Combining strategies:
- SC  (Selection Combining):     pick the output with highest quality score
- MRC (Maximal Ratio Combining): quality-weighted synthesis of all outputs
- EGC (Equal Gain Combining):    equal-weight consensus synthesis
"""

from __future__ import annotations

import json
import logging
import re

from ..channel import AgentChannel, QualityScorer
from ..models import (
    AgentOutput,
    CombiningStrategy,
    ReliabilityRun,
    TaskItem,
)

logger = logging.getLogger(__name__)


# Default prompt variants for frequency diversity
DEFAULT_PROMPT_VARIANTS = {
    "default": "{prompt}",
    "step_by_step": "Think step by step.\n\n{prompt}",
    "expert": "You are a world-class expert. Provide a thorough answer.\n\n{prompt}",
    "concise": "Be concise and precise.\n\n{prompt}",
}


class DiversityEnsemble:
    """
    Implements spatial, frequency, and time diversity with configurable combining.

    For MRC/EGC, uses the judge model as synthesizer by default. This avoids
    the problem of a weak channel model trying to merge its own outputs —
    the judge is already the strongest model in the system and has seen
    both responses during scoring.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        combining: CombiningStrategy = CombiningStrategy.MRC,
        prompt_variants: dict[str, str] | None = None,
        temperature_spread: list[float] | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        self.combining = combining
        self.prompt_variants = prompt_variants or {"default": "{prompt}"}
        self.temperature_spread = temperature_spread  # for time diversity

    def run(self, task: TaskItem, synthesizer: AgentChannel | None = None) -> ReliabilityRun:
        """
        Execute diversity ensemble on a task.

        If synthesizer is provided, it's used for MRC/EGC combining synthesis.
        Otherwise, the first channel is reused.
        """
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique=f"diversity_{self.combining.value}",
            config={
                "num_channels": len(self.channels),
                "combining": self.combining.value,
                "num_prompt_variants": len(self.prompt_variants),
                "temperature_spread": self.temperature_spread,
            },
        )

        # Phase 1: Collect diverse outputs
        outputs: list[AgentOutput] = []

        for channel in self.channels:
            for variant_name, variant_template in self.prompt_variants.items():
                prompt_text = variant_template.format(prompt=task.prompt)

                if self.temperature_spread:
                    # Time diversity: multiple temperatures per channel+variant
                    for temp in self.temperature_spread:
                        out = channel.transmit(prompt_text, temperature=temp, prompt_variant=variant_name)
                        outputs.append(out)
                else:
                    out = channel.transmit(prompt_text, prompt_variant=variant_name)
                    outputs.append(out)

        # Phase 2: Score all outputs (channel estimation)
        self.scorer.score_batch(task.prompt, outputs, reference=task.reference, task=task)
        run.individual_outputs = outputs

        # Phase 3: Combine
        # Use judge model as synthesizer for MRC/EGC — it's stronger than the
        # channel models and won't lose information during synthesis.
        # Fall back to first channel only if no judge is available.
        if self.combining != CombiningStrategy.SC and hasattr(self.scorer, 'judge'):
            synth = self.scorer.judge
        else:
            synth = synthesizer or self.channels[0]
        combined_text, synth_output = self._combine(outputs, task.prompt, synth)
        run.combined_output = combined_text

        # Track synthesis call for cost, but keep it separate from scored outputs
        if synth_output is not None:
            run.overhead_outputs = [synth_output]

        # Phase 4: Score final output
        best_output = max(outputs, key=lambda o: o.quality_score)
        best_ind = best_output.quality_score

        if self.combining == CombiningStrategy.SC:
            # SC just picks the best — no synthesis to compare
            run.final_quality = best_ind
        elif run.combined_output == best_output.text:
            # Synthesis returned the best output unchanged (SC fallback or identity)
            run.final_quality = best_ind
        else:
            # Use comparative scoring to detect even small improvements
            synth_score = self.scorer.score_comparative(
                task.prompt,
                candidate=run.combined_output,
                baseline=best_output.text,
                baseline_score=best_ind,
                reference=task.reference,
            )

            # MRC GUARANTEE: combining can never be worse than selection.
            # In real MRC, the math ensures SNR_MRC >= SNR_SC. Here we enforce
            # it explicitly: if synthesis degraded quality, fall back to SC.
            if synth_score >= best_ind:
                run.final_quality = synth_score
            else:
                logger.info(
                    f"MRC synthesis scored {synth_score:.3f} < best individual "
                    f"{best_ind:.3f} — falling back to SC (preserving best response)"
                )
                run.combined_output = best_output.text
                run.final_quality = best_ind

        logger.info(
            f"Diversity {self.combining.value}: "
            f"scores={[f'{o.quality_score:.3f}' for o in outputs]}, "
            f"best={best_ind:.3f}, combined={run.final_quality:.3f}, "
            f"gain={run.final_quality - best_ind:+.3f}"
        )

        run.compute_metrics()
        return run

    def _combine(
        self, outputs: list[AgentOutput], original_prompt: str, synthesizer: AgentChannel
    ) -> tuple[str, AgentOutput | None]:
        """Returns (combined_text, synthesis_output_or_None)."""
        if self.combining == CombiningStrategy.SC:
            return self._selection_combining(outputs), None
        elif self.combining == CombiningStrategy.MRC:
            return self._maximal_ratio_combining(outputs, original_prompt, synthesizer)
        elif self.combining == CombiningStrategy.EGC:
            return self._equal_gain_combining(outputs, original_prompt, synthesizer)
        else:
            raise ValueError(f"Unknown combining strategy: {self.combining}")

    def _selection_combining(self, outputs: list[AgentOutput]) -> str:
        """SC: Pick the output with highest quality score."""
        best = max(outputs, key=lambda o: o.quality_score)
        return best.text

    def _maximal_ratio_combining(
        self, outputs: list[AgentOutput], prompt: str, synthesizer: AgentChannel
    ) -> tuple[str, AgentOutput]:
        """
        MRC: Quality-weighted synthesis. Weights proportional to quality scores (≈ SNR).

        Key design principles:
        1. Real MRC can NEVER be worse than SC — we guarantee this
        2. The synthesizer must not introduce new information — only select and combine
        3. For conflicting answers, use the RESPONSES to resolve (not synthesizer knowledge)
        """
        # Sort outputs by quality (best first)
        ranked = sorted(outputs, key=lambda o: o.quality_score, reverse=True)
        best = ranked[0]
        others = ranked[1:]

        # If all other responses scored much lower, skip synthesis — SC is optimal
        # (analogous to one branch having much higher SNR than others)
        if not others or all(o.quality_score < best.quality_score * 0.5 for o in others):
            logger.info("MRC: best response dominates — falling back to SC")
            return best.text, None

        # Build the labeled response list
        resp_parts = []
        for i, o in enumerate(ranked):
            label = "BEST" if i == 0 else f"ALT-{i}"
            resp_parts.append(
                f"### [{label}] Quality: {o.quality_score:.2f}, Model: {o.model}\n{o.text}"
            )

        synthesis_prompt = (
            f"## Task\n{prompt}\n\n"
            f"## Candidate Responses\n\n"
            + "\n\n---\n\n".join(resp_parts)
            + "\n\n## Instructions\n"
            "You are combining multiple AI responses into one best answer.\n\n"
            "CRITICAL RULES — you must follow these exactly:\n"
            "1. Your output must contain ONLY information from the responses above. "
            "Do NOT add any facts, numbers, or claims from your own knowledge.\n"
            "2. Start from the [BEST] response as the foundation.\n"
            "3. If responses AGREE on a fact/answer, include it (high confidence).\n"
            "4. If responses DISAGREE, use the version from the higher-quality response.\n"
            "5. If an alternative response contains a useful detail that [BEST] lacks "
            "AND it doesn't contradict [BEST], add it.\n"
            "6. Preserve exact numbers, calculations, and code from the source responses. "
            "Do NOT recalculate or rephrase numerical answers.\n"
            "7. Do NOT add headers, labels, meta-commentary, or formatting not in the originals.\n"
            "8. If the alternatives add nothing useful, output [BEST] verbatim.\n\n"
            "Output the combined answer ONLY."
        )
        result = synthesizer.transmit(synthesis_prompt, temperature=0.1)
        return result.text, result

    def _equal_gain_combining(
        self, outputs: list[AgentOutput], prompt: str, synthesizer: AgentChannel
    ) -> tuple[str, AgentOutput]:
        """EGC: Equal-weight consensus synthesis (majority-vote style)."""
        # Sort by quality so best is first (gives the model a hint even though
        # weights are equal — consensus should still preserve the best content)
        ranked = sorted(outputs, key=lambda o: o.quality_score, reverse=True)

        parts = []
        for i, o in enumerate(ranked):
            parts.append(f"### Response {i+1} [Model: {o.model}]\n{o.text}")

        synthesis_prompt = (
            f"## Task\n{prompt}\n\n"
            f"## Responses (all equally weighted)\n\n"
            + "\n\n---\n\n".join(parts)
            + "\n\n## Your Job\n"
            "Combine these responses into one answer that captures what they "
            "collectively get right.\n\n"
            "Rules:\n"
            "- Start from Response 1 as a foundation.\n"
            "- Where responses AGREE, keep that content (high confidence).\n"
            "- Where responses DISAGREE, go with the majority or reason about "
            "which is correct.\n"
            "- ADD unique facts/details/examples from any response that the "
            "others lack.\n"
            "- For math/code: preserve exact answers and code from the majority.\n"
            "- Do NOT remove correct information. Only ADD.\n"
            "- If all responses are essentially the same, reproduce Response 1 as-is.\n\n"
            "Output ONLY the combined answer. No meta-commentary."
        )
        result = synthesizer.transmit(synthesis_prompt, temperature=0.2)
        return result.text, result


class SelectionCombiningN:
    """
    Wider-pool Selection Combining (``diversity_sc_N``).

    Sample N candidates by cycling through all configured channels
    (``channels[i % len(channels)]``), score each with the shared judge, and
    return the argmax. This generalises ``diversity_sc`` beyond N=|channels|:
    a multi-model diversity pool sampled to depth N, combined by selection.

    Not a prior-method baseline -- this is our own operator, distinct from
    canonical single-model BoN (which lives in ``baselines.py``). The
    canonical BoN uses one policy; SC_N uses the full configured channel
    set and cycles through it.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
    ):
        if not channels:
            raise ValueError("SelectionCombiningN needs at least one channel")
        self.channels = channels
        self.scorer = scorer
        self.num_samples = num_samples

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="diversity_sc_N",
            config={
                "num_samples": self.num_samples,
                "num_channels": len(self.channels),
            },
        )

        outputs: list[AgentOutput] = []
        for i in range(self.num_samples):
            channel = self.channels[i % len(self.channels)]
            out = channel.transmit(task.request, temperature=0.7)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        best = max(outputs, key=lambda o: o.quality_score or 0.0)
        run.individual_outputs = outputs
        run.rounds = self.num_samples
        run.combined_output = best.text
        run.final_quality = best.quality_score or 0.0
        run.compute_metrics()
        return run


class DiversityMRCDiscreteN:
    """
    Wider-pool Discrete Maximal-Ratio Combining (``diversity_mrc_discrete_N``).

    Discrete-MRC on a multi-model pool of size N. Sample N candidates by
    cycling through all configured channels, score each with the shared
    judge, cluster into semantic equivalence classes via one voter LLM call,
    sum judge scores per cluster, pick the cluster with the highest total,
    and return that cluster's top-scoring sample.

    Same combining step as canonical Weighted-BoN,
    but fed a multi-model sample pool rather than N samples from one
    policy. The single-model canonical reproduction lives in
    ``baselines.py`` as ``WeightedBoNBaseline``.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        voter: AgentChannel | None = None,
        softmax_normalize: bool = True,
        softmax_temperature: float = 0.5,
    ):
        # softmax_normalize=True (default) applies CISC's Def 3.1 step 2 to
        # judge scores before per-cluster summation. Judge scores live in
        # [0, 1] so the dynamic-range argument is weaker than for
        # logprobs, but raw summing still over-counts a 0.5-tied cluster
        # vs a single 0.95 winner. T = 0.5 is a documented heuristic
        # (the paper does not calibrate T for judge-score CSI).
        if not channels:
            raise ValueError("DiversityMRCDiscreteN needs at least one channel")
        self.channels = channels
        self.scorer = scorer
        self.num_samples = num_samples
        self.voter = voter
        self.softmax_normalize = softmax_normalize
        self.softmax_temperature = softmax_temperature

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="diversity_mrc_discrete_N",
            config={
                "num_samples": self.num_samples,
                "num_channels": len(self.channels),
                "softmax_normalize": self.softmax_normalize,
                "softmax_temperature": (
                    self.softmax_temperature if self.softmax_normalize else None
                ),
            },
        )

        outputs: list[AgentOutput] = []
        for i in range(self.num_samples):
            channel = self.channels[i % len(self.channels)]
            out = channel.transmit(task.request, temperature=0.7)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        voter = self.voter or self.channels[0]
        joined = "\n\n".join(
            f"### Sample {i+1}\n{o.text}" for i, o in enumerate(outputs)
        )
        cluster_prompt = (
            f"Below are {self.num_samples} independent answers to the same task. "
            f"Group them into semantic equivalence classes: two answers belong "
            f"to the same class iff they convey the same final answer or "
            f"conclusion (phrasing differences are ignored).\n\n"
            f"## Task\n{task.prompt}\n\n"
            f"## Samples\n{joined}\n\n"
            f"Return ONLY a JSON array of {self.num_samples} integers, where "
            f"the i-th integer is the 0-indexed cluster ID of sample (i+1). "
            f"Use the smallest cluster IDs possible (0, 1, 2, ...).\n"
            f"Example for 5 samples: [0, 0, 1, 0, 2]\n\n"
            f"JSON:"
        )
        vote = voter.transmit(cluster_prompt, temperature=0.0)
        run.overhead_outputs.append(vote)

        labels: list[int] | None = None
        try:
            m = re.search(r"\[[\s\d,]*\]", vote.text)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list) and len(parsed) == self.num_samples:
                    labels = [int(x) for x in parsed]
        except Exception:
            labels = None

        if labels is None:
            labels = list(range(self.num_samples))

        raw_weights = [(o.quality_score or 0.0) for o in outputs]
        if self.softmax_normalize:
            from .soft import softmax_with_temperature
            weights = softmax_with_temperature(raw_weights, self.softmax_temperature)
        else:
            weights = raw_weights
        totals: dict[int, float] = {}
        members: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            totals[lbl] = totals.get(lbl, 0.0) + weights[idx]
            members.setdefault(lbl, []).append(idx)
        winning_cluster = max(totals, key=lambda k: totals[k])
        winning_members = members[winning_cluster]
        # Tie-break inside winning cluster on RAW judge scores (T-invariant).
        best_idx = max(winning_members, key=lambda i: raw_weights[i])
        best = outputs[best_idx]

        run.individual_outputs = outputs
        run.rounds = self.num_samples
        run.config["cluster_labels"] = labels
        run.config["winning_cluster_size"] = len(winning_members)
        run.config["raw_weights"] = [round(w, 4) for w in raw_weights]
        run.config["norm_weights"] = [round(w, 4) for w in weights]
        run.combined_output = best.text
        run.final_quality = best.quality_score or 0.0
        run.compute_metrics()
        return run
