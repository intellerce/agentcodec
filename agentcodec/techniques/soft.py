"""
Soft-Output Techniques: Logprob-Based Signal Processing

Communication analog: Real receivers use soft-output demodulation — each
received symbol carries a log-likelihood ratio (LLR) indicating the
decoder's confidence in each bit. Hard-decision decoders discard this
information (keeping only the bit value), losing ~2 dB of coding gain.
Soft-decision decoders exploit LLRs for dramatically better performance.

Agent analog: LLMs can output token-level log-probabilities. These are
directly analogous to per-symbol LLRs. The existing techniques in this
framework use "hard decisions" — message-level quality scores from a judge.
The soft variants here use token-level logprobs for:

- **SoftDiversityMRC**: MRC combining weighted by model confidence (mean
  logprob per branch) instead of judge scores. Saves judge calls on the
  combining step and uses the model's own certainty as the SNR estimate.

- **SoftFountainDecoder**: Logprob-based confidence estimation and erasure
  marking. Samples with low mean logprob are treated as erasures (unreliable
  symbols) rather than relying on judge scores for gating.

- **SoftACMRouter**: Channel quality estimation from generation logprobs.
  Instead of a separate LLM call to estimate task difficulty, a short probe
  generation's mean logprob IS the CQI — low logprob = hard channel = route
  to higher protection. Zero overhead for channel estimation.

These are OPTIONAL modes. The existing hard-decision techniques are untouched.
Technique names use a "_soft" suffix so they appear alongside their hard
counterparts in results and plots.

Requires: backend that supports logprobs (Ollama, vLLM, OpenAI).
Anthropic does not currently expose logprobs — soft techniques will
gracefully degrade to hard-decision behavior if logprobs are unavailable.
"""

from __future__ import annotations

import logging
import math
import re

from ..channel import AgentChannel, QualityScorer
from ..models import (
    AgentOutput,
    CombiningStrategy,
    HARQMode,
    ReliabilityRun,
    TaskItem,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_logprobs(output: AgentOutput) -> bool:
    """Check if an output has usable logprob data."""
    return output.mean_logprob is not None


def _logprob_to_confidence(mean_logprob: float) -> float:
    """
    Convert mean log-probability to a [0, 1] confidence score.

    Mapping: mean_logprob is typically in [-inf, 0]. Values close to 0 mean
    high confidence (probability near 1). We use exp(mean_logprob) which
    maps to geometric mean token probability — a natural confidence measure.

    Typical ranges (empirical):
      mean_logprob ~ -0.1  → confidence ~ 0.90  (very confident)
      mean_logprob ~ -0.5  → confidence ~ 0.61  (moderate)
      mean_logprob ~ -1.0  → confidence ~ 0.37  (uncertain)
      mean_logprob ~ -2.0  → confidence ~ 0.14  (very uncertain)
    """
    return math.exp(mean_logprob)


# ---------------------------------------------------------------------------
# Verbal confidence (single-step variant of Taubenfeld et al. 2025, App. B)
# ---------------------------------------------------------------------------
#
# The CISC paper's Verbal-100 extraction (Table 5) instructs the model to rate
# its confidence on a 0-100 scale. The two-step variant in the paper appends a
# follow-up prompt and reads the next token; the single-step variant
# (Appendix B) lifts that instruction into the original prompt and parses the
# score from the model's own response. We use the single-step variant
# universally because it requires no logprob exposure and works on any
# backend, including Anthropic.
#
# Output range matches the paper's column (raw integer 0-100), so the
# calibrated T values from Figure 8's Verbal 0-100 column can be used
# unchanged as `softmax_temperature` overrides.

VERBAL_100_SUFFIX = (
    "\n\nAfter your final answer, on a new line, write exactly:\n"
    "Confidence: <X>\n"
    "where <X> is your self-assessed confidence on a scale of 0 to 100 "
    "(0 = certainly wrong, 100 = certainly correct). Use only an integer."
)

_VERBAL_100_RE = re.compile(
    r"(?im)^\s*confidence\s*[:=]\s*(\d{1,3})\s*$"
)


def append_verbal_confidence_prompt(prompt: str, scale: int = 100) -> str:
    """Augment a task prompt to elicit a single-step verbal confidence score.

    Currently only `scale=100` is implemented (matches Figure 8's Verbal
    0-100 column and the paper's strongest verbal variant).
    """
    if scale != 100:
        raise NotImplementedError(
            f"Only scale=100 is supported; got {scale}. "
            f"Verbal Binary (scale=2) and other scales would need their own "
            f"prompt and parser."
        )
    return prompt + VERBAL_100_SUFFIX


def softmax_with_temperature(values: list[float], T: float) -> list[float]:
    """CISC's Def 3.1, step 2: c̃_i = exp(c_i/T) / Σ_j exp(c_j/T).

    Numerically stabilized with the standard max-shift trick. Used by the
    soft combining techniques to convert raw weights (logprobs, judge
    scores, verbal confidence) into a calibrated probability distribution
    before per-cluster summation or display to a synthesizer.

    The paper's Appendix C explicitly notes that softmax normalization
    *without* temperature scaling is "strongly discouraged"; the right T
    interpolates between vanilla majority vote (T → ∞) and pure
    argmax-by-confidence (T → 0). Calibrated T values from Figure 8:
        Response Probability (logprob) → median 0.1
        Verbal 0-100                   → median 8
        Judge score                    → not calibrated by the paper;
                                         T ≈ 0.5 is a reasonable default
                                         given judge scores live in [0,1].

    On empty input, returns an empty list. On a singleton, returns [1.0].
    """
    if not values:
        return []
    if T <= 0:
        raise ValueError(f"softmax temperature must be > 0; got {T}")
    if len(values) == 1:
        return [1.0]
    scaled = [v / T for v in values]
    shift = max(scaled)
    exps = [math.exp(s - shift) for s in scaled]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


def parse_verbal_confidence(text: str, scale: int = 100) -> float | None:
    """Parse the trailing `Confidence: <int>` line from a model response.

    Returns the raw integer (clamped to [0, scale]) when found, else None.
    Returns None — *not* a default — so that callers can decide whether to
    treat a missing/unparseable confidence as "skip this sample" or as a
    sentinel value. This mirrors the paper's filter-empty-confidences
    convention in `aggregators.majority_with_conf`.
    """
    if scale != 100:
        raise NotImplementedError(f"Only scale=100 is supported; got {scale}.")
    if not text:
        return None
    # Search bottom-up — the score is supposed to be on the last line, but
    # some models add whitespace or trailing notes after it.
    matches = list(_VERBAL_100_RE.finditer(text))
    if not matches:
        return None
    raw = int(matches[-1].group(1))
    return float(max(0, min(scale, raw)))


# ---------------------------------------------------------------------------
# Soft Diversity MRC
# ---------------------------------------------------------------------------

class SoftDiversityMRC:
    """
    Maximal Ratio Combining with logprob-derived SNR weights.

    In real MRC, each diversity branch is weighted by its SNR. The hard-decision
    version uses judge scores as the weight proxy. This soft version uses the
    model's own token-level confidence (mean logprob) — a direct analog of
    per-branch SNR measurement at the receiver.

    Advantages over hard MRC:
    - Weights come from the model itself (no judge call for weighting)
    - Confidence is measured at generation time, not post-hoc
    - Per-token logprobs capture uncertainty the judge can't see
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        prompt_variants: dict[str, str] | None = None,
        softmax_normalize: bool = True,
        softmax_temperature: float = 0.1,
    ):
        # softmax_normalize=True (default) maps raw `exp(mean_logprob)`
        # confidences through CISC's Def 3.1 normalization before display
        # to the synthesizer. This fixes a numerical underflow bug for
        # long responses (where exp(mean_logprob) collapses to ~1e-100)
        # and gives the synthesizer interpretable relative weights.
        # softmax_normalize=False reproduces the legacy raw-prob behavior.
        # T = 0.1 matches the paper's Figure 8 median for Response
        # Probability extraction.
        self.channels = channels
        self.scorer = scorer
        self.prompt_variants = prompt_variants or {"default": "{prompt}"}
        self.softmax_normalize = softmax_normalize
        self.softmax_temperature = softmax_temperature

    def run(self, task: TaskItem, synthesizer: AgentChannel | None = None) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="diversity_mrc_soft",
            config={
                "num_channels": len(self.channels),
                "combining": "mrc_soft",
                "num_prompt_variants": len(self.prompt_variants),
                "softmax_normalize": self.softmax_normalize,
                "softmax_temperature": (
                    self.softmax_temperature if self.softmax_normalize else None
                ),
            },
        )

        # Phase 1: Generate branches WITH logprobs
        outputs: list[AgentOutput] = []
        for channel in self.channels:
            for variant_name, variant_template in self.prompt_variants.items():
                prompt_text = variant_template.format(prompt=task.prompt)
                out = channel.transmit(
                    prompt_text,
                    prompt_variant=variant_name,
                    request_logprobs=True,
                )
                outputs.append(out)

        # Phase 2: Score all outputs (still needed for final_quality measurement)
        self.scorer.score_batch(task.prompt, outputs, reference=task.reference, task=task)
        run.individual_outputs = outputs

        # Require logprobs — no fallback to hard decisions.
        missing = [o for o in outputs if not _has_logprobs(o)]
        if missing:
            models = {o.model for o in missing}
            raise RuntimeError(
                f"SoftDiversityMRC requires logprobs but backend returned none "
                f"for {len(missing)}/{len(outputs)} outputs (models: {models}). "
                f"Ensure your backend supports logprobs (Ollama, vLLM, OpenAI)."
            )

        # Phase 3: Soft MRC combining — weights from logprobs
        raw_weights = [_logprob_to_confidence(o.mean_logprob) for o in outputs]
        if self.softmax_normalize:
            # Paper Def 3.1, step 2 — restores meaningful spread when raw
            # weights collapse near zero (long responses, low mean_logprob).
            weights = softmax_with_temperature(raw_weights, self.softmax_temperature)
        else:
            weights = raw_weights
        run.config["weight_source"] = "logprob"
        run.config["raw_weights"] = [round(w, 6) for w in raw_weights]

        # Log soft SNR estimates
        for o, w in zip(outputs, weights, strict=False):
            lp = f"logprob={o.mean_logprob:.3f}" if _has_logprobs(o) else "no_logprobs"
            logger.info(
                f"SoftMRC branch: model={o.model} {lp} "
                f"conf={w:.3f} judge={o.quality_score:.3f}"
            )

        # Store weights in config for analysis
        run.config["branch_weights"] = weights
        run.config["branch_logprobs"] = [
            o.mean_logprob if _has_logprobs(o) else None for o in outputs
        ]

        # Synthesize using logprob-weighted prompt
        synth = synthesizer or (
            self.scorer.judge if hasattr(self.scorer, 'judge') else self.channels[0]
        )
        combined_text, synth_output = self._soft_combine(outputs, weights, task.prompt, synth)
        run.combined_output = combined_text
        if synth_output is not None:
            run.overhead_outputs = [synth_output]

        # Phase 4: Score final output with MRC guarantee (combining >= selection)
        best_output = max(outputs, key=lambda o: o.quality_score)
        best_ind = best_output.quality_score

        if synth_output is None:
            # Fast path: one branch dominated
            run.final_quality = best_ind
        else:
            synth_score = self.scorer.score_comparative(
                task.prompt,
                candidate=run.combined_output,
                baseline=best_output.text,
                baseline_score=best_ind,
                reference=task.reference,
            )
            if synth_score >= best_ind:
                run.final_quality = synth_score
            else:
                logger.info(
                    f"SoftMRC synthesis scored {synth_score:.3f} < best {best_ind:.3f} "
                    f"— falling back to SC"
                )
                run.combined_output = best_output.text
                run.final_quality = best_ind

        run.compute_metrics()
        return run

    def _soft_combine(
        self,
        outputs: list[AgentOutput],
        weights: list[float],
        prompt: str,
        synthesizer: AgentChannel,
    ) -> tuple[str, AgentOutput | None]:
        """MRC combining with soft (logprob-derived) weights."""
        # Pair and sort by weight descending
        paired = sorted(zip(outputs, weights, strict=False), key=lambda x: x[1], reverse=True)
        best_out, best_w = paired[0]

        # If best branch dominates (weight > 2× all others), skip synthesis
        if len(paired) > 1 and all(best_w > 2 * w for _, w in paired[1:]):
            logger.info("SoftMRC: best branch dominates by >2× — SC fast path")
            return best_out.text, None

        total_w = sum(w for _, w in paired) or 1.0
        resp_parts = []
        for i, (o, w) in enumerate(paired):
            norm_w = w / total_w
            label = "PRIMARY" if i == 0 else f"BRANCH-{i+1}"
            lp_str = f", Logprob: {o.mean_logprob:.2f}" if _has_logprobs(o) else ""
            resp_parts.append(
                f"### [{label}] Weight: {norm_w:.2f}{lp_str}, "
                f"Model: {o.model}\n{o.text}"
            )

        synthesis_prompt = (
            f"## Task\n{prompt}\n\n"
            f"## Candidate Responses (weighted by model confidence)\n\n"
            + "\n\n---\n\n".join(resp_parts)
            + "\n\n## Soft MRC Combining Instructions\n"
            "Responses are ranked by the model's own confidence (log-probability).\n"
            "Higher weight = the model was more certain of its output.\n\n"
            "Rules:\n"
            "1. Start from [PRIMARY] — the highest-confidence response.\n"
            "2. When responses AGREE, include the content (cross-validated signal).\n"
            "3. When responses DISAGREE, trust the higher-weight response.\n"
            "4. Add unique details from lower branches ONLY if they don't conflict "
            "with the primary and are plausible.\n"
            "5. Do NOT add information not present in any response.\n"
            "6. If primary is already complete, output it as-is.\n\n"
            "Output the combined answer ONLY."
        )
        result = synthesizer.transmit(synthesis_prompt, temperature=0.1)
        return result.text, result


# ---------------------------------------------------------------------------
# Soft Diversity MRC on the discrete answer space
# ---------------------------------------------------------------------------

class SoftDiversityMRCDiscreteN:
    """
    Multi-model discrete-MRC with intrinsic logprob CSI.

    The strict superset of CISC and ``DiversityMRCDiscreteN``:
      * pool: multi-model channel set sampled in round-robin (like
        ``DiversityMRCDiscreteN``)
      * CSI:  per-sample token-logprob confidence c_i = exp(mean_logprob_i)
              (like CISC), no judge call on the combining path

    Operator: cluster the N samples via one voter LLM call, sum c_i within
    each cluster, return the top-c_i member of the cluster with the largest
    total. Collapses to CISC at |channels|=1, and to DiversityMRCDiscreteN
    if confidence weights are replaced with judge scores.

    Caveat (documented for honesty rather than fixed in code): logprobs from
    different model families are not on the same scale -- DeepSeek-R1's
    mean_logprob is not directly comparable to Phi-3's. Summing c_i across
    models inside a cluster is therefore a heuristic, not a calibrated
    weighting. CISC sidesteps this because all N samples come from one
    policy. Whether this matters empirically is an open question this
    operator is intended to answer.

    Backend constraint: every channel in the pool must expose logprobs.
    Anthropic backends are unsupported and raise RuntimeError; mixed
    Anthropic+Ollama configurations should drop the Anthropic channel
    before running this technique.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        voter: AgentChannel | None = None,
        softmax_normalize: bool = True,
        softmax_temperature: float = 0.1,
    ):
        # softmax_normalize=True (default) applies CISC's Def 3.1 step 2 to
        # per-sample confidences before per-cluster summation. Beyond the
        # numerical-stability win, it partly mitigates the cross-model
        # logprob-scale-mismatch flagged in the class docstring: even
        # though Mistral-12B's mean_logprob is not directly comparable to
        # Qwen-72B's, softmax normalizes within the sample set so one
        # model's underflow doesn't silently zero out its contribution.
        if not channels:
            raise ValueError("SoftDiversityMRCDiscreteN needs at least one channel")
        self.channels = channels
        self.scorer = scorer
        self.num_samples = num_samples
        self.voter = voter
        self.softmax_normalize = softmax_normalize
        self.softmax_temperature = softmax_temperature

    def run(self, task: TaskItem) -> ReliabilityRun:
        import json
        import re
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="diversity_mrc_discrete_N_soft",
            config={
                "num_samples": self.num_samples,
                "num_channels": len(self.channels),
                "csi_source": "logprob",
                "softmax_normalize": self.softmax_normalize,
                "softmax_temperature": (
                    self.softmax_temperature if self.softmax_normalize else None
                ),
            },
        )

        outputs: list[AgentOutput] = []
        for i in range(self.num_samples):
            channel = self.channels[i % len(self.channels)]
            out = channel.transmit(
                task.prompt, temperature=0.7, request_logprobs=True,
            )
            # Final-quality bookkeeping needs a judge score per sample, but
            # the *combining weight* is the logprob (no judge call on the
            # MRC path itself).
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        # Hard fail on missing logprobs: this technique is defined by its
        # CSI source, so a silent fallback to judge weights would defeat
        # the comparison with DiversityMRCDiscreteN.
        missing = [o for o in outputs if not _has_logprobs(o)]
        if missing:
            models = {o.model for o in missing}
            raise RuntimeError(
                f"SoftDiversityMRCDiscreteN requires token logprobs, but "
                f"backend returned none for {len(missing)}/{len(outputs)} "
                f"samples (models: {models}). Drop the offending channel(s) "
                f"or run diversity_mrc_discrete_N (judge-CSI) instead."
            )

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
            # Voter parse failure -> each sample is its own cluster, so
            # the operator degenerates to argmax-confidence selection
            # (BoN with intrinsic CSI on a multi-model pool).
            labels = list(range(self.num_samples))

        raw_confidences = [_logprob_to_confidence(o.mean_logprob) for o in outputs]
        if self.softmax_normalize:
            confidences = softmax_with_temperature(
                raw_confidences, self.softmax_temperature
            )
        else:
            confidences = raw_confidences
        # Tie-break inside winning cluster uses RAW confidences so the
        # ranking within a cluster doesn't depend on T (softmax is
        # monotonic, but we want the absolute-most-confident sample).
        totals: dict[int, float] = {}
        members: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            totals[lbl] = totals.get(lbl, 0.0) + confidences[idx]
            members.setdefault(lbl, []).append(idx)
        winning_cluster = max(totals, key=lambda k: totals[k])
        winning_members = members[winning_cluster]
        best_idx = max(winning_members, key=lambda i: raw_confidences[i])
        best = outputs[best_idx]

        run.individual_outputs = outputs
        run.rounds = self.num_samples
        run.config["cluster_labels"] = labels
        run.config["winning_cluster_size"] = len(winning_members)
        run.config["raw_confidences"] = [round(c, 6) for c in raw_confidences]
        run.config["norm_confidences"] = [round(c, 4) for c in confidences]
        run.combined_output = best.text
        run.final_quality = best.quality_score or 0.0
        run.compute_metrics()
        return run


# ---------------------------------------------------------------------------
# Soft Fountain Decoder
# ---------------------------------------------------------------------------

class SoftFountainDecoder:
    """
    Rateless fountain decoder with logprob-based soft decisions.

    Differences from hard FountainDecoder:
    - Confidence estimation uses mean logprob (model's own certainty)
      instead of judge scores. This is a free signal — no extra LLM call.
    - Erasure marking uses logprob threshold instead of quality-score band.
    - Samples with very low logprob are treated as erasures and excluded
      from synthesis, analogous to how real fountain decoders discard
      symbols with erasure flags.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        confidence_threshold: float = 0.85,
        max_samples: int = 10,
        min_samples: int = 2,
    ):
        self.channels = channels
        self.scorer = scorer
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.min_samples = min_samples

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="fountain_soft",
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
            channel = self.channels[channel_idx % len(self.channels)]
            channel_idx += 1

            temp = 0.5 + (sample_num % 5) * 0.1
            out = channel.transmit(
                task.prompt, temperature=temp, request_logprobs=True,
            )
            if not _has_logprobs(out):
                raise RuntimeError(
                    f"SoftFountainDecoder requires logprobs but backend "
                    f"returned none for model {out.model}. "
                    f"Ensure your backend supports logprobs (Ollama, vLLM, OpenAI)."
                )
            # Still score with judge (needed for final_quality and comparison)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)
            run.rounds = sample_num

            if sample_num >= self.min_samples:
                confidence = self._estimate_confidence_soft(outputs)
                logger.info(
                    f"Fountain-soft sample {sample_num}: confidence={confidence:.3f}, "
                    f"logprob={out.mean_logprob:.3f}, "
                    f"judge={out.quality_score:.3f}"
                )
                if confidence >= self.confidence_threshold:
                    break

        run.individual_outputs = outputs

        # Decode with soft erasure marking
        best_output = max(outputs, key=lambda o: o.quality_score)
        decoded_text, synth_output = self._decode_soft(outputs, task.prompt)

        if synth_output is not None:
            run.overhead_outputs = [synth_output]
            synth_score = self.scorer.score(
                task.prompt, decoded_text,
                reference=task.reference, task=task,
            )
            if synth_score >= best_output.quality_score:
                run.combined_output = decoded_text
                run.final_quality = synth_score
            else:
                run.combined_output = best_output.text
                run.final_quality = best_output.quality_score
        else:
            run.combined_output = decoded_text
            run.final_quality = best_output.quality_score

        run.compute_metrics()
        return run

    def _estimate_confidence_soft(self, outputs: list[AgentOutput]) -> float:
        """
        Soft confidence estimation using logprobs.

        Blends model confidence (logprob) with agreement (consistency of
        confidence across samples). All outputs must have logprobs.
        """
        confidences = [_logprob_to_confidence(o.mean_logprob) for o in outputs]
        mean_conf = sum(confidences) / len(confidences)

        # Agreement: how consistent are the logprob-confidences?
        if len(confidences) >= 2:
            sorted_c = sorted(confidences, reverse=True)
            spread = sorted_c[0] - sorted_c[-1]
            agreement = 1.0 - min(spread, 1.0)
        else:
            agreement = 0.5

        return 0.6 * mean_conf + 0.4 * agreement

    def _decode_soft(
        self, outputs: list[AgentOutput], prompt: str,
    ) -> tuple[str, AgentOutput | None]:
        """
        Soft ML decoder with logprob-based erasure marking.

        Samples are ranked by mean logprob (model confidence). Low-confidence
        samples are erased (excluded). High-confidence samples drive the decode.
        """
        # Sort by logprob confidence (best = highest mean_logprob, closest to 0)
        sorted_outputs = sorted(outputs, key=lambda o: o.mean_logprob, reverse=True)
        best = sorted_outputs[0]

        # Erasure marking: compute confidence threshold from best sample
        best_conf = _logprob_to_confidence(best.mean_logprob)
        # Keep samples within 50% of best confidence (generous band)
        erasure_threshold = best_conf * 0.5
        surviving = [
            o for o in sorted_outputs
            if _logprob_to_confidence(o.mean_logprob) >= erasure_threshold
        ]
        if not surviving:
            surviving = [best]

        erased = len(sorted_outputs) - len(surviving)
        if erased > 0:
            logger.info(
                f"Fountain-soft: erased {erased}/{len(sorted_outputs)} "
                f"samples below confidence threshold {erasure_threshold:.3f}"
            )

        # Single survivor or dominant best → return directly
        if len(surviving) == 1:
            return surviving[0].text, None

        # Dominance check
        if _has_logprobs(surviving[0]) and _has_logprobs(surviving[1]):
            gap = surviving[0].mean_logprob - surviving[1].mean_logprob
            if gap > 0.5:  # ~1.65× more confident
                return surviving[0].text, None

        # Build synthesis prompt with logprob weights
        total_conf = sum(
            _logprob_to_confidence(o.mean_logprob) for o in surviving
        ) or 1.0

        weighted_parts = []
        for i, o in enumerate(surviving):
            conf = _logprob_to_confidence(o.mean_logprob)
            weight = conf / total_conf
            meta = f"Confidence: {conf:.2f}, Logprob: {o.mean_logprob:.2f}"
            weighted_parts.append(
                f"### Sample {i+1} [Weight: {weight:.2f}, {meta}, "
                f"Model: {o.model}]\n{o.text}"
            )

        synth_prompt = (
            f"## Original Task\n{prompt}\n\n"
            f"## Collected Samples (ranked by model confidence)\n\n"
            + "\n\n---\n\n".join(weighted_parts)
            + "\n\n## Soft ML Decoding Instructions\n"
            "Samples are ranked by the model's own confidence in its output. "
            "Low-confidence samples have been erased (excluded).\n\n"
            "Sample 1 is your PRIMARY answer. Build the final answer FROM Sample 1.\n\n"
            "You may incorporate content from other samples ONLY when:\n"
            "- It adds a detail clearly missing from Sample 1 AND\n"
            "- It does not contradict Sample 1\n\n"
            "When samples conflict, trust the higher-confidence sample. "
            "If Sample 1 is already complete, output it as-is.\n\n"
            "Output ONLY the final decoded answer."
        )

        synth_channel = self.channels[0]
        result = synth_channel.transmit(synth_prompt, temperature=0.2)
        return result.text, result


# ---------------------------------------------------------------------------
# Soft ACM Router
# ---------------------------------------------------------------------------

class SoftACMRouter:
    """
    Adaptive Coding & Modulation with logprob-based CQI.

    In real wireless, ACM selects modulation/coding based on channel quality
    measured from pilot symbols. The hard ACM uses a separate LLM call to
    estimate task difficulty — an extra cost with its own error rate.

    Soft ACM uses a short "probe" generation WITH logprobs as the pilot.
    The mean logprob of the probe IS the channel quality indicator:
    - High logprob (close to 0) → model is confident → easy channel → less protection
    - Low logprob (very negative) → model is uncertain → hard channel → more protection

    This saves the difficulty-estimation call AND provides a more direct
    measurement of actual channel quality (the model's own uncertainty)
    rather than a meta-judgment about difficulty.
    """

    def __init__(
        self,
        channels: dict[str, AgentChannel],
        scorer: QualityScorer,
        acm_profiles: list[dict] | None = None,
        category_profiles: dict[str, list[dict]] | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        # Default fallback profiles (CQI-only, used when no category table matches).
        # Calibrated on cache2 oracle winners: fountain dominates qa/reasoning,
        # diversity_egc dominates creative, harq_cc dominates code. FEC-only
        # profiles were removed after they underperformed on every category.
        self.profiles = acm_profiles or [
            {
                "name": "MCS-0: High-confidence fountain",
                "confidence_range": (0.70, 1.0),
                "technique": "fountain",
                "num_branches": 2,
            },
            {
                "name": "MCS-1: Moderate fountain",
                "confidence_range": (0.50, 0.70),
                "technique": "fountain",
                "num_branches": 2,
            },
            {
                "name": "MCS-2: HARQ-IR",
                "confidence_range": (0.30, 0.50),
                "technique": "harq_ir",
                "max_rounds": 3,
            },
            {
                "name": "MCS-3: Full Protection",
                "confidence_range": (0.0, 0.30),
                "technique": "diversity_mrc",
                "num_branches": 2,
            },
        ]
        # Per-category profiles override the global CQI table when task.category
        # matches a key. Calibrated on cache2 oracle winners (same as hard ACM).
        self.category_profiles = category_profiles or {
            "code": [
                {
                    "name": "code/default",
                    "confidence_range": (0.0, 1.0),
                    "technique": "harq_cc",
                    "max_rounds": 2,
                },
            ],
            "creative": [
                {
                    "name": "creative/default",
                    "confidence_range": (0.0, 1.0),
                    "technique": "diversity_egc",
                    "num_branches": 2,
                },
            ],
            "qa": [
                {
                    "name": "qa/default",
                    "confidence_range": (0.0, 1.0),
                    "technique": "fountain",
                    "num_branches": 2,
                },
            ],
            "reasoning": [
                {
                    "name": "reasoning/default",
                    "confidence_range": (0.0, 1.0),
                    "technique": "fountain",
                    "num_branches": 2,
                },
            ],
        }

    def run(self, task: TaskItem) -> ReliabilityRun:
        # Step 1: Probe — short generation with logprobs to measure CQI
        probe_channel = next(iter(self.channels.values()))
        confidence, probe_output = self._probe_cqi(task, probe_channel)

        # Step 2: Select profile (prefer per-category table if configured)
        cat = task.category.value if hasattr(task.category, "value") else str(task.category)
        table = self.category_profiles.get(cat) if self.category_profiles else None
        routing_mode = "category" if table else "cqi"
        profile = self._select_profile(confidence, table=table)

        logger.info(
            f"SoftACM CQI probe: logprob={probe_output.mean_logprob}, "
            f"confidence={confidence:.3f} → {profile['name']} "
            f"({profile['technique']}) [{routing_mode}]"
        )

        # Step 3: Execute selected technique
        run = self._execute_profile(task, profile, probe_channel)
        run.technique = f"acm_soft_{profile['technique']}"
        run.overhead_outputs.append(probe_output)
        run.config["cqi_confidence"] = confidence
        run.config["cqi_logprob"] = probe_output.mean_logprob
        run.config["selected_profile"] = profile["name"]
        run.config["weight_source"] = "logprob"
        run.config["routing_category"] = cat
        run.config["routing_mode"] = routing_mode
        run.compute_metrics()
        return run

    def _probe_cqi(
        self, task: TaskItem, channel: AgentChannel,
    ) -> tuple[float, AgentOutput]:
        """
        CQI measurement via pilot probe.

        Generate a short response with logprobs. The mean logprob tells us
        how confident the model is on this task — our channel quality indicator.
        """
        # Short probe: ask for a brief answer to measure confidence
        probe_prompt = (
            f"Give a brief, direct answer to the following (2-3 sentences max):\n\n"
            f"{task.prompt}"
        )
        result = channel.transmit(
            probe_prompt, temperature=0.3, request_logprobs=True,
        )

        if not _has_logprobs(result):
            raise RuntimeError(
                f"SoftACMRouter requires logprobs but backend returned none "
                f"for model {result.model}. "
                f"Ensure your backend supports logprobs (Ollama, vLLM, OpenAI)."
            )
        confidence = _logprob_to_confidence(result.mean_logprob)

        return confidence, result

    def _select_profile(self, confidence: float, table: list[dict] | None = None) -> dict:
        """Select MCS profile matching the CQI confidence level."""
        tbl = table if table is not None else self.profiles
        for profile in tbl:
            low, high = profile["confidence_range"]
            if low <= confidence < high:
                return profile
        # Fallback to maximum protection
        return tbl[-1]

    def _execute_profile(
        self, task: TaskItem, profile: dict, primary_channel: AgentChannel,
    ) -> ReliabilityRun:
        """Execute the selected technique."""
        from .diversity import DiversityEnsemble
        from .fec import FECService
        from .fountain import FountainDecoder
        from .harq import HARQService

        technique = profile["technique"]

        if technique == "fec":
            svc = FECService(
                primary_channel, self.scorer,
                code_rate=profile.get("code_rate", 0.50),
            )
            return svc.run(task)

        elif technique == "harq_ir":
            svc = HARQService(
                primary_channel, self.scorer,
                mode=HARQMode.IR,
                max_rounds=profile.get("max_rounds", 3),
            )
            return svc.run(task)

        elif technique == "harq_cc":
            svc = HARQService(
                primary_channel, self.scorer,
                mode=HARQMode.CC,
                max_rounds=profile.get("max_rounds", 2),
            )
            return svc.run(task)

        elif technique == "fountain":
            channels_list = list(self.channels.values())
            n = profile.get("num_branches", 2)
            if len(channels_list) < max(n, 2):
                channels_list = (channels_list * max(n, 2))[:max(n, 2)]
            svc = FountainDecoder(
                channels=channels_list[:max(n, 2)],
                scorer=self.scorer,
            )
            return svc.run(task)

        elif technique in ("diversity_mrc", "diversity_egc", "diversity_sc"):
            channels_list = list(self.channels.values())
            n = profile.get("num_branches", 2)
            if len(channels_list) < n:
                channels_list = channels_list * n
            combining_map = {
                "diversity_mrc": CombiningStrategy.MRC,
                "diversity_egc": CombiningStrategy.EGC,
                "diversity_sc":  CombiningStrategy.SC,
            }
            svc = DiversityEnsemble(
                channels_list[:n], self.scorer,
                combining=combining_map[technique],
            )
            return svc.run(task, synthesizer=primary_channel)

        else:
            # Unknown technique — fall back to single uncoded transmission
            run = ReliabilityRun(
                task_id=task.id,
                task_category=task.category.value,
                technique="acm_soft_uncoded",
                config={"model": primary_channel.model},
            )
            out = primary_channel.transmit(task.request)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            run.individual_outputs = [out]
            run.combined_output = out.text
            run.final_quality = out.quality_score
            run.compute_metrics()
            return run
