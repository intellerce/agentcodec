"""
Direct implementations of comparable prior methods, run as standalone baselines
alongside the AgentCodec techniques.

These are NOT AgentCodec techniques -- they are faithful reproductions of the
canonical recipe in each referenced paper, used to provide head-to-head
evidence that each AgentCodec technique meets or exceeds its prior-method
analog at matched inference budget.

Included:
- SelfConsistencyBaseline           (Wang et al., 2023; canonical exact-match
                                     mode when ``answer_extractor`` is supplied,
                                     Universal-SC LLM voter otherwise)
- SelfRefineBaseline                (Madaan et al., 2023; includes the paper's
                                     STOP function via a generic STOP|CONTINUE
                                     prefix)
- ChainOfVerificationBaseline       (Dhuliawala et al., 2023; Factored variant)
- BestOfNBaseline                   (Cobbe et al., 2021; Lightman et al., 2024
                                     -- single-model)
- WeightedBoNBaseline               (Snell et al., 2024; judge-weighted
                                     single-model BoN, extrinsic-CSI variant,
                                     canonical exact-match path when
                                     ``answer_extractor`` is supplied)
- CISCBaseline                      (Taubenfeld et al., 2025 -- canonical CISC,
                                     intrinsic logprob CSI, single-model)
- MixtureOfAgentsBaseline           (Wang et al., 2025; verbatim aggregator
                                     instruction, T=0.7 throughout)
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, ReliabilityRun, TaskItem

logger = logging.getLogger(__name__)


class SelfConsistencyBaseline:
    """
    Self-Consistency (Wang et al., 2023, arXiv:2203.11171).

    Sample N reasoning paths at T=0.7, marginalize over reasoning paths by
    majority voting on the final answer (paper's Section 2.2).

    Two aggregation paths are supported:

    * ``answer_extractor`` provided → CANONICAL Wang+2023. Each output's
      final answer is extracted to a normalized string; equivalence
      classes are formed by exact string match; the mode is returned via
      ``Counter.most_common``. No LLM call on the aggregation path.
      Empty/None extractions are filtered out. This is the only path
      that matches the paper.
    * ``answer_extractor`` is None → free-form fallback. A neutral voter
      LLM is asked to identify the most-frequent answer by content.
      This is closer to Universal Self-Consistency (Chen et al., 2023)
      than to canonical SC -- it is an extension to free-form tasks
      where no deterministic extractor exists. In this regime the
      technique label is suffixed with ``_llm_voter`` so runs can be
      told apart in logs.

    No quality weighting (that would be Diversity MRC / CISC).
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        voter: AgentChannel | None = None,
        answer_extractor: Callable[[AgentOutput, TaskItem], str | None] | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        self.num_samples = num_samples
        self.voter = voter  # used only in the free-form (no-extractor) path
        self.answer_extractor = answer_extractor

    def run(self, task: TaskItem) -> ReliabilityRun:
        from collections import Counter

        aggregation_mode = (
            "exact_match" if self.answer_extractor is not None else "llm_voter"
        )
        technique_label = (
            "self_consistency"
            if self.answer_extractor is not None
            else "self_consistency_llm_voter"
        )
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique=technique_label,
            config={
                "num_samples": self.num_samples,
                "num_channels": len(self.channels),
                "aggregation": aggregation_mode,
            },
        )

        outputs: list[AgentOutput] = []
        for i in range(self.num_samples):
            channel = self.channels[i % len(self.channels)]
            temp = 0.7  # canonical Wang+2023 setting
            out = channel.transmit(task.request, temperature=temp)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        run.individual_outputs = outputs
        run.rounds = self.num_samples

        if self.answer_extractor is not None:
            # Canonical Wang+2023: deterministic extraction + Counter mode.
            # No LLM call on the aggregation path.
            extracted: list[str | None] = [
                self.answer_extractor(o, task) for o in outputs
            ]
            counter = Counter(
                a for a in extracted if a is not None and a != ""
            )
            if counter:
                winning_answer, winning_count = counter.most_common(1)[0]
                # Any member of the winning equivalence class has the same
                # extracted answer; pick the best-written representative
                # (highest judge score) for the returned text.
                winning_members = [
                    i for i, a in enumerate(extracted) if a == winning_answer
                ]
                best_idx = max(
                    winning_members,
                    key=lambda i: outputs[i].quality_score or 0.0,
                )
                combined_text = outputs[best_idx].text
                run.config["winning_answer"] = winning_answer
                run.config["winning_count"] = winning_count
            else:
                # All extractions empty: degenerate to first sample.
                combined_text = outputs[0].text
                run.config["winning_answer"] = ""
                run.config["winning_count"] = 0
            run.config["extracted_answers"] = [
                a if a is not None else "" for a in extracted
            ]
        else:
            # Free-form fallback (NOT canonical Wang+2023; closer to
            # Universal Self-Consistency, Chen et al. 2023): a neutral
            # voter LLM picks the most representative answer.
            voter = self.voter or self.channels[0]
            joined = "\n\n".join(
                f"### Sample {i+1}\n{o.text}" for i, o in enumerate(outputs)
            )
            vote_prompt = (
                f"Below are {self.num_samples} independent answers to the same task. "
                f"Identify the MOST FREQUENT final answer (majority vote by content, "
                f"not by wording). Return ONLY the majority answer's text, with no "
                f"commentary.\n\n"
                f"## Task\n{task.prompt}\n\n"
                f"## Candidate Answers\n{joined}\n\n"
                f"## Majority answer:"
            )
            vote = voter.transmit(vote_prompt, temperature=0.0)
            combined_text = vote.text.strip() or outputs[0].text
            run.overhead_outputs.append(vote)

        run.combined_output = combined_text
        run.final_quality = self.scorer.score(
            task.prompt, combined_text, reference=task.reference, task=task,
        )
        run.compute_metrics()
        return run


class SelfRefineBaseline:
    """
    Self-Refine (Madaan et al., 2023, arXiv:2303.17651).

    Implements the paper's three-step loop (Section 2.1):
        y_0     = M(p_gen    ∥ x)
        fb_t    = M(p_fb     ∥ x ∥ y_t)
        y_{t+1} = M(p_refine ∥ x ∥ y_t ∥ fb_t)
    with the paper's STOP function (Section 2.2): the critic emits a stop
    signal when no further improvement is needed, otherwise the loop runs
    for at most ``max_rounds`` iterations. We use a generic
    ``STOP|CONTINUE`` prefix in place of the paper's task-specific stop
    tokens (e.g., "[Optimal]" for code optimization, "everything is fine"
    for sentiment reversal); the mechanism is the same, applied at a
    task-agnostic level. Set ``early_stop=False`` to disable the stop
    signal and force a fixed-K critique loop (not paper-faithful; useful
    only for ablations).

    Acknowledged adaptations vs. the paper:
      - Zero-shot prompts (paper uses few-shot, task-specific
        p_gen/p_fb/p_refine).
      - The final iteration is the output, as in the paper (no
        best-of-sequence guard).

    Unlike AgentCodec HARQ-IR/turbo, there is:
      - no structured JSON critique (free-form feedback),
      - no extrinsic-information constraint (critic sees everything),
      - no alpha damping or severity floor,
      - no interleaver (critic always uses the same lens).
    """

    def __init__(
        self,
        channel: AgentChannel,
        scorer: QualityScorer,
        max_rounds: int = 3,
        early_stop: bool = True,
    ):
        self.channel = channel
        self.scorer = scorer
        self.max_rounds = max_rounds
        # Paper-faithful default: True. The paper's STOP function exits
        # the loop early when the critic signals no-further-improvement;
        # disabling this reduces the implementation to a fixed-K critique
        # loop, which systematically over-revises and is the known
        # failure mode the stop signal exists to prevent.
        self.early_stop = early_stop

    @staticmethod
    def _is_stop_signal(critique_text: str) -> bool:
        """Detect a STOP token at the start of the critic's response.

        Matches case-insensitively, tolerates leading whitespace and
        trailing punctuation. Any first token other than STOP (including
        CONTINUE, or an empty response) means "keep refining".
        """
        if not critique_text:
            return False
        stripped = critique_text.strip()
        if not stripped:
            return False
        # First whitespace- or punctuation-delimited token of the response.
        first_token = re.split(r"[\s.,:;!?\-]+", stripped, maxsplit=1)[0]
        return first_token.upper() == "STOP"

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="self_refine",
            config={
                "max_rounds": self.max_rounds,
                "model": self.channel.model,
                "early_stop": self.early_stop,
            },
        )

        current = self.channel.transmit(task.request, temperature=0.7)
        current.quality_score = self.scorer.score(
            task.prompt, current.text, reference=task.reference, task=task,
        )
        history = [current]

        stop_reason = "max_rounds"
        for k in range(1, self.max_rounds + 1):
            if self.early_stop:
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
            critique = self.channel.transmit(critique_prompt, temperature=0.5)
            history.append(critique)
            run.rounds = k

            # Paper's STOP function (Section 2.2): exit the loop when the
            # critic signals no further improvement. The generic
            # STOP|CONTINUE prefix stands in for the paper's task-specific
            # stop tokens; the mechanism is identical.
            if self.early_stop and self._is_stop_signal(critique.text):
                stop_reason = "model_signal"
                break

            revise_prompt = (
                f"## Original Task\n{task.prompt}\n\n"
                f"## Previous Answer\n{current.text}\n\n"
                f"## Critique\n{critique.text}\n\n"
                f"Produce an improved answer that addresses the critique. "
                f"Return only the improved answer."
            )
            revised = self.channel.transmit(revise_prompt, temperature=0.7)
            revised.quality_score = self.scorer.score(
                task.prompt, revised.text, reference=task.reference, task=task,
            )
            history.append(revised)
            current = revised

        run.config["stop_reason"] = stop_reason
        run.individual_outputs = history
        run.combined_output = current.text
        run.final_quality = current.quality_score or 0.0
        run.compute_metrics()
        return run


class ChainOfVerificationBaseline:
    """
    Chain-of-Verification / CoVe -- Factored variant
    (Dhuliawala et al., 2023, arXiv:2309.11495, Section 3.2).

    The paper defines four variants: Joint, 2-Step, Factored, and
    Factor+Revise. This implements the FACTORED variant: each
    verification question is answered IN ISOLATION (no access to the
    baseline answer or to the other verifications) to eliminate
    confirmation bias from the baseline draft.

    4-step pipeline:
      1. Generate baseline answer.
      2. Plan verification questions that probe the baseline's factual content.
      3. Execute each verification question INDEPENDENTLY (no access to the
         baseline answer, to avoid confirmation bias).
      4. Generate final answer consistent with the verification results.

    Acknowledged adaptation: paper uses few-shot exemplars for the
    planning and final-answer prompts; here both are zero-shot.

    Unlike AgentCodec FEC, the parity section is fixed (verification questions)
    and cannot be rate-adapted.
    """

    def __init__(
        self,
        channel: AgentChannel,
        scorer: QualityScorer,
        num_verification_questions: int = 3,
    ):
        self.channel = channel
        self.scorer = scorer
        self.num_verifications = num_verification_questions

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="chain_of_verification",
            config={
                "num_verification_questions": self.num_verifications,
                "model": self.channel.model,
            },
        )

        # Step 1: baseline answer
        baseline = self.channel.transmit(task.request, temperature=0.7)
        baseline.quality_score = self.scorer.score(
            task.prompt, baseline.text, reference=task.reference, task=task,
        )

        # Step 2: plan verification questions
        plan_prompt = (
            f"## Task\n{task.prompt}\n\n"
            f"## Draft Answer\n{baseline.text}\n\n"
            f"Generate exactly {self.num_verifications} short verification "
            f"questions that, if answered independently, would test the "
            f"correctness of the draft answer's factual claims. Number them "
            f"1), 2), 3). Produce only the numbered questions, one per line."
        )
        plan = self.channel.transmit(plan_prompt, temperature=0.3)

        # Parse questions
        questions = []
        for line in plan.text.splitlines():
            line = line.strip()
            m = re.match(r"^\s*\d+[\.\)\:]\s*(.+)$", line)
            if m:
                questions.append(m.group(1).strip())
        questions = questions[: self.num_verifications]
        if not questions:
            # Fallback: treat every non-empty line as a question.
            questions = [
                ln.strip() for ln in plan.text.splitlines() if ln.strip()
            ][: self.num_verifications]

        # Step 3: execute each verification independently (no baseline context)
        verifications: list[AgentOutput] = []
        for q in questions:
            v = self.channel.transmit(
                f"Answer this question as accurately as possible:\n\n{q}",
                temperature=0.3,
            )
            verifications.append(v)

        # Step 4: generate revised final answer consistent with verifications
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
        final = self.channel.transmit(final_prompt, temperature=0.5)
        final.quality_score = self.scorer.score(
            task.prompt, final.text, reference=task.reference, task=task,
        )

        run.individual_outputs = [baseline, final]
        run.overhead_outputs = [plan, *verifications]
        run.rounds = 1 + self.num_verifications
        run.combined_output = final.text
        run.final_quality = final.quality_score or 0.0
        run.compute_metrics()
        return run


class BestOfNBaseline:
    """
    Canonical Best-of-N (Cobbe et al., 2021; Lightman et al., 2024).

    Sample N candidates from a single policy model via temperature sampling,
    score each with the shared judge, return the argmax. Canonical BoN in the
    literature uses a trained PRM/ORM verifier; here we reuse the shared
    judge model so the inference budget matches the other baselines.

    Uses only the first configured channel so all N candidates come from one
    policy -- this single-model constraint is what distinguishes BoN from our
    multi-model ``diversity_sc_N`` operator in ``diversity.py``.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
    ):
        if not channels:
            raise ValueError("BestOfNBaseline needs at least one channel")
        self.channel = channels[0]
        self.scorer = scorer
        self.num_samples = num_samples

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="best_of_n",
            config={
                "num_samples": self.num_samples,
                "model": self.channel.model,
            },
        )

        outputs: list[AgentOutput] = []
        for _ in range(self.num_samples):
            out = self.channel.transmit(task.request, temperature=0.7)
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


class WeightedBoNBaseline:
    """
    Weighted Best-of-N (Snell et al., 2024, arXiv:2408.03314, Section 3.2).

    Sample N candidates from a single policy model via temperature sampling,
    group them into semantic equivalence classes, sum judge scores within
    each class, pick the class with the highest total, and return that
    class's top-scoring sample. This is MRC on the discrete answer space.

    Two aggregation paths are supported (mirrors CISCBaseline's design):

    * ``answer_extractor`` provided → CANONICAL Snell+2024. Equivalence
      classes are formed by exact match on extracted final answers (the
      paper does this on numeric answers for math); no voter LLM call.
      Empty/None extractions are filtered out.
    * ``answer_extractor`` is None → free-form fallback. One voter LLM
      call returns a JSON cluster assignment. NOT in the paper; an
      extension to free-form tasks where no extractable answer exists.

    Uses only the first configured channel so all N candidates come from
    one policy, matching the canonical single-model setup. The multi-model
    wider-pool analog lives as ``DiversityMRCDiscreteN`` in
    ``diversity.py``.

    Acknowledged adaptation: paper uses a trained PRM/ORM verifier; here
    we reuse the shared judge model so the inference budget matches the
    other baselines (the same swap is acknowledged in BestOfNBaseline).
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        voter: AgentChannel | None = None,
        answer_extractor: Callable[[AgentOutput, TaskItem], str | None] | None = None,
    ):
        if not channels:
            raise ValueError("WeightedBoNBaseline needs at least one channel")
        self.channel = channels[0]
        self.scorer = scorer
        self.num_samples = num_samples
        self.voter = voter  # used only in the free-form fallback path
        self.answer_extractor = answer_extractor

    def run(self, task: TaskItem) -> ReliabilityRun:
        import json
        aggregation_mode = (
            "exact_match" if self.answer_extractor is not None else "voter_cluster"
        )
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="weighted_bon",
            config={
                "num_samples": self.num_samples,
                "model": self.channel.model,
                "aggregation": aggregation_mode,
            },
        )

        outputs: list[AgentOutput] = []
        for _ in range(self.num_samples):
            out = self.channel.transmit(task.request, temperature=0.7)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        if self.answer_extractor is not None:
            # Canonical Snell+2024 path: exact-match clustering on
            # extracted answers; sum judge scores per class; pick the
            # top-scoring member of the winning class. No LLM call on
            # the aggregation path.
            extracted: list[str | None] = [
                self.answer_extractor(o, task) for o in outputs
            ]
            totals_str: dict[str, float] = {}
            members_by_ans: dict[str, list[int]] = {}
            for idx, ans in enumerate(extracted):
                if ans is None or ans == "":
                    continue
                totals_str[ans] = totals_str.get(ans, 0.0) + (
                    outputs[idx].quality_score or 0.0
                )
                members_by_ans.setdefault(ans, []).append(idx)
            if totals_str:
                winning_answer = max(totals_str, key=lambda k: totals_str[k])
                winning_members = members_by_ans[winning_answer]
                best_idx = max(
                    winning_members,
                    key=lambda i: outputs[i].quality_score or 0.0,
                )
                run.config["winning_answer"] = winning_answer
                run.config["winning_cluster_size"] = len(winning_members)
            else:
                # All extractions empty: degenerate to pure BoN argmax.
                best_idx = max(
                    range(len(outputs)),
                    key=lambda i: outputs[i].quality_score or 0.0,
                )
                run.config["winning_answer"] = ""
                run.config["winning_cluster_size"] = 0
            run.config["extracted_answers"] = [
                a if a is not None else "" for a in extracted
            ]
        else:
            # Free-form fallback (NOT in the paper): voter LLM clusters.
            voter = self.voter or self.channel
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

            # Parse the cluster assignment; fall back to per-sample
            # singletons (pure BoN) on parse failure.
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

            totals: dict[int, float] = {}
            members: dict[int, list[int]] = {}
            for idx, lbl in enumerate(labels):
                totals[lbl] = totals.get(lbl, 0.0) + (
                    outputs[idx].quality_score or 0.0
                )
                members.setdefault(lbl, []).append(idx)
            winning_cluster = max(totals, key=lambda k: totals[k])
            winning_members = members[winning_cluster]
            best_idx = max(
                winning_members, key=lambda i: outputs[i].quality_score or 0.0
            )
            run.config["cluster_labels"] = labels
            run.config["winning_cluster_size"] = len(winning_members)

        best = outputs[best_idx]
        run.individual_outputs = outputs
        run.rounds = self.num_samples
        run.combined_output = best.text
        run.final_quality = best.quality_score or 0.0
        run.compute_metrics()
        return run


class CISCBaseline:
    """
    Canonical Confidence-Informed Self-Consistency / CISC
    (Taubenfeld et al., 2025, arXiv:2502.06233; reference implementation:
    google-research/google-research/cisc).

    Implements Definition 3.1 of the paper:
      1. Confidence Extraction — per-response c_i, sourced from one of:
           - "verbal_100" (default): single-step variant of the paper's
             Verbal 0-100 prompt (Table 5) — the model writes its own
             confidence as a `Confidence: <int>` line at the end of the
             response. Works on any backend (no logprob exposure needed,
             including Anthropic). Single LLM call per sample.
           - "response_probability": c_i = exp(mean_logprob_i), the
             geometric mean of token probabilities (paper's intrinsic
             extraction). Requires backends that expose logprobs.
      2. Confidence Normalization — softmax with tunable temperature T:
            c̃_i = exp(c_i / T) / Σ_j exp(c_j / T)
         T → ∞ collapses CISC to vanilla self-consistency; T → 0 collapses
         it to argmax-by-confidence. The paper tunes T per (model, method)
         on a 10% held-out set via 80-point log-spaced grid search
         1e-4..1e4 (Appendix D); we expose it as `softmax_temperature`.
         Default depends on csi_source — see __init__.
      3. Aggregation — confidence-weighted majority vote:
            â = argmax_a Σ_i 1[a_i = a] · c̃_i

    Two aggregation paths are supported:

    * `answer_extractor` provided → canonical CISC. Each output's final
      answer is extracted to a normalized string; equivalence classes are
      formed by exact string match (mirrors `Counter`-on-extracted-answers
      in the official aggregators.py). Empty/None extractions are
      filtered out per the reference implementation. This is the only
      path that matches the paper.
    * `answer_extractor` is None → free-form fallback. Equivalence
      classes are inferred via a single voter-LLM clustering call. This
      is *not* in the paper — it is an extension to free-form tasks
      where no extractable final answer exists.

    Vs. WeightedBoNBaseline: same discrete-MRC operator on a single-policy
    pool; CISC uses self-assessed confidence (no extra judge call) and
    softmax-normalizes before voting, whereas WeightedBoN uses an extrinsic
    judge score and does not softmax-normalize.
    """

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        voter: AgentChannel | None = None,
        softmax_temperature: float | None = None,
        answer_extractor: Callable[[AgentOutput, TaskItem], str | None] | None = None,
        csi_source: str = "verbal_100",
    ):
        # Default T per csi_source uses the median value in Figure 8 of the
        # paper for that extraction method, treating raw c_i in the paper's
        # native scale (so per-model overrides from Figure 8 can be used
        # directly without rescaling).
        #   verbal_100 → median 8 (over 9 models; range 3..90)
        #   response_probability → median 0.1 (range 0.09..2)
        # The paper tunes T per (model, method) on a 10% held-out set via
        # 80-point log-spaced grid search 1e-4..1e4 (App. D). For faithful
        # per-model performance the caller should override this default.
        # T → ∞ collapses to vanilla SC; T → 0 to argmax-by-confidence.
        DEFAULT_T_BY_SOURCE = {
            "verbal_100": 8.0,
            "response_probability": 0.1,
        }
        if csi_source not in DEFAULT_T_BY_SOURCE:
            raise ValueError(
                f"csi_source must be one of {list(DEFAULT_T_BY_SOURCE)}; "
                f"got {csi_source!r}"
            )
        if not channels:
            raise ValueError("CISCBaseline needs at least one channel")
        if softmax_temperature is None:
            softmax_temperature = DEFAULT_T_BY_SOURCE[csi_source]
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0")
        self.channel = channels[0]
        self.scorer = scorer
        self.num_samples = num_samples
        self.voter = voter
        self.softmax_temperature = softmax_temperature
        self.answer_extractor = answer_extractor
        self.csi_source = csi_source

    def run(self, task: TaskItem) -> ReliabilityRun:
        import json
        import math

        from .soft import (
            append_verbal_confidence_prompt,
            parse_verbal_confidence,
        )
        aggregation_mode = (
            "exact_match" if self.answer_extractor is not None else "voter_cluster"
        )
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="cisc",
            config={
                "num_samples": self.num_samples,
                "model": self.channel.model,
                "csi_source": self.csi_source,
                "softmax_temperature": self.softmax_temperature,
                "aggregation": aggregation_mode,
            },
        )

        # Sampling — branches on csi_source:
        #   verbal_100: append the paper's Verbal-100 instruction (single-step
        #     variant of App. B); plain text generation, no logprobs needed.
        #   response_probability: request logprobs; c_i = exp(mean_logprob).
        prompt = task.prompt
        request_lp = self.csi_source == "response_probability"
        if self.csi_source == "verbal_100":
            prompt = append_verbal_confidence_prompt(task.prompt, scale=100)

        outputs: list[AgentOutput] = []
        for _ in range(self.num_samples):
            out = self.channel.transmit(
                prompt, temperature=0.7, request_logprobs=request_lp,
            )
            # Final-quality bookkeeping still needs a judge score per
            # candidate, but the *combining weight* is the self-assessed
            # confidence (verbal score or logprob).
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            outputs.append(out)

        # Per-sample raw confidence c_i (paper's Definition 3.1, step 1).
        raw_confidences: list[float] = []
        if self.csi_source == "response_probability":
            # Hard fail on backends without logprob support — silently
            # falling back to judge weights would make this indistinguishable
            # from WeightedBoNBaseline and defeat the point of having both.
            missing = [o for o in outputs if o.mean_logprob is None]
            if missing:
                models = {o.model for o in missing}
                raise RuntimeError(
                    f"CISCBaseline(csi_source='response_probability') requires "
                    f"token logprobs, but backend returned none for "
                    f"{len(missing)}/{len(outputs)} samples (models: {models}). "
                    f"Switch csi_source to 'verbal_100' for backends without "
                    f"logprobs (e.g. Anthropic), or use a backend that exposes "
                    f"them (Ollama, vLLM, OpenAI)."
                )
            raw_confidences = [math.exp(o.mean_logprob) for o in outputs]
        else:
            # verbal_100: parse the trailing `Confidence: <int>` line. Per
            # the paper's filter-empty-confidences convention, samples with
            # no parseable score get c_i = 0 (still counted for vote eligibility
            # via aggregation, but contribute zero softmax mass at any T > 0).
            parsed: list[float | None] = [
                parse_verbal_confidence(o.text, scale=100) for o in outputs
            ]
            unparsed = sum(1 for p in parsed if p is None)
            if unparsed == len(outputs):
                raise RuntimeError(
                    f"CISCBaseline(csi_source='verbal_100') could not parse "
                    f"any of the {len(outputs)} samples for "
                    f"`Confidence: <int>`. The model is not following the "
                    f"verbal-confidence instruction. Consider switching "
                    f"csi_source to 'response_probability' if logprobs are "
                    f"available, or check that the prompt is reaching the "
                    f"model intact."
                )
            if unparsed:
                logger.warning(
                    f"CISCBaseline: {unparsed}/{len(outputs)} samples did not "
                    f"emit a parseable `Confidence: <int>` line; treating as "
                    f"c_i=0 (per CISC's filter-empty-confidences convention)."
                )
            raw_confidences = [p if p is not None else 0.0 for p in parsed]
            run.config["verbal_confidences"] = raw_confidences
            run.config["verbal_unparsed"] = unparsed

        # Confidence Normalization (paper's Definition 3.1, step 2):
        #   c̃_i = exp(c_i / T) / Σ_j exp(c_j / T)
        # numerically stabilized with the standard max-shift trick.
        T = self.softmax_temperature
        scaled = [c / T for c in raw_confidences]
        shift = max(scaled)
        exps = [math.exp(s - shift) for s in scaled]
        Z = sum(exps)
        norm_confidences = [e / Z for e in exps]

        if self.answer_extractor is not None:
            # Canonical path: Counter-on-extracted-answers, mirroring
            # google-research/cisc/post_processing/aggregators.py.
            extracted: list[str | None] = [
                self.answer_extractor(o, task) for o in outputs
            ]
            totals_str: dict[str, float] = {}
            members_by_ans: dict[str, list[int]] = {}
            for idx, ans in enumerate(extracted):
                if ans is None or ans == "":
                    continue
                totals_str[ans] = totals_str.get(ans, 0.0) + norm_confidences[idx]
                members_by_ans.setdefault(ans, []).append(idx)
            if totals_str:
                winning_answer = max(totals_str, key=lambda k: totals_str[k])
                winning_members = members_by_ans[winning_answer]
                best_idx = max(winning_members, key=lambda i: raw_confidences[i])
                run.config["winning_answer"] = winning_answer
                run.config["winning_cluster_size"] = len(winning_members)
            else:
                # All extractions empty: degenerate to argmax-confidence,
                # matching the paper's empty-pool fallback `('', 0)` modulo
                # our need to still return a representative output.
                best_idx = max(
                    range(len(outputs)), key=lambda i: raw_confidences[i]
                )
                run.config["winning_answer"] = ""
                run.config["winning_cluster_size"] = 0
            run.config["extracted_answers"] = [
                a if a is not None else "" for a in extracted
            ]
        else:
            # Free-form fallback (NOT from the paper): infer equivalence
            # classes via one voter-LLM clustering call.
            voter = self.voter or self.channel
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
                # Voter failed: each sample its own cluster -> degenerates to
                # selection by max confidence (BoN with intrinsic CSI).
                labels = list(range(self.num_samples))

            totals: dict[int, float] = {}
            members: dict[int, list[int]] = {}
            for idx, lbl in enumerate(labels):
                totals[lbl] = totals.get(lbl, 0.0) + norm_confidences[idx]
                members.setdefault(lbl, []).append(idx)
            winning_cluster = max(totals, key=lambda k: totals[k])
            winning_members = members[winning_cluster]
            best_idx = max(winning_members, key=lambda i: raw_confidences[i])
            run.config["cluster_labels"] = labels
            run.config["winning_cluster_size"] = len(winning_members)

        best = outputs[best_idx]
        run.individual_outputs = outputs
        run.rounds = self.num_samples
        run.config["raw_confidences"] = [round(c, 4) for c in raw_confidences]
        run.config["norm_confidences"] = [round(c, 4) for c in norm_confidences]
        run.combined_output = best.text
        run.final_quality = best.quality_score or 0.0
        run.compute_metrics()
        return run


class MixtureOfAgentsBaseline:
    """
    Mixture-of-Agents / MoA (Wang et al., 2025, arXiv:2406.04692;
    official repo: togethercomputer/MoA).

    Multi-layer MoA: each of L layers has K = num_channels agents. Layer 1
    proposes from the task prompt; layers 2..L are aggregator-proposers that
    each see all outputs from the previous layer and produce a refined draft.
    A single final aggregator synthesizes the layer-L outputs into the answer.

    Faithfulness notes:
      - The aggregator instruction is the verbatim string from the paper /
        official repo (see ``_AGGREGATOR_INSTRUCTION`` below).
      - Sampling temperature is 0.7 throughout, matching the official
        repo's default sampling temperature (0.7) for both proposers and
        the final aggregator. Lowering the final-aggregator temperature
        compresses the ensemble diversity the method depends on.
      - References are formatted as a numbered list (``1. ..\\n2. ..``)
        following the official ``inject_references_to_messages`` helper.
      - The official repo passes the instruction + references as a SYSTEM
        message and the user query as the USER message. Our
        ``AgentChannel.system_prompt`` is fixed at channel construction
        time and cannot be overridden per-call, so we collapse the
        aggregator turn into a single user message that preserves the
        instruction → references → query ordering. This is the only
        structural deviation from the official implementation.
      - The paper's default is L=3 proposer layers. Here L defaults to
        ``ceil(num_samples / num_channels)`` to keep total proposer-call
        budget L*K close to ``num_samples`` for comparability with the
        other matched-budget baselines; pass ``num_layers=3`` (or any
        explicit value) to use the paper default.
    """

    # Verbatim from Wang et al. 2025 Section 2 (and the official
    # togethercomputer/MoA `inject_references_to_messages` helper).
    # Do not paraphrase: the paper's gains depend in part on this
    # specific adversarial framing of the candidate responses.
    _AGGREGATOR_INSTRUCTION = (
        "You have been provided with a set of responses from various "
        "open-source models to the latest user query. Your task is to "
        "synthesize these responses into a single, high-quality response. "
        "It is crucial to critically evaluate the information provided in "
        "these responses, recognizing that some of it may be biased or "
        "incorrect. Your response should not simply replicate the given "
        "answers but should offer a refined, accurate, and comprehensive "
        "reply to the instruction. Ensure your response is well-structured, "
        "coherent, and adheres to the highest standards of accuracy and "
        "reliability."
    )

    def __init__(
        self,
        channels: list[AgentChannel],
        scorer: QualityScorer,
        num_samples: int = 5,
        aggregator: AgentChannel | None = None,
        num_layers: int | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        self.num_samples = num_samples
        self.aggregator = aggregator
        if num_layers is None:
            num_layers = max(1, math.ceil(num_samples / max(1, len(channels))))
        self.num_layers = num_layers

    def _aggregator_prompt(self, task_prompt: str, prior: list[AgentOutput]) -> str:
        # Format mirrors the official MoA repo: instruction, then
        # "Responses from models:" followed by a numbered list of
        # references, then the user query. The official repo puts the
        # instruction+references in the system message and the query in
        # the user message; collapsing to a single user message
        # preserves their relative ordering, which is what matters for
        # the aggregator's framing.
        joined = "\n".join(f"{i+1}. {o.text}" for i, o in enumerate(prior))
        return (
            f"{self._AGGREGATOR_INSTRUCTION}\n\n"
            f"Responses from models:\n{joined}\n\n"
            f"{task_prompt}"
        )

    def run(self, task: TaskItem) -> ReliabilityRun:
        # Temperature 0.7 throughout matches the official
        # togethercomputer/MoA repo's default sampling temperature (0.7)
        # for proposers and aggregator alike.
        K = len(self.channels)
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="mixture_of_agents",
            config={
                "num_samples": self.num_samples,
                "num_channels": K,
                "num_layers": self.num_layers,
                "aggregator": (self.aggregator or self.channels[0]).model,
            },
        )

        # Layer 1: K proposers sample directly from the task prompt.
        prev_layer: list[AgentOutput] = []
        for i in range(K):
            out = self.channels[i].transmit(task.request, temperature=0.7)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task,
            )
            prev_layer.append(out)
        all_proposals: list[AgentOutput] = list(prev_layer)

        # Layers 2..L: each agent refines based on the previous layer's outputs.
        for _ in range(self.num_layers - 1):
            prompt = self._aggregator_prompt(task.prompt, prev_layer)
            next_layer: list[AgentOutput] = []
            for i in range(K):
                out = self.channels[i].transmit(prompt, temperature=0.7)
                out.quality_score = self.scorer.score(
                    task.prompt, out.text, reference=task.reference, task=task,
                )
                next_layer.append(out)
            all_proposals.extend(next_layer)
            prev_layer = next_layer

        # Final aggregator: single synthesis over the last layer.
        aggregator = self.aggregator or self.channels[0]
        synthesis = aggregator.transmit(
            self._aggregator_prompt(task.prompt, prev_layer),
            temperature=0.7,
        )
        run.overhead_outputs.append(synthesis)

        combined_text = synthesis.text.strip() or prev_layer[0].text
        run.individual_outputs = all_proposals
        run.rounds = len(all_proposals)
        run.combined_output = combined_text
        run.final_quality = self.scorer.score(
            task.prompt, combined_text, reference=task.reference, task=task,
        )
        run.compute_metrics()
        return run
