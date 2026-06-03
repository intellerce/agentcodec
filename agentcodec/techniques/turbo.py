"""
Technique 3: Turbo Decoder

Communication analog: Turbo codes use two SISO decoders exchanging extrinsic
information iteratively. Each decoder refines its estimate using soft information
from the other.

Agent analog: A Generator and a Critic iteratively refine output.
- Generator produces/refines the answer (decoder 1)
- Critic evaluates and provides extrinsic information (decoder 2)
- Generator incorporates feedback in next iteration
- Exchange of "extrinsic information" = feedback that is NEW, not just repetition

Key distinction from HARQ-IR: Turbo decoding uses an INTERLEAVER between the
two component decoders. Each iteration, the critic evaluates through a different
lens (correctness → completeness → reasoning → clarity), decorrelating its
observations so extrinsic information is genuinely independent per iteration.
HARQ-IR uses the same evaluation approach each round.

Design rationale (v4 — extrinsic scaling + fixed iterations):
- Same-model critic (matching real turbo codes: both decoders use same trellis)
- Interleaver: rotating evaluation lens per iteration (decorrelation)
- Extrinsic information scaling (α < 1): limits corrections per round to top-K
  by severity. In real turbo decoders, extrinsic LLRs are scaled by α ∈ [0.5, 1]
  to prevent oscillation (a well-known fix, see Vogt & Fingscheidt 2001).
  Without scaling, aggressive corrections cause the generator to over-correct,
  introducing new errors while fixing old ones → oscillation instead of
  convergence. Our analog: filter to the most severe corrections each round.
- Fixed-iteration mode: run all rounds (no aggressive stall detection).
  Real turbo decoders run a fixed iteration count. Regression protection
  (best-of-sequence) ensures extra iterations can't hurt quality.
- Per-iteration comparative scoring breaks the quantization barrier
  (analog: SISO decoding produces soft LLRs, not hard decisions)
- Structured critique forces independent extrinsic information per iteration
- Correction-based generator pass prevents destructive rewrite
  (analog: decoder updates beliefs incrementally, not from scratch)
"""

from __future__ import annotations

import json
import logging
import re

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, ReliabilityRun, TaskItem

logger = logging.getLogger(__name__)


def _parse_structured_critique(critique_text: str) -> dict:
    """
    Parse a structured critique response into a normalized format.
    Identical to the HARQ version — shared logic for JSON critique parsing
    with graceful fallback for weak critics.
    """
    raw = critique_text.strip()

    for text_to_try in [raw, raw.strip("`").strip()]:
        try:
            data = json.loads(text_to_try)
            if isinstance(data, list):
                return {"issues": data, "raw_text": raw}
            if isinstance(data, dict) and "issues" in data:
                return {"issues": data["issues"], "raw_text": raw}
        except (json.JSONDecodeError, ValueError):
            pass

    code_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", raw, re.DOTALL)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            if isinstance(data, list):
                return {"issues": data, "raw_text": raw}
            if isinstance(data, dict) and "issues" in data:
                return {"issues": data["issues"], "raw_text": raw}
        except (json.JSONDecodeError, ValueError):
            pass

    arr_match = re.search(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", raw, re.DOTALL)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return {"issues": data, "raw_text": raw}
        except (json.JSONDecodeError, ValueError):
            pass

    upper = raw.upper()
    pass_signals = [
        "CONVERGED" in upper and len(raw) < 300,
        "NO NEW ISSUES" in upper and len(raw) < 300,
        "NO SIGNIFICANT ISSUES" in upper and len(raw) < 300,
        raw.strip() == "[]",
    ]
    if any(pass_signals):
        return {"issues": [], "raw_text": raw}

    # --- Try parsing the simpler ISSUE N: WRONG/SHOULD BE format ---
    # This catches output from weak critics that can't produce JSON.
    issue_pattern = re.compile(
        r'ISSUE\s*\d+\s*:\s*(?:WRONG|FIX|ERROR)\s*:\s*"([^"]+)"\s*'
        r'(?:SHOULD\s*BE|CHANGE\s*TO|CORRECT(?:ION)?)\s*:\s*(.+?)(?=\nISSUE|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    missing_pattern = re.compile(
        r'ISSUE\s*\d+\s*:\s*MISSING\s+near\s+"([^"]+)"\s*:\s*(.+?)(?=\nISSUE|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    parsed_issues = []
    for m in issue_pattern.finditer(raw):
        parsed_issues.append({
            "quote": m.group(1).strip(),
            "type": "factual_error",
            "correction": m.group(2).strip(),
            "severity": "major",
        })
    for m in missing_pattern.finditer(raw):
        parsed_issues.append({
            "quote": m.group(1).strip(),
            "type": "missing_content",
            "detail": m.group(2).strip(),
            "severity": "minor",
        })
    if parsed_issues:
        return {"issues": parsed_issues, "raw_text": raw}

    return {"issues": [{"raw": raw}], "raw_text": raw}


class TurboDecoder:
    """
    Iterative Generator-Critic refinement analogous to turbo decoding.

    Key properties matching real turbo codes:
    1. Two SISO decoders (generator + critic) with matching capability
    2. INTERLEAVER: rotating evaluation lens decorrelates critic observations
    3. Extrinsic information exchange — each pass produces NEW info, not repetition
    4. Extrinsic scaling (α): limits corrections per round to prevent oscillation
    5. Fixed-iteration execution — regression protection makes extra rounds free
    """

    # Severity ordering for extrinsic scaling (highest priority first)
    _SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2, "": 3}

    def __init__(
        self,
        generator: AgentChannel,
        critic: AgentChannel | None,
        scorer: QualityScorer,
        max_iterations: int = 6,
        quality_threshold: float = 0.90,
        # --- v2 convergence control ---
        early_exit: bool = False,
        convergence_window: int = 2,
        convergence_epsilon: float = 0.015,
        # --- v5 extrinsic scaling + severity floor ---
        # Fraction of corrections to apply per round (α ∈ (0, 1]).
        # In real turbo decoders, extrinsic LLRs are scaled by α < 1 to
        # prevent oscillation (Vogt & Fingscheidt, "Improving the max-log-MAP
        # turbo decoder", Electronics Letters 2000). v4 used α=0.7 but
        # empirical analysis on 14B models showed the FIRST refinement
        # round degrades quality most (0.627 → 0.592 on hard tasks) —
        # consistent with over-correction even with α=0.7. v5 lowers α to
        # 0.5 for aggressive damping from round 1.
        extrinsic_scale: float = 0.5,
        # Severity floor — only apply corrections at or above this severity.
        # Maps to LLR magnitude thresholding: extrinsic info below a reliability
        # floor is discarded as noise. "minor" = likely fabricated on polished
        # answers, so drop them entirely. Accept only "major" and "critical".
        severity_floor: str = "major",
        # Max corrections per round (hard cap, applied after scaling).
        # Acts as LLR clipping — prevents any single round from making
        # too many changes regardless of how many issues the critic finds.
        max_corrections_per_round: int = 2,
    ):
        self.generator = generator
        # Default: use provided critic channel (same-complexity, communication-faithful).
        # Can be overridden to use a stronger model (e.g. judge) via config.
        self.critic = critic or generator
        self.scorer = scorer
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        # early_exit=False: run all iterations (standard turbo decoder).
        # early_exit=True: allow score-plateau and empty-critique early stopping.
        self.early_exit = early_exit
        self.convergence_window = convergence_window
        self.convergence_epsilon = convergence_epsilon
        self.extrinsic_scale = extrinsic_scale
        self.severity_floor = severity_floor
        self.max_corrections_per_round = max_corrections_per_round

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="turbo",
            config={
                "generator": self.generator.model,
                "critic": self.critic.model,
                "max_iterations": self.max_iterations,
                "quality_threshold": self.quality_threshold,
                "early_exit": self.early_exit,
                "extrinsic_scale": self.extrinsic_scale,
                "severity_floor": self.severity_floor,
                "max_corrections_per_round": self.max_corrections_per_round,
            },
        )

        all_outputs: list[AgentOutput] = []
        overhead: list[AgentOutput] = []
        accumulated_corrections: list[dict] = []  # Fix B: correction buffer
        score_history: list[float] = []

        gen_model = self.generator.model
        crit_model = self.critic.model
        judge_model = self.scorer.judge.model
        logger.info(
            f"Turbo [{task.id}]: generator={gen_model}, "
            f"critic={crit_model}, judge={judge_model}"
        )

        # ---- Iteration 0: initial generation ----
        gen_out = self.generator.transmit(task.request)
        gen_out.quality_score = self.scorer.score(
            task.prompt, gen_out.text, reference=task.reference, task=task
        )
        all_outputs.append(gen_out)
        score_history.append(gen_out.quality_score)

        best_text = gen_out.text
        best_score = gen_out.quality_score
        run.rounds = 1
        logger.info(
            f"Turbo [{task.id}] iter 0: generator({gen_model}) → "
            f"scored by judge({judge_model}) = {best_score:.3f}"
        )

        if gen_out.quality_score >= self.quality_threshold:
            run.individual_outputs = all_outputs
            run.combined_output = best_text
            run.final_quality = gen_out.quality_score
            run.compute_metrics()
            return run

        # ---- Turbo iterations: Generator <-> Critic exchange ----
        # v5: adaptive α damping + early stop on regression streak.
        # - On regression: halve α (stronger extrinsic scaling for next round)
        # - On improvement: relax α toward initial value
        # - After 2 consecutive regressions: break (EXIT trajectory diverging)
        # Regression protection (best-of-sequence) guarantees the final output
        # is never worse than the initial attempt, regardless of iteration path.
        current_alpha = self.extrinsic_scale
        consecutive_regressions = 0
        for iteration in range(1, self.max_iterations):
            # --- Critic pass (decoder 2): generate extrinsic information ---
            logger.debug(
                f"Turbo [{task.id}] iter {iteration}: critic({crit_model}) reviewing..."
            )
            extrinsic_text, critic_output = self._critic_pass(
                task, best_text, best_score, accumulated_corrections,
                iteration=iteration,
            )
            overhead.append(critic_output)

            # Parse structured critique
            critique_data = _parse_structured_critique(extrinsic_text)
            new_issues = critique_data["issues"]

            # --- Convergence check ---
            no_new_issues = (
                len(new_issues) == 0
                or (len(new_issues) == 1
                    and not new_issues[0].get("raw")
                    and not new_issues[0].get("quote"))
            )

            if no_new_issues:
                if best_score >= self.quality_threshold * 0.9:
                    logger.info(
                        f"Turbo iter {iteration}: no new issues, quality "
                        f"{best_score:.3f} near threshold — converged"
                    )
                    break
                elif self.early_exit:
                    logger.info(
                        f"Turbo iter {iteration}: critic found no issues but quality "
                        f"{best_score:.3f} is low — stopping (early_exit=True)"
                    )
                    break
                else:
                    logger.info(
                        f"Turbo iter {iteration}: critic found no issues but quality "
                        f"{best_score:.3f} is low — forcing continuation"
                    )
                    if not new_issues:
                        new_issues = [{"raw": critique_data["raw_text"]}]

            # Score-trajectory convergence (only when early_exit=True)
            if self.early_exit and self._score_plateau(score_history):
                logger.info(
                    f"Turbo iter {iteration}: score plateau detected "
                    f"({score_history[-self.convergence_window:]}) — stopping"
                )
                break

            # --- Extrinsic scaling with current α (may be damped from regressions) ---
            scaled_issues = self._scale_extrinsic(new_issues, alpha=current_alpha)

            # --- Generator pass (decoder 1): apply scaled corrections ---
            logger.info(
                f"Turbo [{task.id}] iter {iteration}: generator({gen_model}) "
                f"applying {len(scaled_issues)}/{len(new_issues)} corrections "
                f"(α={current_alpha:.2f})"
            )
            refined = self._generator_pass(
                task, best_text, scaled_issues
            )

            # --- Score via comparative scoring ---
            refined.quality_score = self.scorer.score_comparative(
                task.prompt,
                candidate=refined.text,
                baseline=best_text,
                baseline_score=best_score,
                reference=task.reference,
            )
            all_outputs.append(refined)
            run.rounds = iteration + 1
            score_history.append(refined.quality_score)

            # Regression protection + adaptive α damping.
            if refined.quality_score >= best_score:
                delta = refined.quality_score - best_score
                logger.info(
                    f"Turbo [{task.id}] iter {iteration}: "
                    f"gen({gen_model}) → judge({judge_model}) = {refined.quality_score:.3f} "
                    f"({'+' if delta >= 0 else ''}{delta:.3f}, accepted)"
                )
                best_text = refined.text
                best_score = refined.quality_score
                # Only add APPLIED corrections to dedup buffer — rejected ones
                # are still present in best_text and should remain fair game
                # for the critic in later rounds.
                accumulated_corrections.extend(
                    c for c in scaled_issues if "quote" in c
                )
                consecutive_regressions = 0
                # Relax α back toward initial value after successful iteration
                current_alpha = min(
                    self.extrinsic_scale, current_alpha * 1.2
                )
            else:
                logger.info(
                    f"Turbo [{task.id}] iter {iteration}: "
                    f"gen({gen_model}) → judge({judge_model}) = {refined.quality_score:.3f} "
                    f"< best {best_score:.3f} (regression, keeping best)"
                )
                consecutive_regressions += 1
                # Damp α: halve scaling factor for next round (stronger damping)
                current_alpha = max(0.1, current_alpha * 0.5)

            if best_score >= self.quality_threshold:
                break

            # Early stop: 2 consecutive regressions = EXIT trajectory diverging.
            # Further iterations waste tokens without recovering. Regression
            # protection means the final output is still the best seen so far.
            if consecutive_regressions >= 2:
                logger.info(
                    f"Turbo [{task.id}] iter {iteration}: "
                    f"{consecutive_regressions} consecutive regressions — "
                    f"stopping (decoder diverging, best={best_score:.3f})"
                )
                break

        # ---- Final assembly ----
        run.individual_outputs = all_outputs
        run.overhead_outputs = overhead
        run.combined_output = best_text
        # Use the best observed score directly — rescoring introduces noise
        # because binary checklist scoring can flip borderline checks differently
        # on each call. The best_score was already recorded when selected.
        run.final_quality = best_score

        run.compute_metrics()
        return run

    # ==================================================================
    # Helpers
    # ==================================================================

    def _scale_extrinsic(
        self, issues: list[dict], alpha: float | None = None
    ) -> list[dict]:
        """
        Extrinsic information scaling + severity floor.

        In real turbo decoders, extrinsic LLRs are multiplied by a scaling
        factor α < 1 and also thresholded to reject low-magnitude LLRs as
        noise. We apply both:

        1. Severity floor: drop issues below the reliability threshold.
           Minor issues on already-polished answers are the critic
           fabricating problems — noise, not signal.
        2. α scaling: keep only top ceil(n * α) of the remaining structured
           corrections, sorted by severity. α may be damped per-iteration
           by the caller in response to regressions (adaptive damping).
        3. Hard cap: max_corrections_per_round (LLR clipping).
        """
        import math

        if alpha is None:
            alpha = self.extrinsic_scale

        if not issues:
            return issues

        # Separate structured (with severity) from raw fallback issues
        structured = [c for c in issues if "quote" in c]
        raw = [c for c in issues if "quote" not in c]

        # --- Severity floor: drop issues below threshold ---
        floor_rank = self._SEVERITY_ORDER.get(self.severity_floor, 1)
        structured = [
            c for c in structured
            if self._SEVERITY_ORDER.get(c.get("severity", ""), 3) <= floor_rank
        ]

        # Sort structured issues by severity (most severe first)
        structured.sort(
            key=lambda c: self._SEVERITY_ORDER.get(
                c.get("severity", ""), 3
            )
        )

        # Apply α scaling: keep top ceil(n * α) of what survived the floor.
        n_structured = len(structured)
        n_keep_structured = math.ceil(n_structured * alpha) if n_structured else 0
        # At least 1 structured correction if any survived the floor
        if n_structured > 0:
            n_keep_structured = max(n_keep_structured, 1)

        # Hard cap on total corrections
        result = structured[:n_keep_structured]
        if len(result) < self.max_corrections_per_round and raw and not structured:
            # Fallback: if no structured issues survived, allow one raw item
            result.extend(raw[:1])
        result = result[:self.max_corrections_per_round]

        return result

    def _score_plateau(self, score_history: list[float]) -> bool:
        """
        Detect score plateau — no improvement over the convergence window.
        Analog: EXIT chart analysis — if mutual information trajectory flattened,
        additional iterations provide no benefit.
        """
        if len(score_history) < self.convergence_window + 1:
            return False
        recent = score_history[-(self.convergence_window + 1):]
        return max(recent) - min(recent) < self.convergence_epsilon

    # ==================================================================
    # Interleaver — the key turbo code differentiator
    # ==================================================================
    #
    # In real turbo codes, an interleaver shuffles the bit sequence between
    # the two component decoders. This decorrelates their inputs so that
    # each decoder's extrinsic information is genuinely independent.
    #
    # Without an interleaver, two identical decoders produce correlated
    # output and turbo decoding degrades to simple iterative decoding
    # (identical to HARQ-IR).
    #
    # Our interleaver analog: each iteration, the critic evaluates the
    # output through a DIFFERENT LENS (correctness, completeness,
    # reasoning, clarity). This forces the critic to examine different
    # aspects of the answer, producing decorrelated extrinsic information.
    #
    # The lens cycle ensures that even with the same model, each turbo
    # iteration explores a different failure mode — like rotating the
    # interleaver pattern across decoding iterations.

    _INTERLEAVER_LENSES = [
        {
            "name": "correctness",
            "instruction": (
                "Focus on FACTUAL CORRECTNESS. Check every claim, number, name, "
                "and technical detail. Is the answer actually right? Are there "
                "any factual errors, wrong numbers, or incorrect reasoning steps?"
            ),
        },
        {
            "name": "completeness",
            "instruction": (
                "Focus on COMPLETENESS. Does the answer fully address the task? "
                "Are there missing parts, skipped steps, unexplained points, or "
                "important details that were left out? What should be added?"
            ),
        },
        {
            "name": "reasoning",
            "instruction": (
                "Focus on REASONING and LOGIC. Are the reasoning steps sound? "
                "Are there logical gaps, unsupported conclusions, circular "
                "arguments, or flawed deductions? Check the chain of reasoning."
            ),
        },
        {
            "name": "clarity",
            "instruction": (
                "Focus on CLARITY and PRECISION. Is the answer clear and "
                "unambiguous? Are there vague statements, contradictions, "
                "poorly explained concepts, or confusing structure?"
            ),
        },
    ]

    def _critic_pass(
        self,
        task: TaskItem,
        current_text: str,
        current_score: float,
        prior_corrections: list[dict],
        iteration: int = 0,
    ) -> tuple[str, AgentOutput]:
        """
        Critic produces structured extrinsic information — new insights not
        already conveyed.

        The interleaver adds a light focus hint per iteration (correctness →
        completeness → reasoning → clarity) to decorrelate observations across
        iterations, but does NOT restrict the critic from flagging other issues.
        This matches HARQ-IR's proven prompt structure with a small addition.
        """
        # Select interleaver lens for this iteration
        lens = self._INTERLEAVER_LENSES[iteration % len(self._INTERLEAVER_LENSES)]
        logger.debug(f"Turbo interleaver: iteration {iteration} → lens '{lens['name']}'")

        prompt = (
            f"## Task\n{task.prompt}\n\n"
            f"## AI Response to Evaluate (current quality: {current_score:.2f}/1.00)\n"
            f"{current_text}\n\n"
        )

        if task.reference:
            prompt += f"## Reference Answer\n{task.reference}\n\n"

        # List prior corrections for deduplication
        if prior_corrections:
            structured_prior = [c for c in prior_corrections if "quote" in c]
            if structured_prior:
                prompt += "## Previously Identified Issues (DO NOT repeat these)\n"
                for c in structured_prior:
                    quote = c.get("quote", "?")[:120]
                    prompt += f'- Already fixed: "{quote}"\n'
                prompt += "\n"

        prompt += (
            "## Instructions\n"
            "Find SPECIFIC errors or gaps in the response. For each issue:\n"
            "1. QUOTE the exact problematic text from the response\n"
            "2. State what is wrong and why\n"
            "3. Provide the specific correction or addition needed\n\n"
            f"Pay special attention to {lens['name'].upper()} issues this round, "
            "but flag ANY problem you find.\n\n"
            "Respond with a JSON array of issues:\n"
            "```json\n"
            "[\n"
            '  {"quote": "exact wrong text", "type": "factual_error", '
            '"correction": "what it should say", "severity": "major"},\n'
            '  {"quote": "text before gap", "type": "missing_content", '
            '"detail": "content that should be added", "severity": "minor"}\n'
            "]\n"
            "```\n\n"
            "Issue types: factual_error, missing_content, reasoning_gap, unclear\n"
            "Severity: critical, major, minor\n\n"
            "IMPORTANT:\n"
            "- Do NOT invent problems that don't exist — only flag genuine errors\n"
            "- Do NOT repeat previously identified issues (listed above)\n"
            "- QUOTE the exact text from the response, not a paraphrase\n"
            "- If no NEW issues remain, respond with exactly: []\n"
        )

        result = self.critic.transmit(prompt, temperature=0.2)
        return result.text, result

    def _generator_pass(
        self,
        task: TaskItem,
        current_text: str,
        corrections: list[dict],
    ) -> AgentOutput:
        """
        Generator refines answer using extrinsic information from critic.
        Fix B: correction-based prompt (non-destructive accumulation).

        Analogous to SISO decoder 1 using extrinsic LLRs from decoder 2
        to update its belief about each bit — it doesn't re-decode from scratch,
        it incrementally adjusts.
        """
        structured = [c for c in corrections if "quote" in c]
        raw_items = [c for c in corrections if "raw" in c and "quote" not in c]

        if structured:
            # --- Correction-based prompt (preferred) ---
            parts = [
                f"## Original Task\n{task.prompt}\n\n",
                f"## Your Current Answer\n{current_text}\n\n",
                "## Specific Corrections to Apply\n"
                "Apply each correction below. Keep ALL other content exactly as-is.\n\n",
            ]
            for i, c in enumerate(structured, 1):
                quote = c.get("quote", "")
                ctype = c.get("type", "issue")
                severity = c.get("severity", "")
                sev_tag = f" [{severity}]" if severity else ""

                if ctype == "missing_content":
                    detail = c.get("detail", c.get("correction", ""))
                    parts.append(
                        f"{i}. ADD{sev_tag} near \"{quote[:80]}\":\n"
                        f"   {detail}\n\n"
                    )
                else:
                    correction = c.get("correction", "")
                    parts.append(
                        f"{i}. FIX{sev_tag}: \"{quote[:80]}\"\n"
                        f"   Change to: {correction}\n\n"
                    )

            if raw_items:
                parts.append("## Additional Feedback\n")
                for item in raw_items:
                    parts.append(f"{item['raw'][:500]}\n\n")

            parts.append(
                "## Instructions\n"
                "Apply ONLY the corrections listed above.\n"
                "- Do NOT rewrite or restructure content not mentioned above\n"
                "- Do NOT remove correct content\n"
                "- Output the COMPLETE corrected answer\n"
            )
            prompt = "".join(parts)
        else:
            # --- Fallback: full-context refinement ---
            feedback_text = "\n\n".join(
                c.get("raw", str(c))[:600] for c in corrections
            )
            prompt = (
                f"## Original Task\n{task.prompt}\n\n"
                f"## Your Current Answer\n{current_text}\n\n"
                f"## Feedback to Address\n{feedback_text}\n\n"
                "## Instructions\n"
                "Produce a COMPLETE improved answer that:\n"
                "1. Fixes every specific issue from the feedback\n"
                "2. Preserves everything that was already correct\n"
                "3. Does NOT over-correct — keep correct content as-is\n\n"
                "Output the COMPLETE answer (not a diff or summary)."
            )

        return self.generator.transmit(prompt, temperature=0.3)
