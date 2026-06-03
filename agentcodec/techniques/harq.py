"""
Technique 2: Hybrid ARQ (HARQ)

Communication analog:
- HARQ-CC (Chase Combining): Retransmit same data, combine all copies equally.
  -> Agent retries the same prompt, combines all attempts.
- HARQ-IR (Incremental Redundancy): Retransmit with new parity bits.
  -> Agent retries with critic feedback (new information), iteratively improving.

Stops when quality exceeds threshold or max rounds reached.

Design rationale (v2 — communication-faithful fixes):
- Per-round comparative scoring breaks the judge quantization barrier
  (analog: soft-output decoding vs hard-decision decoding)
- Structured critique forces independent parity per round
  (analog: proper puncturing ensures new parity bits each retransmission)
- Correction-based refinement prevents destructive rewrite
  (analog: soft buffer accumulation, not re-decode-from-scratch)
- Score-trajectory convergence replaces unreliable text-based PASS detection
  (analog: EXIT chart monitoring vs decoder self-report)
- Fixed-iteration mode (early_exit=False) matches real turbo/HARQ decoders
"""

from __future__ import annotations

import json
import logging
import re

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, HARQMode, ReliabilityRun, TaskItem

logger = logging.getLogger(__name__)


def _parse_structured_critique(critique_text: str) -> dict:
    """
    Parse a structured critique response into a normalized format.

    Returns:
        {
            "issues": [{"quote": ..., "type": ..., "correction": ..., "severity": ...}, ...],
            "raw_text": str  (original text, used as fallback)
        }
    If JSON parsing fails, falls back to treating the text as a single
    unstructured issue — ensures the system degrades gracefully for weak critics.
    """
    raw = critique_text.strip()

    # --- Try JSON parsing (best case: critic produced clean JSON) ---
    for text_to_try in [raw, raw.strip("`").strip()]:
        try:
            data = json.loads(text_to_try)
            if isinstance(data, list):
                return {"issues": data, "raw_text": raw}
            if isinstance(data, dict) and "issues" in data:
                return {"issues": data["issues"], "raw_text": raw}
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Try extracting JSON from markdown code blocks ---
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

    # --- Try extracting an inline JSON array ---
    # Greedy `\[.*\]` (single quantifier, no nested lazy quantifiers): scans
    # once forward to the last `]` in O(n). The previous nested-lazy pattern
    # `\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]` was catastrophically backtrack-
    # prone on critic outputs containing many `{...}` fragments without a
    # closing `]` and could hang for hours. Length is naturally bounded by
    # the critic's max_tokens.
    arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return {"issues": data, "raw_text": raw}
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Check for explicit PASS / no-issues signals ---
    upper = raw.upper()
    pass_signals = [
        "PASS" in upper and "NO SIGNIFICANT ISSUES" in upper,
        "NO NEW ISSUES" in upper and len(raw) < 300,
        raw.strip() == "[]",
        "NO ISSUES" in upper and "FOUND" in upper and len(raw) < 300,
    ]
    if any(pass_signals):
        return {"issues": [], "raw_text": raw}

    # --- Fallback: treat entire text as one unstructured issue ---
    # This ensures the refinement loop still gets feedback even when
    # the critic can't produce structured output (e.g. 3B models).
    return {"issues": [{"raw": raw}], "raw_text": raw}


class HARQService:
    """
    Implements HARQ-CC and HARQ-IR for iterative agent refinement.
    """

    def __init__(
        self,
        channel: AgentChannel,
        scorer: QualityScorer,
        mode: HARQMode = HARQMode.IR,
        max_rounds: int = 5,
        quality_threshold: float = 0.85,
        critic_channel: AgentChannel | None = None,
        # --- v2 convergence control ---
        early_exit: bool = False,
        convergence_window: int = 2,
        convergence_epsilon: float = 0.015,
    ):
        self.channel = channel
        self.scorer = scorer
        self.mode = mode
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        # Default: same-model critic (communication-faithful — same-complexity decoder).
        # Can be overridden to use a stronger model (e.g. judge) via config.
        self.critic = critic_channel or channel
        # early_exit=False: run all rounds (communication-faithful, like real decoders).
        # early_exit=True: allow score-plateau and empty-critique early stopping.
        self.early_exit = early_exit
        self.convergence_window = convergence_window
        self.convergence_epsilon = convergence_epsilon

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique=f"harq_{self.mode.value}",
            config={
                "mode": self.mode.value,
                "max_rounds": self.max_rounds,
                "quality_threshold": self.quality_threshold,
                "model": self.channel.model,
                "critic_model": self.critic.model,
                "early_exit": self.early_exit,
            },
        )

        if self.mode == HARQMode.CC:
            self._run_chase_combining(task, run)
        else:
            self._run_incremental_redundancy(task, run)

        run.compute_metrics()
        return run

    # ==================================================================
    # HARQ-CC (Chase Combining) — unchanged
    # ==================================================================

    def _run_chase_combining(self, task: TaskItem, run: ReliabilityRun):
        """
        HARQ-CC: Generate multiple attempts, combine all equally.
        Like chase combining — same codeword retransmitted, soft-combine at receiver.
        """
        outputs: list[AgentOutput] = []

        for round_num in range(1, self.max_rounds + 1):
            out = self.channel.transmit(task.request)
            out.quality_score = self.scorer.score(
                task.prompt, out.text, reference=task.reference, task=task
            )
            outputs.append(out)
            run.rounds = round_num

            if out.quality_score >= self.quality_threshold:
                run.combined_output = out.text
                run.final_quality = out.quality_score
                run.individual_outputs = outputs
                return

        # Combine all attempts equally (chase combining)
        run.individual_outputs = outputs
        combined_text, combine_output = self._combine_cc(outputs, task.prompt)
        run.combined_output = combined_text
        run.overhead_outputs = [combine_output]

        best_output = max(outputs, key=lambda o: o.quality_score)
        run.final_quality = self.scorer.score_comparative(
            task.prompt,
            candidate=run.combined_output,
            baseline=best_output.text,
            baseline_score=best_output.quality_score,
            reference=task.reference,
        )

    # ==================================================================
    # HARQ-IR (Incremental Redundancy) — v2 with all fixes
    # ==================================================================

    def _run_incremental_redundancy(self, task: TaskItem, run: ReliabilityRun):
        """
        HARQ-IR with communication-faithful improvements:

        Modification A — Comparative scoring per round:
            Independent scoring quantizes to ~6 levels (55% of scores = 0.730).
            Comparative scoring shows the judge both versions side-by-side,
            breaking the quantization barrier. This is the analog of soft-output
            (SISO) decoding vs hard-decision decoding — the decoder produces
            continuous LLRs, not hard bits.

        Modification B — Non-destructive accumulation (correction-based refinement):
            Real HARQ-IR accumulates received parity in a soft buffer and
            re-decodes from the full buffer. Our analog: instead of prompting
            the generator to "rewrite the entire answer" (destructive), we
            provide specific corrections to apply (accumulative). This prevents
            the generator from losing correct content during rewrites.

        Modification C — Structured critique (independent parity):
            Real IR requires each retransmission to carry NEW, independent
            parity bits. We enforce this by requiring the critic to produce
            structured JSON with quoted issues, enabling deduplication across
            rounds. Vague feedback ("could be better") is the analog of sending
            the same parity bits again — zero incremental information.

        Modification D — Fixed-iteration mode (early_exit=False):
            Real turbo/HARQ decoders run a fixed number of iterations.
            The PASS/CONVERGED text detection is unreliable for weak models
            (3B critics claim PASS on 29% of tasks despite low quality).
            Default: run all rounds, rely on regression protection.

        Modification E — Score-trajectory convergence:
            When early_exit=True, convergence is detected by monitoring the
            score trajectory (plateau detection), not by parsing the critic's
            text. This is the analog of EXIT chart analysis in turbo codes.
        """
        # ---- Round 1: initial attempt (absolute scoring) ----
        current_output = self.channel.transmit(task.request)
        current_output.quality_score = self.scorer.score(
            task.prompt, current_output.text, reference=task.reference, task=task
        )
        outputs = [current_output]
        run.rounds = 1

        best_output = current_output
        score_history = [current_output.quality_score]

        # If already above threshold, no retransmission needed
        if current_output.quality_score >= self.quality_threshold:
            run.individual_outputs = outputs
            run.combined_output = current_output.text
            run.final_quality = current_output.quality_score
            return

        # ---- Iterative refinement with accumulated corrections ----
        overhead: list[AgentOutput] = []
        accumulated_corrections: list[dict] = []  # Modification B: correction buffer (soft buffer)

        for round_num in range(2, self.max_rounds + 1):
            # --- Critic pass: generate new parity (Modification C: structured) ---
            critique_text, critique_output = self._get_structured_critique(
                task.prompt,
                current_output.text,
                task.reference,
                current_score=current_output.quality_score,
                prior_corrections=accumulated_corrections,
            )
            overhead.append(critique_output)

            # Parse structured critique
            critique_data = _parse_structured_critique(critique_text)
            new_issues = critique_data["issues"]

            # --- Convergence check (Modification D + E) ---
            no_new_issues = (
                len(new_issues) == 0
                or ((len(new_issues) == 1 and not new_issues[0].get("raw"))
                and not new_issues[0].get("quote"))
            )

            if no_new_issues:
                if self.early_exit and current_output.quality_score >= self.quality_threshold * 0.9:
                    # Genuine convergence: no issues AND quality is high
                    logger.info(
                        f"HARQ-IR round {round_num}: no new issues, quality "
                        f"{current_output.quality_score:.3f} near threshold — stopping"
                    )
                    break
                elif self.early_exit:
                    logger.info(
                        f"HARQ-IR round {round_num}: critic found no issues but quality "
                        f"{current_output.quality_score:.3f} is low — stopping (early_exit=True)"
                    )
                    break
                else:
                    # Modification D: force continue despite critic saying no issues
                    logger.info(
                        f"HARQ-IR round {round_num}: critic found no issues but quality "
                        f"{current_output.quality_score:.3f} is low — forcing continuation"
                    )
                    # Use raw critique text as fallback feedback
                    if not new_issues:
                        new_issues = [{"raw": critique_data["raw_text"]}]

            # Modification E: score-trajectory convergence (only when early_exit=True)
            if self.early_exit and self._score_plateau(score_history):
                logger.info(
                    f"HARQ-IR round {round_num}: score plateau detected "
                    f"({score_history[-self.convergence_window:]}) — stopping"
                )
                break

            # --- Accumulate corrections (Modification B: soft buffer) ---
            structured_issues = [c for c in new_issues if "quote" in c]
            accumulated_corrections.extend(structured_issues)

            # --- Generator pass: apply corrections (Modification B) ---
            refinement_prompt = self._build_correction_prompt(
                task.prompt, best_output.text, new_issues
            )

            refined = self.channel.transmit(refinement_prompt)

            # --- Score via comparative scoring (Modification A) ---
            # Comparative scoring shows the judge both versions side-by-side,
            # producing finer-grained scores than independent scoring.
            # This breaks the 0.730 quantization barrier where 55% of
            # independent scores collapse to a single value.
            refined.quality_score = self.scorer.score_comparative(
                task.prompt,
                candidate=refined.text,
                baseline=current_output.text,
                baseline_score=current_output.quality_score,
                reference=task.reference,
            )

            outputs.append(refined)
            run.rounds = round_num
            score_history.append(refined.quality_score)

            # Regression protection: only advance if quality didn't drop
            if refined.quality_score >= best_output.quality_score:
                current_output = refined
                best_output = refined
                logger.info(
                    f"HARQ-IR round {round_num}: quality {refined.quality_score:.3f} "
                    f"(+{refined.quality_score - score_history[-2]:.3f}, accepted)"
                )
            else:
                logger.info(
                    f"HARQ-IR round {round_num}: quality {refined.quality_score:.3f} "
                    f"< best {best_output.quality_score:.3f} (regression, keeping best)"
                )
                current_output = best_output

            if best_output.quality_score >= self.quality_threshold:
                break

        # ---- Final assembly ----
        run.individual_outputs = outputs
        run.overhead_outputs = overhead
        run.combined_output = best_output.text
        # Use the best observed score directly — rescoring introduces noise
        # because binary checklist scoring can flip borderline checks differently
        # on each call. The best_output was already scored when selected.
        run.final_quality = best_output.quality_score

    # ==================================================================
    # Helpers
    # ==================================================================

    def _score_plateau(self, score_history: list[float]) -> bool:
        """
        Detect score plateau — no improvement over the convergence window.

        Analog: EXIT chart analysis in turbo codes. If the mutual information
        trajectory has flattened, additional iterations won't help.
        """
        if len(score_history) < self.convergence_window + 1:
            return False
        recent = score_history[-(self.convergence_window + 1):]
        return max(recent) - min(recent) < self.convergence_epsilon

    def _get_structured_critique(
        self,
        prompt: str,
        output: str,
        reference: str | None,
        current_score: float = 0.0,
        prior_corrections: list[dict] | None = None,
    ) -> tuple[str, AgentOutput]:
        """
        Critic generates structured 'parity information' — specific, quotation-based
        issues that can be mechanically applied as corrections.

        Modification C: By requiring structured output with exact quotes, we ensure:
        1. Each issue is specific and verifiable (not vague)
        2. Deduplication across rounds is trivial (compare quotes)
        3. The generator receives precise correction instructions
        4. Each round produces genuinely NEW parity (not correlated repeats)

        Falls back gracefully: if the critic can't produce JSON (e.g. 3B models),
        the parser returns the raw text as a single unstructured issue, and the
        refinement prompt falls back to the v1 full-rewrite style.
        """
        critic_prompt = (
            f"## Task\n{prompt}\n\n"
            f"## AI Response to Evaluate (current quality: {current_score:.2f}/1.00)\n"
            f"{output}\n\n"
        )
        if reference:
            critic_prompt += f"## Reference Answer\n{reference}\n\n"

        # List prior corrections for deduplication (Modification C)
        if prior_corrections:
            structured_prior = [c for c in prior_corrections if "quote" in c]
            if structured_prior:
                critic_prompt += (
                    "## Previously Identified Issues (DO NOT repeat these)\n"
                )
                for c in structured_prior:
                    quote = c.get("quote", "?")[:120]
                    critic_prompt += f'- Already fixed: "{quote}"\n'
                critic_prompt += "\n"

        critic_prompt += (
            "## Instructions\n"
            "Find SPECIFIC errors or gaps in the response. For each issue:\n"
            "1. QUOTE the exact problematic text from the response\n"
            "2. State what is wrong and why\n"
            "3. Provide the specific correction or addition needed\n\n"
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

        result = self.critic.transmit(critic_prompt, temperature=0.2)
        return result.text, result

    def _build_correction_prompt(
        self,
        task_prompt: str,
        current_text: str,
        corrections: list[dict],
    ) -> str:
        """
        Build a correction-based refinement prompt (Modification B: non-destructive accumulation).

        Instead of asking the generator to "produce a COMPLETE improved answer"
        (which risks losing correct content), we provide specific corrections
        to apply while preserving everything else.

        Analog: In real HARQ-IR, the decoder accumulates all received parity
        in a soft buffer and re-decodes from the complete buffer. It does NOT
        discard previously received symbols. Our correction-based prompt is
        the agent analog — the original answer is the "received signal" and
        corrections are "accumulated parity bits."

        Falls back to full-rewrite style when corrections aren't structured
        (e.g. when the critic couldn't produce JSON).
        """
        structured = [c for c in corrections if "quote" in c]
        raw_items = [c for c in corrections if "raw" in c and "quote" not in c]

        if structured:
            # --- Correction-based prompt (preferred) ---
            parts = [
                f"## Original Task\n{task_prompt}\n\n",
                f"## Current Answer\n{current_text}\n\n",
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

            # Append any raw feedback as supplementary context
            if raw_items:
                parts.append("## Additional Feedback\n")
                for item in raw_items:
                    parts.append(f"{item['raw'][:500]}\n\n")

            parts.append(
                "## Instructions\n"
                "Apply ONLY the corrections listed above.\n"
                "- Do NOT rewrite or restructure content that is not mentioned\n"
                "- Do NOT remove correct content\n"
                "- Do NOT add information beyond what the corrections specify\n"
                "- Output the COMPLETE corrected answer\n"
            )
            return "".join(parts)

        else:
            # --- Fallback: full-rewrite prompt (for unstructured critique) ---
            feedback_text = "\n\n".join(
                c.get("raw", str(c))[:600] for c in corrections
            )
            return (
                f"## Original Task\n{task_prompt}\n\n"
                f"## Your Current Answer\n{current_text}\n\n"
                f"## Feedback to Address\n{feedback_text}\n\n"
                "## Instructions\n"
                "Produce a COMPLETE improved answer that:\n"
                "1. Fixes every specific issue from the feedback above\n"
                "2. Preserves everything that was already correct\n"
                "3. Does NOT over-correct — keep correct content as-is\n\n"
                "Output ONLY the complete improved answer."
            )

    def _combine_cc(
        self, outputs: list[AgentOutput], prompt: str
    ) -> tuple[str, AgentOutput]:
        """
        Combine all CC attempts via soft combining.
        Analog: chase combining averages received copies to boost SNR.
        """
        parts = []
        for i, o in enumerate(outputs):
            parts.append(
                f"### Attempt {i+1} [Quality: {o.quality_score:.2f}]\n{o.text}"
            )
        combine_prompt = (
            f"## Original Task\n{prompt}\n\n"
            f"## Multiple Independent Attempts\n\n"
            + "\n\n---\n\n".join(parts)
            + "\n\n## Chase Combining Instructions\n"
            "These are independent attempts at the same task. Combine them:\n\n"
            "1. **CONSENSUS**: Where do most attempts agree? This is high-confidence signal.\n"
            "2. **UNIQUE DETAILS**: What does each attempt include that others don't?\n"
            "3. **CONFLICTS**: Where do they disagree? Trust the higher-quality attempt, "
            "or reason about which is correct.\n"
            "4. **COMBINE**: Produce one answer that includes consensus + all unique "
            "correct details.\n\n"
            "Output ONLY the combined answer."
        )
        result = self.critic.transmit(combine_prompt, temperature=0.3)
        return result.text, result
