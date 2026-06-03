"""
Technique 5: Forward Error Correction (FEC)

Communication analog: FEC adds structured redundancy (parity bits) to the
transmitted signal. The receiver uses this redundancy to detect and correct
errors without retransmission. Code rate r = k/n determines the redundancy
level (lower rate = more redundancy = more protection).

Agent analog: Generate the main answer (systematic bits) and then generate
each parity section as a SEPARATE LLM call. Each parity call focuses on one
type of redundancy:
- Step-by-step reasoning / chain-of-thought (parity 1)
- Self-verification / consistency check (parity 2)
- Confidence assessment with error detection (parity 3)
- Alternative approach / independent re-derivation (parity 4)

The "decoder" receives ALL sections and cross-checks them for consistency,
using the redundancy to detect/correct errors in the main answer.

Design rationale (v2):
- Each parity section is a separate LLM call (not crammed into one prompt).
  This ensures each section gets the model's full attention and produces
  genuine redundancy — analogous to how real FEC computes each parity
  symbol independently from the information bits.
- The decoder sees all sections as separate inputs, enabling true
  syndrome decoding (cross-checking between independent parity checks).
- Code rate genuinely controls cost: r=0.75 → 2 calls, r=0.33 → 4 calls.
"""

from __future__ import annotations

import logging

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, ReliabilityRun, TaskItem

logger = logging.getLogger(__name__)


# Parity section definitions: each is a separate LLM call
PARITY_SECTIONS = {
    1.0: [],                                                    # No parity (uncoded)
    0.75: ["reasoning"],                                        # Rate 3/4: 1 parity call
    0.50: ["reasoning", "verification"],                        # Rate 1/2: 2 parity calls
    0.33: ["reasoning", "verification", "alternative"],         # Rate 1/3: 3 parity calls
    0.25: ["reasoning", "verification", "alternative", "confidence"],  # Rate 1/4: 4 parity calls
}

# Prompts for each parity section — each is sent as a separate LLM call
# with the original task AND the main answer as context.
SECTION_PROMPTS = {
    "reasoning": (
        "## Task\n{task}\n\n"
        "## Proposed Answer\n{answer}\n\n"
        "## Your Job: Step-by-Step Reasoning Check\n"
        "Work through this task step by step from scratch. Show your complete "
        "reasoning process. Do NOT simply restate the proposed answer — derive "
        "the answer independently and show every step.\n\n"
        "If your reasoning leads to a different conclusion than the proposed "
        "answer, clearly state the discrepancy."
    ),
    "verification": (
        "## Task\n{task}\n\n"
        "## Proposed Answer\n{answer}\n\n"
        "## Your Job: Verification and Error Detection\n"
        "Carefully verify every claim in the proposed answer:\n"
        "1. Check each factual claim — is it correct?\n"
        "2. Check the reasoning — does each step follow logically?\n"
        "3. Check completeness — is anything important missing?\n"
        "4. Check for contradictions — does any part conflict with another?\n\n"
        "For each issue found, state:\n"
        "- WHAT is wrong (quote the specific text)\n"
        "- WHY it is wrong\n"
        "- WHAT the correct version should be\n\n"
        "If no errors found, state: 'VERIFIED: No errors detected.'"
    ),
    "alternative": (
        "## Task\n{task}\n\n"
        "## Your Job: Independent Solution\n"
        "Solve this task using a COMPLETELY DIFFERENT approach or method than "
        "you might normally use. This serves as a cross-check.\n\n"
        "- If this is a reasoning problem, try a different reasoning strategy\n"
        "- If this is a factual question, approach it from a different angle\n"
        "- If this is a creative task, use a different structure or perspective\n\n"
        "Output ONLY your independent solution — do NOT reference any prior answer."
    ),
    "confidence": (
        "## Task\n{task}\n\n"
        "## Proposed Answer\n{answer}\n\n"
        "## Your Job: Confidence and Uncertainty Assessment\n"
        "For each major claim or section of the proposed answer, rate your "
        "confidence (HIGH / MEDIUM / LOW) and explain why:\n\n"
        "- HIGH: Strong evidence, well-established fact, clear logical derivation\n"
        "- MEDIUM: Reasonable but could be wrong, some uncertainty in reasoning\n"
        "- LOW: Uncertain, speculative, or based on assumptions that may not hold\n\n"
        "Flag any LOW confidence items as potential errors that need correction."
    ),
}


class FECService:
    """
    Forward Error Correction via structured redundancy.

    Each parity section is generated as a separate LLM call, ensuring
    genuine redundancy proportional to the code rate.
    """

    def __init__(
        self,
        channel: AgentChannel,
        scorer: QualityScorer,
        code_rate: float = 0.50,
        decoder_channel: AgentChannel | None = None,
    ):
        self.channel = channel
        self.scorer = scorer
        self.code_rate = code_rate
        self.decoder = decoder_channel or channel

        # Find closest supported code rate
        rates = sorted(PARITY_SECTIONS.keys())
        self.effective_rate = min(rates, key=lambda r: abs(r - code_rate))
        self.parity_sections = PARITY_SECTIONS[self.effective_rate]

    def run(self, task: TaskItem) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique=f"fec_{self.effective_rate}",
            config={
                "code_rate": self.code_rate,
                "effective_rate": self.effective_rate,
                "parity_sections": self.parity_sections,
                "num_parity_calls": len(self.parity_sections),
                "model": self.channel.model,
            },
        )

        # --- Phase 1: Generate systematic bits (main answer) ---
        main_output = self.channel.transmit(task.request)
        main_output.quality_score = self.scorer.score(
            task.prompt, main_output.text, reference=task.reference, task=task
        )
        run.individual_outputs.append(main_output)

        if not self.parity_sections:
            # Rate 1.0 — uncoded
            run.combined_output = main_output.text
            run.final_quality = main_output.quality_score
            run.compute_metrics()
            return run

        # --- Phase 2: Generate each parity section as a separate LLM call ---
        parity_outputs: list[AgentOutput] = []
        parity_texts: dict[str, str] = {}

        for section_name in self.parity_sections:
            prompt_template = SECTION_PROMPTS[section_name]
            section_prompt = prompt_template.format(
                task=task.prompt,
                answer=main_output.text,
            )
            parity_out = self.channel.transmit(section_prompt)
            parity_outputs.append(parity_out)
            parity_texts[section_name] = parity_out.text

            logger.debug(
                f"FEC parity '{section_name}': {len(parity_out.text)} chars, "
                f"{parity_out.token_count} tokens"
            )

        run.individual_outputs.extend(parity_outputs)

        # --- Phase 3: Decode — syndrome decoding using all sections ---
        decoded_text, decode_output = self._decode(
            task.prompt, main_output.text, parity_texts
        )
        run.overhead_outputs = [decode_output]
        run.combined_output = decoded_text

        # Score via comparative scoring (decoded vs raw main answer)
        run.final_quality = self.scorer.score_comparative(
            task.prompt,
            candidate=run.combined_output,
            baseline=main_output.text,
            baseline_score=main_output.quality_score,
            reference=task.reference,
        )

        run.compute_metrics()
        return run

    def _decode(
        self,
        original_prompt: str,
        main_answer: str,
        parity_sections: dict[str, str],
    ) -> tuple[str, AgentOutput]:
        """
        FEC decoder: syndrome decoding using main answer + all parity sections.

        Each parity section was generated independently, so discrepancies between
        them reveal errors (syndromes). The decoder cross-checks all sections and
        produces the corrected final answer.
        """
        parts = [
            f"## Original Task\n{original_prompt}\n\n",
            f"## Main Answer\n{main_answer}\n\n",
        ]

        # Add each parity section as a separate numbered block
        for i, (name, text) in enumerate(parity_sections.items(), 1):
            label = name.replace("_", " ").title()
            parts.append(f"## Parity Check {i}: {label}\n{text}\n\n")

        parts.append(
            "## Syndrome Decoding Instructions\n"
            "You have the main answer plus independent cross-checks above. "
            "Use them to DETECT AND CORRECT errors:\n\n"
            "**Step 1 — Check parity constraints:**\n"
        )

        if "reasoning" in parity_sections:
            parts.append(
                "- Does the step-by-step reasoning reach the same conclusion "
                "as the main answer? If not, which is correct?\n"
            )
        if "verification" in parity_sections:
            parts.append(
                "- Did the verification find errors? Are those findings valid?\n"
            )
        if "alternative" in parity_sections:
            parts.append(
                "- Does the independent solution agree with the main answer? "
                "If they differ, determine which is correct by comparing reasoning.\n"
            )
        if "confidence" in parity_sections:
            parts.append(
                "- Are any LOW confidence items actually wrong? Focus scrutiny there.\n"
            )

        parts.append(
            "\n**Step 2 — Identify errors (syndromes):**\n"
            "List any inconsistencies found between the main answer and the "
            "parity checks.\n\n"
            "**Step 3 — Correct and output:**\n"
            "Fix all detected errors. When sections disagree, trust the one with "
            "stronger reasoning or more evidence.\n"
            "Produce ONLY the final corrected answer — clean, complete, no labels."
        )

        decode_prompt = "".join(parts)
        result = self.decoder.transmit(decode_prompt, temperature=0.2)
        return result.text, result
