"""
18 — Deterministic score modes (beyond code).

Most examples score with the LLM judge's 15-criterion checklist. But for
tasks with a *verifiable* answer, AgentCodec can extract that answer and
grade it deterministically, then blend it with the judge. The non-code
modes (``15``/``16`` cover ``code``):

  * ``exact_letter``  — multiple-choice: extract the chosen letter (A/B/C/D)
  * ``numeric``       — extract the final number, compare exactly
  * ``relaxed``       — numeric within a relative tolerance (default ±5%)
  * ``yes_no``        — extract a yes/no verdict
  * ``exact_match``   — normalized full-string equality

Under the default ``score_strategy="blended"``::

    final_quality = 0.6 × deterministic{0 or 1} + 0.4 × judge_checklist

So a correct-but-terse answer can still lose points on presentation, and a
fluent-but-wrong answer is capped at 0.4. This example runs the same
``baseline`` technique on one prompt per mode, prints the blended
``final_quality`` with its scoring-path note, and — to make the blend
concrete — also calls the deterministic scorer directly so you can see the
``{0, 1}`` component in isolation (the same call the library makes
internally).

Run:
    python examples/18_score_modes.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule
from agentcodec.scoring import (
    extract_letter,
    extract_number,
    extract_yes_no,
    score_deterministic,
)

from _common import judge_block, model_block, print_result


def build() -> ReliabilityModule:
    return ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.3)],
        "judge": judge_block(),
        "strategy": {"type": "fixed", "technique": "baseline"},
        "defaults": {"category": "auto"},
    })


# (label, score_mode, prompt, reference, category)
CASES = [
    (
        "exact_letter (MMLU-style)",
        "exact_letter",
        "Which planet is the largest in our Solar System?\n"
        "A) Earth  B) Saturn  C) Jupiter  D) Neptune\n"
        "Answer with the letter only.",
        "C",
        "qa",
    ),
    (
        "numeric (math)",
        "numeric",
        "A shop sells pens at 3 for $2. How many dollars do 12 pens cost? "
        "Give the final answer as a number.",
        "8",
        "math",
    ),
    (
        "yes_no (verification)",
        "yes_no",
        "Is 91 a prime number? Answer yes or no, then briefly justify.",
        "no",          # 91 = 7 × 13
        "qa",
    ),
]


def main() -> None:
    mod = build()
    extractors = {
        "exact_letter": extract_letter,
        "numeric": extract_number,
        "yes_no": extract_yes_no,
    }
    with mod:
        for label, mode, prompt, reference, category in CASES:
            result = mod.run(
                prompt, category=category, reference=reference, score_mode=mode,
            )
            print_result(result, label=label, score_mode=mode)

            # Make the blend concrete: show the extracted value and the
            # raw deterministic {0, 1} component the library blended with
            # the judge to get `final_quality` above.
            answer = (result.text or "").strip()
            extracted = extractors[mode](answer)
            det = score_deterministic(mode, answer, reference)
            print(f"  extracted  : {extracted!r}  (reference {reference!r})")
            print(f"  det signal : {det:.0f}   "
                  f"→ final = 0.6 × {det:.0f} + 0.4 × judge = {result.final_quality:.3f}")


if __name__ == "__main__":
    main()
