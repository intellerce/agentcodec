"""
16 — Code scoring, end-to-end through an LLM.

Companion to ``15_code_scoring.py`` (which calls the scorer on hand-
written candidates). This one drives the full pipeline:

    prompt → ReliabilityModule.run → LLM generates code →
    sandbox runs the test harness → final_quality reflects actual execution.

The task metadata carries ``source="humaneval"`` so the library
auto-infers ``score_mode="code"`` and routes scoring into the sandbox.
The blended formula then becomes
``0.6 × {0, 0.5, 1} + 0.4 × judge_15_criteria`` — see README
§"Code scoring" for the trade-off matrix.

Two runs on the same prompt, same model pool:

  1. ``baseline``  — single LLM call, scored once.
  2. ``harq_ir``   — iterative critic-and-refine; converges only when
                     the sandbox score crosses the quality threshold.
                     Often takes 2–3 rounds on this task with an 8B model.

Watch the per-round progress in the INFO log to see harq_ir's refinement
catch and fix code that the baseline ships broken. Set the sandbox via
``$AGENTCODEC_CODE_SANDBOX`` (default ``subprocess``, ``docker`` is
opt-in — README §Code scoring).

Run::

    python examples/16_code_scoring_end_to_end.py
"""
from __future__ import annotations

import logging

from agentcodec import ReliabilityModule
from agentcodec.code_exec import _selected_backend

from _common import (
    BASE_URL,
    API_KEY,
    JUDGE,
    MODEL_A,
    MODEL_B,
    critic_same,
    explain_score,
    judge_block,
    model_block,
)


# HumanEval/10 — find the shortest palindrome starting with a given string.
# Picked because it's non-trivial (slicing + reversing) but well within
# qwen3:8b's range, so we see a real pass/fail signal without forcing
# the model into a regime where every attempt fails.
PROMPT = """\
Implement the following Python function. Return only the function body
inside a fenced ```python code block — no commentary, no usage examples.

```python
def make_palindrome(s: str) -> str:
    \"\"\"
    Find the shortest palindrome that begins with the supplied string.
    Algorithm:
      - Find the longest postfix of `s` that is itself a palindrome.
      - Append the reverse of the prefix that comes before that postfix
        to the end of `s`.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    \"\"\"
```
"""

METADATA = {
    "source":      "humaneval",
    "entry_point": "make_palindrome",
    "test_code": (
        "def check(candidate):\n"
        "    assert candidate('') == ''\n"
        "    assert candidate('x') == 'x'\n"
        "    assert candidate('xyz') == 'xyzyx'\n"
        "    assert candidate('xyx') == 'xyx'\n"
        "    assert candidate('jerry') == 'jerryrrej'\n"
    ),
}


def build(technique: str) -> ReliabilityModule:
    """Two-model pool + dedicated judge from _common.py."""
    return ReliabilityModule.from_dict({
        "models": [
            model_block(MODEL_A, temperature=0.7),
            model_block(MODEL_B, temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {"type": "fixed", "technique": technique},
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


def _print_code_preview(text: str, indent: str = "    ") -> None:
    """Print the first ~10 lines of an LLM reply, indented for readability."""
    for line in text.strip().splitlines()[:12]:
        print(f"{indent}{line}")
    if len(text.strip().splitlines()) > 12:
        print(f"{indent}…")


def run_once(technique: str, label: str) -> None:
    print(f"\n=== {label} ({technique}) ===")
    mod = build(technique)
    try:
        result = mod.run(
            prompt=PROMPT,
            reference="",                # unused for code
            category="code",
            metadata=METADATA,            # source="humaneval" → score_mode="code"
            return_trace=True,
        )
    finally:
        mod.close()

    print(f"  cost (USD)     : ${result.cost_usd:.6f}")
    print(f"  latency_s      : {result.latency_s:.2f}")
    # Full breakdown: sandbox pass/fail blended with the judge checklist,
    # per-criterion, on the final answer (return_trace=True is set below).
    explain_score(result, score_mode="code")
    print("  LLM output (preview):")
    _print_code_preview(result.text or "")


def main() -> None:
    # Surface harq_ir's per-round refinement decisions at INFO so you can
    # see the score climb across rounds — and the [SCORE …] line that
    # shows the blended det + judge split per attempt.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Sandbox backend: {_selected_backend()}")
    print(f"Models         : {MODEL_A} + {MODEL_B}; judge = {JUDGE}")
    print(f"Endpoint       : {BASE_URL}")
    print()
    print(
        "Same prompt, two configurations: baseline (one shot) and harq_ir "
        "(iterative critic-and-refine). The sandbox grades each attempt; "
        "harq_ir uses the score to decide whether to keep refining."
    )

    run_once("baseline", label="Layer A — single call, scored once")
    run_once("harq_ir",  label="Layer B — iterative refinement; sandbox drives early-exit")

    print(
        "\nTakeaways:\n"
        "  * baseline final_quality reflects whether the LLM's first try "
        "passes every assert in the sandbox.\n"
        "  * harq_ir converges only when the sandbox score crosses the "
        "library's default quality_threshold (0.85). When the first attempt "
        "fails, watch the critic prompt the refiner with the specific "
        "failure mode and the next round usually patches it."
    )


if __name__ == "__main__":
    main()
