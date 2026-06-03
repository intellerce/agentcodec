"""
19 — Rateless (Fountain) and fixed-rate (FEC) redundancy.

Two communication-theoretic families that trade extra inference budget for
reliability, in opposite ways:

  * ``fountain`` — *rateless*. Keep drawing fresh samples and folding them
    in until the judge is satisfied (or ``max_samples`` is hit). You don't
    pick a redundancy level up front; the channel decides how much it needs
    per prompt. Easy prompts finish in 1-2 samples; hard ones draw more.

  * ``fec_0.75`` / ``fec_0.50`` / ``fec_0.33`` — *fixed-rate* forward error
    correction. The code rate is the fraction of "useful payload" vs.
    redundancy, so a *lower* rate spends *more* samples on redundancy:
        0.75 → light redundancy, cheapest
        0.50 → balanced
        0.33 → heavy redundancy, most robust + most expensive
    Unlike fountain, the budget is committed before you see the prompt.

The analogy: fountain is like a rateless erasure code (send until the
receiver acks); FEC is a fixed block code (decide the overhead in advance).

This runs all four on the same prompt and prints quality, cost, and the
number of LLM calls so you can see rateless adaptivity vs. fixed overhead.
Soft-output ``fountain_soft`` (logprob-weighted) is covered in example 21.

Run:
    python examples/19_fountain_fec.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import explain_score, judge_block, model_block, print_result


PROMPT = (
    "Explain why the sky appears blue during the day but red/orange at "
    "sunset. Cover the physical mechanism, not just the observation."
)
CATEGORY = "reasoning"


def build(technique: str) -> ReliabilityModule:
    return ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.7)],
        "judge": judge_block(),
        "strategy": {
            "type": "fixed",
            "technique": technique,
            # Cap fountain's draws so the demo stays quick; raise for harder
            # tasks where it needs more samples to satisfy the judge.
            "params": {"max_samples": 6} if technique == "fountain" else {},
        },
        "defaults": {"category": "auto"},
    })


TECHNIQUES = ["fountain", "fec_0.75", "fec_0.50", "fec_0.33"]


def main() -> None:
    print(f"Prompt: {PROMPT}\n")
    rows = []
    for technique in TECHNIQUES:
        mod = build(technique)
        with mod:
            result = mod.run(PROMPT, category=CATEGORY, return_trace=True)
        rows.append((technique, result))
        print_result(result, label=technique)

    print("\n=== redundancy vs. cost ===")
    print(f"{'technique':<12s}  {'quality':>8s}  {'cost ($)':>10s}  {'calls':>6s}")
    print("-" * 44)
    for technique, r in rows:
        print(f"{technique:<12s}  {(r.final_quality or 0.0):>8.3f}  "
              f"${r.cost_usd:>9.6f}  {r.num_llm_calls:>6d}")
    print(
        "\nNote: fountain's call count is *adaptive* (it stops when the judge "
        "is satisfied);\nthe fec_* call counts are fixed by their code rate "
        "regardless of prompt difficulty."
    )

    # Detailed score for the rateless run, criterion by criterion.
    print("\nBreakdown — fountain:")
    explain_score(rows[0][1])


if __name__ == "__main__":
    main()
