"""
04 — HARQ-IR (Hybrid ARQ, Incremental Redundancy).

Generate → judge → if quality is too low, the critic produces a structured
*delta* (what's wrong + how to fix) → re-generate with that delta appended.
Repeat until quality crosses the threshold or ``max_rounds`` is hit.

The critic is what makes HARQ-IR different from naive self-refine: the
next-round prompt carries *concrete, judge-verified gaps*, not just "try
harder." That's the "incremental redundancy" in HARQ-IR — each round adds
fresh, targeted information to the channel.

Run:
    python examples/04_harq_ir.py

Common knobs (strategy.params):
    max_rounds       : cap on re-generation rounds (default 5)
    quality_threshold: target judge score for early exit (default 0.85)
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import critic_same, explain_score, judge_block, model_block, print_result


def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.7)],
        "judge": judge_block(),
        # Critic shares the primary channel — recommended for HARQ-IR because
        # the critic needs to think *like* the generator to produce useful deltas.
        # Set `same: false` and supply `model: ...` to use a heavier critic.
        "critic": critic_same(),
        "strategy": {
            "type": "fixed",
            "technique": "harq_ir",
            "params": {"max_rounds": 4},
        },
        "defaults": {
            "category": "auto",
            "early_exit": True,   # stop the moment quality clears the threshold
            "on_error": "fallback_baseline",
        },
    })

    prompt = (
        "Prove that the sum of the first n odd positive integers is n^2. "
        "Use a one-line algebraic argument AND a one-line picture-style "
        "argument; flag explicitly if either step uses a hidden assumption."
    )
    with mod:
        result = mod.run(prompt, category="reasoning", return_trace=True)
        print_result(result, label="harq_ir (max_rounds=4)")
        # Detailed score: the judge checklist for the final answer after the
        # critic-driven refinement rounds.
        explain_score(result)

        # Show how many rounds were actually used.
        rounds = result.trace.get("rounds")
        if rounds is not None:
            print(f"\n  rounds_used : {rounds}")


if __name__ == "__main__":
    main()
