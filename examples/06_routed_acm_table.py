"""
06 — Routed strategy with an ACM (Adaptive Coding-Modulation) table.

Instead of pinning every prompt to the same technique, route by *estimated
difficulty*: easy prompts go through a cheap baseline, hard ones to a
heavy diversity pipeline. Difficulty is estimated by a one-shot probe
generation, scored by the judge, and then bucketed against the table below.

This is the simplest router — no training cache, no remote service.
For learned routing, see ``configs/lib/routed_acm_table.yaml`` and the
SemKNN README section. The trade-off you're picking here:

    + No training, no remote calls, totally deterministic.
    − You hand-author the buckets and have to revise them as models change.

Run:
    python examples/06_routed_acm_table.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import critic_same, explain_score, judge_block, model_block, print_result


def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [
            model_block("qwen3:8b", temperature=0.7),
            model_block("llama3.1:8b", temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {
            "type": "routed",
            "router": {
                "type": "acm_table",
                "table": [
                    # Easy → 1 baseline call. Cheapest path.
                    {"name": "easy",     "difficulty_range": [0.0, 0.3],
                     "technique": "baseline"},
                    # Moderate → a couple of HARQ-IR rounds.
                    {"name": "moderate", "difficulty_range": [0.3, 0.6],
                     "technique": "harq_ir",       "max_rounds": 2},
                    # Hard → more rounds.
                    {"name": "hard",     "difficulty_range": [0.6, 0.85],
                     "technique": "harq_ir",       "max_rounds": 4},
                    # Extreme → spend the budget on diversity.
                    {"name": "extreme",  "difficulty_range": [0.85, 1.0],
                     "technique": "diversity_mrc", "num_branches": 2},
                ],
            },
        },
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })

    prompts = [
        ("What's the capital of France?",                              "qa"),
        ("Sketch a proof of the Cauchy-Schwarz inequality in 4 lines.", "reasoning"),
    ]
    with mod:
        for prompt, cat in prompts:
            result = mod.run(prompt, category=cat, return_trace=True)
            chosen = result.trace.get("routing", {}).get("chosen", "?")
            print(f"\n>>> {prompt}")
            print(f"    router chose: {chosen}")
            print_result(result, label=f"routed → {chosen}")
            # Detailed score for whichever technique the router picked.
            explain_score(result)


if __name__ == "__main__":
    main()
