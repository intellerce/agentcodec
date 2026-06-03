"""
07 — Side-by-side technique comparison on one prompt.

Builds a fresh ``ReliabilityModule`` per technique, runs the same prompt
through each, and prints a one-line summary so you can eyeball the
cost-vs-quality tradeoff before committing to a deployment strategy.

This is *not* a benchmark — for that, use ``run_benchmark.py`` which
amortizes judge variance across many prompts and reports CIs. This script
is a quick "what does my prompt actually look like under each technique".

Run:
    python examples/07_compare_techniques.py
"""
from __future__ import annotations

from typing import Any

from agentcodec import ReliabilityModule

from _common import critic_same, judge_block, model_block


PROMPT = (
    "An unbiased coin is flipped until two heads in a row appear. "
    "What is the expected number of flips? Show your reasoning briefly."
)
CATEGORY = "reasoning"


def build(strategy: dict[str, Any]) -> ReliabilityModule:
    """Common config — only the strategy block changes between runs."""
    return ReliabilityModule.from_dict({
        "models": [
            model_block("llama3.1:8b", temperature=0.7),
            model_block("qwen3:8b", temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": strategy,
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


STRATEGIES: list[tuple[str, dict[str, Any]]] = [
    ("baseline",      {"type": "fixed", "technique": "baseline"}),
    ("self_refine",   {"type": "fixed", "technique": "self_refine",
                       "params": {"max_rounds": 3}}),
    ("diversity_mrc", {"type": "fixed", "technique": "diversity_mrc"}),
    ("harq_ir",       {"type": "fixed", "technique": "harq_ir",
                       "params": {"max_rounds": 3}}),
    ("turbo",         {"type": "fixed", "technique": "turbo",
                       "params": {"max_iterations": 3}}),
]


def main() -> None:
    print(f"Prompt: {PROMPT}\n")
    print(f"{'technique':<16s}  {'cost ($)':>10s}  {'latency':>8s}  {'q_judge':>8s}  preview")
    print("-" * 90)

    for name, strategy in STRATEGIES:
        mod = build(strategy)
        with mod:
            result = mod.run(PROMPT, category=CATEGORY, return_trace=True)

        # Reach into the trace for the final judge score, when available.
        q = result.trace.get("final_quality")
        q_str = f"{q:.3f}" if isinstance(q, (int, float)) else "—"
        preview = (result.text or "").strip().splitlines()[0][:60]

        print(
            f"{name:<16s}  ${result.cost_usd:>9.6f}  "
            f"{result.latency_s:>7.2f}s  {q_str:>8s}  {preview}"
        )


if __name__ == "__main__":
    main()
