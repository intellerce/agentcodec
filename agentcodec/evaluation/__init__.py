"""
Evaluation framework for comparing deployment configurations.

This is *not* the paper benchmark. It's a different axis: take 2-N candidate
``LibraryConfig``\\s, run them on a shared evaluation prompt set, and tell
the user which one to deploy.

Use it to answer:
  - "Should I switch from Config A to Config B?" (paired Wilcoxon + BH)
  - "Which config is on the Pareto frontier of (quality, cost, latency)?"
  - "Is candidate Z significantly worse than prod by more than 2%?" (CI gating)

Example:

    from agentcodec import Evaluator

    ev = Evaluator(
        configs={
            "prod":        "configs/lib/fixed_harq.yaml",
            "candidate":   "configs/lib/routed_semknn.yaml",
        },
        prompts_file="my_eval.jsonl",
        repeats=3,
        parallel_prompts=4,
    )
    report = ev.run()
    report.summary()                  # console table + recommendation
    report.to_markdown("report.md")
    report.to_json("report.json")     # for CI integration
"""

from .evaluator import Evaluator
from .report import ConfigStats, EvalReport, PairwiseComparison

__all__ = [
    "ConfigStats",
    "EvalReport",
    "Evaluator",
    "PairwiseComparison",
]
