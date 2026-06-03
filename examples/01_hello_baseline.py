"""
01 — Hello, baseline.

The smallest possible AgentCodec program. One model, no reliability tricks
(``technique="baseline"``). Useful as a control to compare every other
example against. The same ``ReliabilityResult`` shape is returned for every
technique, so once you're familiar with the output of this script the rest
of the examples will be easy to read.

Run:
    python examples/01_hello_baseline.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import explain_score, judge_block, model_block, print_result


def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.7)],
        "judge": judge_block(),
        "strategy": {
            "type": "fixed",
            "technique": "baseline",
        },
        # Auto-classify task category from the prompt (qa / reasoning / code / …).
        "defaults": {"category": "auto"},
    })

    with mod:
        result = mod.run(
            "In one sentence, what does TCP's three-way handshake achieve?",
            category="qa",
            return_trace=True,   # lets explain_score show the judge checklist
        )
        print_result(result, label="baseline")
        # Detailed score: which of the judge's 15 weighted yes/no criteria
        # this single answer passed, and the resulting weighted score.
        explain_score(result)


if __name__ == "__main__":
    main()
