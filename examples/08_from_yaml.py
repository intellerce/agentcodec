"""
08 — Load a comprehensive config from a YAML file.

Demonstrates the production path: keep the config in a versioned YAML and
let `ReliabilityModule.from_yaml()` parse + validate it. See the companion
file ``example_config.yaml`` in this directory for an annotated walkthrough
of every public knob.

Two things are worth noting:

1. ``LibraryConfig`` is strict (``extra="forbid"``) — typos in field names
   raise at load time, not at first ``mod.run()``. Try renaming
   ``max_rounds`` to ``maxrounds`` in the YAML to see this in action.

2. At construction time the module logs a pricing-tier summary for every
   channel (which models fell back to OpenRouter, which used the table,
   which inferred from parameter count). That's the same summary the
   ``cost_source`` field on every result is computed from.

Run:
    python examples/08_from_yaml.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from agentcodec import ReliabilityModule

from _common import explain_score, print_result


CONFIG_FILE = Path(__file__).parent / "example_config.yaml"


def main() -> None:
    # INFO so the per-channel pricing summary and the [SCORE] judge log
    # lines are visible. Drop to WARNING for a quiet run.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    mod = ReliabilityModule.from_yaml(CONFIG_FILE)

    prompt = (
        "A factory has three machines: A makes 50%% of widgets with a 1%% "
        "defect rate, B makes 30%% with 2%%, C makes 20%% with 4%%. "
        "Given a defective widget, what's the probability it came from C? "
        "Show your steps and report 3 significant figures."
    )
    with mod:
        result = mod.run(prompt, category="reasoning", return_trace=True)
        print_result(result, label=f"loaded from {CONFIG_FILE.name}")
        # Detailed score: the judge checklist behind final_quality.
        explain_score(result)

        # The trace has the routing decision, every individual call, and
        # the cost-source caveats. Dump a quick summary so you can see
        # what production logging gets for free.
        print("\n  --- trace summary ---")
        print(f"  category    : {result.trace.get('category', {}).get('value')}")
        print(f"  routing     : {result.trace.get('routing', {}).get('chosen')}")
        print(f"  cost_source : {result.cost_source}  (worst tier across calls)")
        for caveat in result.cost_caveats():
            print(f"    caveat: {caveat}")


if __name__ == "__main__":
    main()
