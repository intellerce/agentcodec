"""
20 — The rest of the iterative family: Turbo and HARQ-CC.

Example 04 covered HARQ-IR (retry with a critic-produced *delta*). The
library has two more iterative techniques that refine a single answer over
rounds — this example runs them side by side with HARQ-IR as the control:

  * ``harq_ir`` — Hybrid ARQ, Incremental Redundancy. Each retry adds NEW
    targeted information (the critic's structured gap list). [control]

  * ``harq_cc`` — Hybrid ARQ, Chase Combining. On failure, retry and
    *combine* the attempts with equal weight, the way a receiver sums
    repeated transmissions to raise SNR — no new side-information per round,
    just more independent looks at the same prompt.

  * ``turbo`` — Iterative SISO decoding. Generator and critic exchange
    *extrinsic* information back and forth across iterations (each only
    passes along what the other didn't already know), converging like a
    turbo decoder's two component decoders.

All three are sequential (no parallel fan-out), so cost scales with the
number of rounds/iterations actually taken before the quality threshold is
met. Watch ``rounds`` and ``quality`` per technique.

Run:
    python examples/20_turbo_harqcc.py
"""
from __future__ import annotations

from typing import Any

from agentcodec import ReliabilityModule

from _common import critic_same, explain_score, judge_block, model_block, print_result


PROMPT = (
    "Write a clear, correct explanation of how the TCP three-way handshake "
    "establishes a connection, including what each segment's SYN/ACK flags "
    "and sequence numbers are for. Be precise."
)
CATEGORY = "reasoning"


def build(technique: str, params: dict[str, Any]) -> ReliabilityModule:
    return ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.7)],
        "judge": judge_block(),
        "critic": critic_same(),     # critic reuses the primary channel
        "strategy": {"type": "fixed", "technique": technique, "params": params},
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


# (technique, params, one-line description)
CONFIGS: list[tuple[str, dict[str, Any], str]] = [
    ("harq_ir", {"max_rounds": 3},     "control: retry with critic delta"),
    ("harq_cc", {"max_rounds": 3},     "retry + equal-weight combine"),
    ("turbo",   {"max_iterations": 3}, "generator/critic extrinsic exchange"),
]


def main() -> None:
    print(f"Prompt: {PROMPT}\n")
    rows = []
    for technique, params, blurb in CONFIGS:
        mod = build(technique, params)
        with mod:
            result = mod.run(PROMPT, category=CATEGORY, return_trace=True)
        rows.append((technique, result))
        print_result(result, label=f"{technique} — {blurb}")

    print("\n=== iterations vs. quality ===")
    print(f"{'technique':<10s}  {'quality':>8s}  {'rounds':>7s}  "
          f"{'cost ($)':>10s}  {'calls':>6s}")
    print("-" * 50)
    for technique, r in rows:
        print(f"{technique:<10s}  {(r.final_quality or 0.0):>8.3f}  "
              f"{r.rounds:>7d}  ${r.cost_usd:>9.6f}  {r.num_llm_calls:>6d}")

    # Detailed score for turbo, criterion by criterion.
    turbo_result = next((r for t, r in rows if t == "turbo"), rows[0][1])
    print("\nBreakdown — turbo:")
    explain_score(turbo_result)


if __name__ == "__main__":
    main()
