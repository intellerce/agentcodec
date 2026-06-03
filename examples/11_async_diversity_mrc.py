"""
11 — Async streaming with diversity_mrc.

Diversity techniques fan out N branches in parallel via ``asyncio.gather``.
With multiple configured channels you'll see real concurrent execution —
the wall-clock latency is the slowest branch, not the sum.

The synthesizer (judge model by default) merges the branches into the
final answer. For now the synthesizer call lands as a single
``role="synthesis"`` ``TokenEvent`` rather than per-token deltas — full
synth streaming is on the v0.5 roadmap.

Run:
    python examples/11_async_diversity_mrc.py
"""
from __future__ import annotations

import asyncio
import time

from agentcodec import (
    FinalEvent,
    ProgressEvent,
    ReliabilityModule,
    TokenEvent,
)

from _common import critic_same, judge_block, model_block, print_result


async def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [
            model_block("qwen3:8b", temperature=0.7),
            model_block("llama3.1:8b", temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {"type": "fixed", "technique": "diversity_mrc"},
        "defaults": {"category": "auto"},
    })

    prompt = "Compare TCP slow start and BBR in two sentences."
    wall_t0 = time.time()
    branch_starts: dict[int, float] = {}
    branch_completes: dict[int, float] = {}
    synthesis_chars = 0
    final_result = None

    print(f"Streaming diversity_mrc for: {prompt!r}\n")
    with mod:
        async for ev in mod.astream(prompt):
            if isinstance(ev, ProgressEvent):
                if ev.stage == "branches_start":
                    print(f"  [+{ev.elapsed_s:5.2f}s] launching "
                          f"{ev.detail['n_branches']} branches concurrently")
                elif ev.stage == "branch_complete":
                    idx = ev.detail["branch"]
                    branch_completes[idx] = ev.elapsed_s
                    print(f"  [+{ev.elapsed_s:5.2f}s] branch {idx} "
                          f"({ev.detail['model']}) done")
                elif ev.stage == "branches_scored":
                    print(f"  [+{ev.elapsed_s:5.2f}s] all branches scored: "
                          f"{[f'{s:.2f}' for s in ev.detail['scores']]}")
            elif isinstance(ev, TokenEvent) and ev.role == "synthesis":
                synthesis_chars += len(ev.text)
                # The synthesizer single-emits the whole combined text today.
                print(f"  [+{time.time() - wall_t0:5.2f}s] synthesis: "
                      f"{ev.text[:120]}{'…' if len(ev.text) > 120 else ''}")
            elif isinstance(ev, TokenEvent) and ev.role == "answer":
                # SC-fallback path: best branch text is emitted as 'answer'
                # instead of synthesis.
                print(f"  [+{time.time() - wall_t0:5.2f}s] best branch chosen "
                      f"(SC fallback): {ev.text[:120]}"
                      f"{'…' if len(ev.text) > 120 else ''}")
            elif isinstance(ev, FinalEvent):
                final_result = ev.result

    if final_result is not None:
        print_result(final_result, label="diversity_mrc via astream")
        # Demonstrate concurrent execution: branch latencies overlap.
        if len(branch_completes) >= 2:
            sum_indep = sum(branch_completes.values())
            wall = max(branch_completes.values())
            print(f"  concurrent speedup: ~{sum_indep / wall:.1f}x "
                  f"(sum-of-branches / max-branch)")


if __name__ == "__main__":
    asyncio.run(main())
