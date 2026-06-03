"""
03 — Spatial diversity + MRC combining.

Two *different* models answer the prompt in parallel. The judge scores each,
and the synthesizer (the primary channel) folds them into a single
quality-weighted ("maximal-ratio combining") answer.

Key insight from the communications analogy: independent channels with
uncorrelated errors carry MORE information jointly than either alone.
Different model families (qwen + llama, gpt-4o + claude) tend to fail in
*different* ways, so a weighted blend beats either single answer.

Run:
    python examples/03_diversity_mrc.py

Knobs you can pass via strategy.params:
    num_branches : how many parallel generations (defaults to len(models))
    Combining strategy is fixed by the technique name:
      diversity_mrc — quality-weighted blend (recommended)
      diversity_sc  — selection combining (pick the best one)
      diversity_egc — equal-gain combining (unweighted)
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import explain_score, judge_block, model_block, print_result


def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [
            # Two different model families = spatial diversity.
            # Bump temperature so each branch isn't a near-copy of the other.
            model_block("qwen3:8b", temperature=0.7),
            model_block("llama3.1:8b", temperature=0.7),
        ],
        "judge": judge_block(),
        "strategy": {
            "type": "fixed",
            "technique": "diversity_mrc",
        },
        "defaults": {"category": "auto"},
    })

    prompt = (
        "Explain in 3 short paragraphs why HTTP/3 chose QUIC over TCP, "
        "with one concrete trade-off in each paragraph."
    )
    with mod:
        result = mod.run(prompt, category="qa", return_trace=True)
        print_result(result, label="diversity_mrc (qwen3:8b + llama3.1:8b)")
        # Detailed score: per-branch quality, the combining gain, and the
        # judge checklist for the synthesized (combined) answer.
        explain_score(result)

        # Drill into the trace to show per-branch metadata. This is the same
        # dict that's logged to telemetry and persisted in benchmark caches.
        per_call = result.trace.get("per_call", [])
        print(f"\n  branches: {len(per_call)}")
        for i, call in enumerate(per_call, 1):
            print(
                f"    [{i}] model={call['model']:<22s}  "
                f"q={call.get('quality_score'):.3f}  "
                f"cost=${call['cost_usd']:.6f}"
            )


if __name__ == "__main__":
    main()
