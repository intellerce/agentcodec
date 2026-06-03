"""
21 — The full diversity-combining family.

Example 03 showed ``diversity_mrc`` (quality-weighted blend of parallel
branches). Diversity combining is the library's largest family — it borrows
the receiver-side combining schemes from wireless, where independent signal
copies with uncorrelated errors are merged into one stronger estimate.

Combining rules (how the branches are merged):
  * ``diversity_sc``   — Selection Combining: keep the single best branch.
  * ``diversity_egc``  — Equal-Gain Combining: equal-weight consensus.
  * ``diversity_mrc``  — Maximal-Ratio Combining: quality-weighted blend.

Diversity *axes* (what makes the branches independent):
  * ``diversity_spatial``   — different model families (qwen + llama).
  * ``diversity_frequency`` — prompt-phrasing variants of one model.
  * ``diversity_time``      — temperature spread on one model.

Wider pools (N samples per channel, then discrete cluster-vote):
  * ``diversity_sc_N``            — SC over N samples per channel.
  * ``diversity_mrc_discrete_N``  — cluster-vote MRC over N samples.

Soft-output variants (logprob-weighted instead of judge-weighted):
  * ``diversity_mrc_soft`` / ``diversity_mrc_discrete_N_soft`` — weight each
    branch by the model's own token confidence. These need a backend that
    exposes logprobs (Ollama / vLLM / OpenAI); on a backend without them
    the run is skipped with a note.

For each technique we print quality / cost / calls and the combining gain
(final − best single branch). At the end we show the full per-criterion
judge checklist for one representative run via ``explain_score``.

Run:
    python examples/21_diversity_family.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule

from _common import MODEL_A, MODEL_B, explain_score, judge_block, model_block


PROMPT = (
    "What are the main trade-offs between optimistic and pessimistic "
    "concurrency control in databases? Give a concrete scenario favouring "
    "each."
)
CATEGORY = "reasoning"


def build(technique: str) -> ReliabilityModule:
    # Two model families so spatial diversity has something to work with.
    return ReliabilityModule.from_dict({
        "models": [
            model_block(MODEL_A, temperature=0.7),
            model_block(MODEL_B, temperature=0.7),
        ],
        "judge": judge_block(),
        "strategy": {"type": "fixed", "technique": technique},
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


HARD = [
    "diversity_sc", "diversity_egc", "diversity_mrc",
    "diversity_spatial", "diversity_frequency", "diversity_time",
    "diversity_sc_N", "diversity_mrc_discrete_N",
]
SOFT = ["diversity_mrc_soft", "diversity_mrc_discrete_N_soft"]


def run_technique(technique: str):
    """Run one technique; return its result or None if the backend can't."""
    mod = build(technique)
    try:
        with mod:
            return mod.run(PROMPT, category=CATEGORY, return_trace=True)
    except Exception as e:  # e.g. soft variant on a logprob-less backend
        print(f"{technique:<30s}  skipped: {type(e).__name__}: {e}")
        return None


def main() -> None:
    print(f"Prompt: {PROMPT}\n")
    print(f"{'technique':<30s}  {'quality':>8s}  {'best':>6s}  "
          f"{'gain':>6s}  {'cost ($)':>10s}  {'calls':>6s}")
    print("-" * 76)

    representative = None
    for technique in HARD + SOFT:
        r = run_technique(technique)
        if r is None:
            continue
        if representative is None and technique == "diversity_mrc":
            representative = r
        bi = r.best_individual_quality
        dg = r.diversity_gain
        print(
            f"{technique:<30s}  {(r.final_quality or 0.0):>8.3f}  "
            f"{(bi if bi is not None else 0.0):>6.3f}  "
            f"{(dg if dg is not None else 0.0):>+6.3f}  "
            f"${r.cost_usd:>9.6f}  {r.num_llm_calls:>6d}"
        )

    # Detailed score breakdown for one representative run (MRC).
    if representative is not None:
        print("\nRepresentative breakdown — diversity_mrc:")
        explain_score(representative)


if __name__ == "__main__":
    main()
