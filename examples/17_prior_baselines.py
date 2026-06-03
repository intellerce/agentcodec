"""
17 — Prior-method baselines, head-to-head on a verifiable task.

AgentCodec ships faithful reproductions of seven prior test-time
reliability methods so you can compare its communication-theoretic
techniques against the literature at matched inference budget. ``02``
already covers ``self_refine``; this example runs the other six on a
single numeric reasoning problem with a known answer:

  * ``self_consistency``      sample N, majority-vote      (Wang et al. 2023)
  * ``best_of_n``             sample N, judge picks best   (Cobbe et al. 2021)
  * ``weighted_bon``          judge-weighted cluster vote  (Snell et al. 2024)
  * ``cisc``                  confidence-weighted vote     (Taubenfeld et al. 2025)
  * ``mixture_of_agents``     layered agents + aggregator  (Wang et al. 2025)
  * ``chain_of_verification`` draft → verify → revise      (Dhuliawala et al. 2023)

Plus a ``baseline`` control (one call, no reliability) so you can see what
the extra inference budget buys.

Scoring: the prompt has a numeric answer (``reference="20"``), so we pass
``score_mode="numeric"``. Under the default ``score_strategy="blended"``,
``final_quality = 0.6 × numeric-match + 0.4 × judge`` — see the per-row
``quality`` line and ``scoring path`` note printed by ``print_result``.

Important nuance: run through the library, the vote-based baselines
(``self_consistency`` / ``weighted_bon`` / ``cisc``) aggregate over
*free-form* answers via an LLM voter, NOT by exact-match on extracted
numbers. The canonical paper-faithful exact-match path requires
constructing the baseline class directly with an ``answer_extractor`` (see
``agentcodec/techniques/baselines.py``); ``score_mode`` here only governs
how the *final* answer is graded, not how the technique votes internally.

Run:
    python examples/17_prior_baselines.py
"""
from __future__ import annotations

from typing import Any

from agentcodec import ReliabilityModule

from _common import explain_score, judge_block, model_block, print_result


# A small expected-value problem with an exact integer answer. The expected
# number of flips of a fair coin to see two heads in a row is 6 — but we use
# a slightly harder framing the model gets wrong often enough that the
# aggregation methods have something to fix.
PROMPT = (
    "You repeatedly roll a fair six-sided die and sum the results. "
    "What is the expected number of rolls until the running sum first "
    "reaches 20 or more? Round to the nearest whole number and give the "
    "final answer as a single integer."
)
REFERENCE = "6"          # E[rolls] ≈ 20 / 3.5 + boundary ≈ 6
CATEGORY = "reasoning"


def build(technique: str) -> ReliabilityModule:
    """Two-model pool + dedicated judge; only the technique changes.

    Mixture-of-agents and the vote-based baselines benefit from a diverse
    pool, so we configure two model families (see _common.py overrides).
    """
    return ReliabilityModule.from_dict({
        "models": [
            model_block("qwen3:8b", temperature=0.7),
            model_block("llama3.1:8b", temperature=0.7),
        ],
        "judge": judge_block(),
        "strategy": {"type": "fixed", "technique": technique},
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


TECHNIQUES: list[tuple[str, str]] = [
    ("baseline",              "control: one call, no reliability"),
    ("self_consistency",      "sample 5, majority vote"),
    ("best_of_n",             "sample 5, judge picks best"),
    ("weighted_bon",          "judge-weighted cluster vote"),
    ("cisc",                  "confidence-weighted vote (verbal 0-100)"),
    ("mixture_of_agents",     "layered proposers + final aggregator"),
    ("chain_of_verification", "draft → verification Qs → revise"),
]


def main() -> None:
    print(f"Prompt   : {PROMPT}")
    print(f"Reference: {REFERENCE}   (scored with score_mode='numeric')\n")

    rows: list[tuple[str, Any]] = []
    for technique, blurb in TECHNIQUES:
        mod = build(technique)
        with mod:
            result = mod.run(
                PROMPT,
                category=CATEGORY,
                reference=REFERENCE,
                score_mode="numeric",
                return_trace=True,
            )
        rows.append((technique, result))
        print_result(result, label=f"{technique} — {blurb}", score_mode="numeric")

    # Compact leaderboard sorted by quality, with cost so you can eyeball
    # the quality-per-dollar tradeoff the comm-theory techniques target.
    print("\n=== leaderboard (quality desc) ===")
    print(f"{'technique':<24s}  {'quality':>8s}  {'cost ($)':>10s}  {'calls':>6s}")
    print("-" * 56)
    ranked = sorted(rows, key=lambda kv: (kv[1].final_quality or 0.0), reverse=True)
    for technique, r in ranked:
        print(f"{technique:<24s}  {(r.final_quality or 0.0):>8.3f}  "
              f"${r.cost_usd:>9.6f}  {r.num_llm_calls:>6d}")

    # Detailed score for the winner — how the numeric-match + judge blend
    # produced its final_quality, criterion by criterion.
    best_name, best_result = ranked[0]
    print(f"\nWinner breakdown — {best_name}:")
    explain_score(best_result, score_mode="numeric")


if __name__ == "__main__":
    main()
