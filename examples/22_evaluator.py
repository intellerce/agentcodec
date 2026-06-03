"""
22 — Statistical A/B evaluation with the Evaluator.

Examples 07 and 14 eyeball technique differences on one or a few prompts.
That's fine for a sanity check, but judge scores are noisy, so a one-prompt
"X beat Y" can flip on the next prompt. ``agentcodec.Evaluator`` is the
rigorous path: run several configurations over a shared prompt set (with
repeats), then report per-config means with **bootstrap 95% confidence
intervals**, **paired Wilcoxon** significance tests with Benjamini-Hochberg
correction, and a **Pareto frontier** over quality / cost / latency.

This compares three configs on a shared QA set with one shared
``eval_judge``: a ``baseline`` that uses extended thinking, fixed ``harq_ir``
(iterative critic-and-refine), and ``diversity_mrc`` (multi-model ensembling).
The report shows per-config quality with confidence intervals and whether each
difference from the baseline is statistically significant — so you can pick a
deployment strategy on evidence rather than a single eyeballed run.

Notes:
  * Point ``Evaluator`` at *your* production prompts (add a ``reference`` and,
    where you have ground truth, a ``score_mode`` for deterministic grading)
    to measure the lift on your actual workload — that's the intended use.
  * Results are cached per-config as JSONL under ``cache_dir`` so an
    interrupted run resumes instead of re-querying. We use a fresh temp dir
    here; in production point it at a stable path to accumulate/resume.
  * Bump ``repeats`` and add prompts for tighter CIs (cost scales linearly:
    n_configs × n_prompts × repeats LLM pipelines).

Run:
    python examples/22_evaluator.py
"""
from __future__ import annotations

import tempfile

from agentcodec import Evaluator

from _common import judge_block, model_block


def cfg(strategy: dict, thinking: bool | str | dict | None = False) -> dict:
    """A LibraryConfig dict; the strategy block and ``thinking`` differ per config.

    ``thinking`` is passed straight through to the generator channels. The
    comparison this example sets up is **extended thinking (as a reliability
    method) vs. the AgentCodec techniques**:

      * ``baseline`` runs a single call with ``thinking=True`` — i.e. "just let
        the model reason." That's the prior-art test-time-compute baseline.
      * ``harq_ir`` / ``diversity_mrc`` run WITHOUT thinking, so the question
        the report answers is: do the communication-theoretic techniques beat
        plain reasoning at matched quality/cost?

    (``thinking=True`` is the Ollama/vLLM form for qwen3; for Anthropic/OpenAI
    o-series use a {"enabled": True, "budget_tokens": N} dict — see
    12_thinking_capture.py. llama3.1 isn't a reasoning model, so the flag is a
    no-op there, but ``baseline``/``harq_ir`` run on the primary qwen3 channel.)
    """
    # Reasoning models can be slow per call; give them a generous timeout so a
    # single slow generation doesn't error the run (default is 300s for Ollama).
    return {
        "models": [
            model_block("qwen3:8b", temperature=0.7, thinking=thinking,
                        request_timeout_s=600),
            model_block("llama3.1:8b", temperature=0.7, thinking=thinking,
                        request_timeout_s=600),
        ],
        "judge": judge_block(),
        "critic": {"same": True},
        "strategy": strategy,
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    }


CONFIGS = {
    # The comparator: plain model + extended thinking, no AgentCodec technique.
    "baseline":      cfg({"type": "fixed", "technique": "baseline"}, thinking=True),
    # The techniques, WITHOUT thinking — is the machinery worth more than reasoning?
    "harq_ir":       cfg({"type": "fixed", "technique": "harq_ir",
                          "params": {"max_rounds": 3}}),
    "diversity_mrc": cfg({"type": "fixed", "technique": "diversity_mrc"}),
}

# Each record needs at least `prompt`; `reference` + `category` enable
# reference-aware judge scoring. `id` is auto-filled if omitted. Swap in your
# own prompts — add a `score_mode` (e.g. "numeric"/"exact_match") on any that
# have a verifiable answer to grade them deterministically.
PROMPTS = [
    {"id": "q1", "category": "qa",
     "prompt": "What causes the seasons on Earth?",
     "reference": "The tilt of Earth's rotational axis (~23.5°) relative to "
                  "its orbital plane, which changes the angle and duration "
                  "of sunlight over the year — not the distance to the Sun."},
    {"id": "q3", "category": "reasoning",
     "prompt": "Explain why correlation does not imply causation, with an "
               "example.",
     "reference": "Two variables can move together due to a confounder, "
                  "reverse causation, or coincidence; e.g. ice-cream sales "
                  "and drownings both rise with summer heat."},
    # Harder, deterministically-graded items (single verifiable answer): a
    # single pass is error-prone, so harq_ir's verification and diversity_mrc's
    # cross-sample agreement have room to add reliability.
    {"id": "fixed_points", "category": "reasoning", "score_mode": "numeric",
     "prompt": "How many permutations of the numbers 1 through 8 leave exactly "
               "two of the numbers in their original position? Reason step by "
               "step, then give ONLY the final number on the last line.",
     "reference": "7420"},                       # C(8,2) · !6 = 28 · 265
    {"id": "zebra_grid", "category": "reasoning", "score_mode": "exact_match",
     "prompt": "Five houses in a row are numbered 1 to 5, left to right. Five "
               "people — Ann, Ben, Cara, Dan, and Eve — each live in a "
               "different house and each own a different pet (cat, dog, fish, "
               "bird, rabbit). Cara lives in house 1. Dan lives in house 5. "
               "Eve lives immediately to the right of Ann. Ann owns the fish. "
               "The rabbit is in house 3. The dog's house is immediately to "
               "the left of the bird's house. Who owns the dog? Reason step by "
               "step, then give ONLY the name on the last line.",
     "reference": "Ben"},                        # unique: Cara,Ann,Eve,Ben,Dan (1..5)
]


def _make_progress_printer() -> "object":
    """A progress_callback that renders a live bar + running per-config quality.

    Uses tqdm if installed; otherwise falls back to a plain inline bar so the
    example has no hard dependency.
    """
    from collections import defaultdict

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    bar = None
    try:
        from tqdm import tqdm
        bar = tqdm(total=0, unit="run", dynamic_ncols=True)
    except Exception:
        bar = None

    def on_progress(ev: dict) -> None:
        completed, total, cfg_name = ev["completed"], ev["total"], ev["config"]
        rec = ev["record"] or {}
        # Accumulate a running mean quality per config for intermediate results.
        if not ev["cached"] and rec.get("error") is None and rec.get("quality") is not None:
            sums[cfg_name] += float(rec["quality"])
            counts[cfg_name] += 1
        running = "  ".join(
            f"{n}={sums[n] / counts[n]:.3f}(n{counts[n]})"
            for n in counts
        ) or "—"
        err = "  ERROR" if rec.get("error") else ""
        if bar is not None:
            if bar.total != total:
                bar.total = total
                bar.refresh()
            bar.n = completed
            bar.set_description(f"eval [{cfg_name}]")
            bar.set_postfix_str(f"q: {running}{err}")
            bar.refresh()
        else:
            pct = 100.0 * completed / max(total, 1)
            filled = int(pct // 4)
            bar_str = "█" * filled + "·" * (25 - filled)
            print(f"\r[{bar_str}] {completed}/{total} ({pct:4.0f}%)  {running}{err}",
                  end="", flush=True)

    return on_progress


def main() -> None:
    progress = _make_progress_printer()
    with tempfile.TemporaryDirectory(prefix="agentcodec-eval-") as cache_dir:
        evaluator = Evaluator(
            configs=CONFIGS,
            prompts=PROMPTS,
            repeats=2,                  # 2 samples per (config, prompt) for CIs
            eval_judge=judge_block(),   # one shared judge across all configs
            cache_dir=cache_dir,
            on_error="continue",        # record failures instead of aborting
            max_retries=2,              # retry a failed run up to 2× (e.g. timeouts)
        )
        report = evaluator.run(
            baseline="baseline", alpha=0.05, progress_callback=progress,
        )
    print()  # finish the progress line

    # `.summary()` renders the per-config table (quality mean + 95% CI, cost,
    # latency), the paired significance tests vs. the baseline, the Pareto
    # frontier, and the recommendation.
    print(report.summary())

    # The structured fields are also available programmatically, e.g.:
    print("\nMachine-readable view:")
    for c in sorted(report.configs, key=lambda s: s.quality_mean, reverse=True):
        lo, hi = c.quality_ci95
        print(f"  {c.name:<16s} quality {c.quality_mean:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  ${c.cost_usd_mean:.6f}/run  n={c.n_runs}")


if __name__ == "__main__":
    main()
