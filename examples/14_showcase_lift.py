"""
14 — Showcase: AgentCodec lift over baseline on a task set.

Runs the prompts in ``examples/showcase_tasks.json`` through four
configurations and prints a per-task table plus an aggregate summary:

  1. ``baseline``       — single LLM call + judge.
  2. fixed ``harq_ir``  — "always use harq_ir" deployment.
  3. SemKNN at λ=1      — balanced cost/quality.
  4. SemKNN at λ=40     — heavy cost-penalty operating point.

Each SemKNN column prints the technique it chose for that row. For
statistically-rigorous A/B comparisons over your own production
prompts, use ``agentcodec.Evaluator`` (README §Evaluation).

SemKNN talks to the public hosted backend by default; set
``AGENTCODEC_SEMKNN_SERVER_URL`` in ``.env`` to point at a self-hosted
instance.

Run:
    python examples/14_showcase_lift.py
"""
from __future__ import annotations

from typing import Any

from agentcodec import ReliabilityModule

from _common import (
    MODEL_A,
    MODEL_B,
    critic_same,
    judge_block,
    load_showcase_task,
    model_block,
)


# A small set of free-form reasoning / QA tasks (score_mode=None →
# pure judge). Pick from showcase_tasks.json; edit this list to swap
# in your own prompts.
TASK_IDS = [
    "qa_07",
    "hard_reason_07",
    "hard_qa_07",
    "hard_reason_02",
    "extreme_trick_02",
    "custom_5x5_latin_first_row",
]


def build_fixed(technique: str) -> ReliabilityModule:
    """Build a ReliabilityModule for one fixed technique.

    Two models give multi-channel techniques spatial diversity;
    single-channel techniques use only the first.
    """
    return ReliabilityModule.from_dict({
        "models": [
            model_block(MODEL_A, temperature=0.7),
            model_block(MODEL_B, temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {"type": "fixed", "technique": technique},
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


def build_semknn(lambda_: float = 1.0) -> ReliabilityModule:
    """Build a SemKNN-routed module.

    SemKNN picks the technique per-prompt by querying a remote q-matrix
    over a unit-norm BGE embedding of the prompt. Defaults to the public
    hosted backend (``https://agentcodec.intellerce.com``); set
    ``AGENTCODEC_SEMKNN_SERVER_URL`` to point at a self-hosted instance.

    ``lambda_`` slides the operating point on the quality/cost Pareto:
    ``0`` = pure quality, ``1.0`` = balanced, higher = cheaper picks.
    """
    return ReliabilityModule.from_dict({
        "models": [
            model_block(MODEL_A, temperature=0.7),
            model_block(MODEL_B, temperature=0.7),
        ],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {
            "type": "routed",
            "router": {"type": "semknn", "lambda": lambda_},
            # Pass per-technique knobs SemKNN's chosen technique will use.
            "dispatch": {"harq_ir": {"max_rounds": 4}},
        },
        "defaults": {"category": "auto", "on_error": "fallback_baseline"},
    })


def run_one(
    mod: ReliabilityModule, task: dict[str, Any], *, return_trace: bool = False,
) -> dict[str, Any]:
    """Run one task and pull the comparison fields.

    With ``return_trace=True`` the returned dict also carries ``chosen``,
    the technique the router picked for this prompt.
    """
    md = {"source": task["source"]} if task.get("source") else None
    temp = CATEGORY_TEMPERATURES.get(task["category"], 0.7)
    result = mod.run(
        prompt=task["prompt"],
        system=SYSTEM_PROMPT,
        reference=task["reference"],
        category=task["category"],
        metadata=md,
        temperature=temp,
        return_trace=return_trace,
    )
    out: dict[str, Any] = {
        "quality": result.final_quality if result.final_quality is not None else 0.0,
        "cost": result.cost_usd,
        "latency": result.latency_s,
    }
    if return_trace:
        # `trace["router"]["chosen"]` is the technique the router selected
        # for this prompt; populated for both fixed and routed strategies.
        router = result.trace.get("router") or {}
        extra = router.get("extra") or {}
        out["chosen"] = router.get("chosen") or "?"
        out["estimate"] = bool(extra.get("estimate"))
        out["match_quality"] = extra.get("match_quality")
        out["profile_used"] = extra.get("profile_used")
    return out


# Two SemKNN operating points: balanced (λ=1) and cost-biased (λ=40).
SEMKNN_LAMBDAS = [1.0, 40.0]


# Per-category sampling temperatures. Lower on tasks with single-token
# answers (qa, code) so all configs see the same noise budget per task.
CATEGORY_TEMPERATURES = {
    "qa":        0.2,
    "reasoning": 0.4,
    "code":      0.2,
    "creative":  0.8,
}


# Repeats per (task, config) cell. Median across runs is reported. Bump
# higher for tighter error bars at proportional runtime cost.
REPEATS = 3


# Single system prompt covering all task types in this showcase.
SYSTEM_PROMPT = (
    "You are a careful problem-solving assistant. Reason step by step "
    "before answering. End your reply with the final answer in the "
    "exact format the question requests:\n"
    "  • For multi-choice questions: a single letter (A / B / C / D) "
    "on the last line.\n"
    "  • For numeric questions: a line of the form `#### <number>` "
    "containing only the final number (no units, no commentary).\n"
    "  • For code questions: a single ```python``` block containing "
    "only the function implementation."
)


def _lam_label(lam: float) -> str:
    """Pretty-print a lambda value as ``λ=0`` (no trailing zero)."""
    return f"λ={int(lam) if lam == int(lam) else lam}"


def _median(xs: list[float]) -> float:
    """Median over `xs`; robust to a single judge-parse outlier."""
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_repeated(
    mod: ReliabilityModule,
    task: dict[str, Any],
    *,
    repeats: int = REPEATS,
    return_trace: bool = False,
) -> dict[str, Any]:
    """Run a task ``repeats`` times and return median quality + mean cost."""
    runs = [run_one(mod, task, return_trace=return_trace) for _ in range(repeats)]
    qualities = [r["quality"] for r in runs]
    out: dict[str, Any] = {
        "quality":   _median(qualities),
        "quality_min": min(qualities),
        "quality_max": max(qualities),
        "cost":      sum(r["cost"] for r in runs) / repeats,
        "latency":   sum(r["latency"] for r in runs) / repeats,
        "n_repeats": repeats,
    }
    if return_trace:
        # Routing is embedding-deterministic per prompt → same `chosen` and
        # `estimate` across the N rolls. Pick from the first roll.
        out["chosen"] = runs[0].get("chosen", "?")
        out["estimate"] = runs[0].get("estimate")
        out["match_quality"] = runs[0].get("match_quality")
        out["profile_used"] = runs[0].get("profile_used")
    return out


def main() -> None:
    print(
        "Building baseline + harq_ir + SemKNN modules "
        f"(λ ∈ {[int(x) if x == int(x) else x for x in SEMKNN_LAMBDAS]})…"
    )
    baseline_mod = build_fixed("baseline")
    harq_ir_mod = build_fixed("harq_ir")
    # One SemKNN module per λ value.
    semknn_mods: dict[float, ReliabilityModule] = {
        lam: build_semknn(lambda_=lam) for lam in SEMKNN_LAMBDAS
    }

    # Each λ column shows ``<technique-13ch> <q-4ch>`` = 18 chars.
    semknn_header = "  ".join(
        f"{_lam_label(lam) + ' (tech    med_q)':<19}" for lam in SEMKNN_LAMBDAS
    )
    header = (
        f"  {'task':<22}  {'cat':<10}  "
        f"{'base':<5}  {'harq':<5}  " + semknn_header
        + f"  [base spread over {REPEATS} runs]"
    )
    print()
    print(f"All numbers are MEDIAN quality over N={REPEATS} repeats per cell;")
    print(f"per-category temperatures: {CATEGORY_TEMPERATURES}.")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: list[dict[str, Any]] = []
    try:
        for tid in TASK_IDS:
            task = load_showcase_task(tid)
            temp = CATEGORY_TEMPERATURES.get(task["category"], 0.7)
            print(f"\n[{tid}] running {REPEATS}× per config "
                  f"(category={task['category']}, temp={temp})…", flush=True)

            b = run_repeated(baseline_mod, task)
            h = run_repeated(harq_ir_mod, task)
            # return_trace=True so we can read which technique SemKNN
            # chose for each λ (printed inline as ``<technique> <q>``).
            semknn_runs = {
                lam: run_repeated(semknn_mods[lam], task, return_trace=True)
                for lam in SEMKNN_LAMBDAS
            }

            rows.append({
                "task": tid, "category": task["category"],
                "baseline": b, "harq_ir": h,
                "semknn": semknn_runs,
            })

            semknn_cells = []
            for lam in SEMKNN_LAMBDAS:
                s = semknn_runs[lam]
                # Tag with "*" suffix when the backend flagged estimate mode.
                est = "*" if s.get("estimate") else " "
                tech = (s["chosen"] or "?")[:13]
                semknn_cells.append(f"{tech:<13}{est}{s['quality']:.2f}  ")
            print(
                f"  {tid:<22}  {task['category']:<10}  "
                f"{b['quality']:.2f}   {h['quality']:.2f}   "
                + "  ".join(semknn_cells)
                + f"  [base spread {b['quality_min']:.2f}–{b['quality_max']:.2f}]"
            )
    finally:
        baseline_mod.close()
        harq_ir_mod.close()
        for m in semknn_mods.values():
            m.close()

    if not rows:
        return

    print("  " + "-" * (len(header) - 2))
    n = len(rows)

    def avg(col: str, sub: str) -> float:
        return sum(r[col][sub] for r in rows) / n

    def avg_semknn(lam: float, sub: str) -> float:
        return sum(r["semknn"][lam][sub] for r in rows) / n

    aq_b, ac_b = avg("baseline", "quality"), avg("baseline", "cost")
    aq_h, ac_h = avg("harq_ir", "quality"), avg("harq_ir", "cost")

    def cost_x(num: float, den: float) -> str:
        return f"{num / den:.1f}×" if den > 0 else "n/a"

    print(f"\nAggregate over {n} task(s):")
    print(f"  baseline       : quality {aq_b:.3f}   cost ${ac_b:.4f}")
    print(f"  harq_ir        : quality {aq_h:.3f}   cost ${ac_h:.4f}   "
          f"(Δq {aq_h - aq_b:+.3f}, cost {cost_x(ac_h, ac_b)} baseline)")
    for lam in SEMKNN_LAMBDAS:
        aq, ac = avg_semknn(lam, "quality"), avg_semknn(lam, "cost")
        print(
            f"  semknn {_lam_label(lam):<7}: quality {aq:.3f}   "
            f"cost ${ac:.4f}   "
            f"(Δq {aq - aq_b:+.3f}, cost {cost_x(ac, ac_b)} baseline)"
        )

    # Show technique picks per λ — the headline diagnostic for the sweep.
    from collections import Counter
    print("\nTechnique picks per λ:")
    for lam in SEMKNN_LAMBDAS:
        picks = Counter(r["semknn"][lam]["chosen"] for r in rows)
        pick_str = ", ".join(f"{tech}×{n_}" for tech, n_ in picks.most_common())
        print(f"  {_lam_label(lam):<6} : {pick_str}")

    # The estimate flag is set by the backend's profile-matching layer,
    # which is independent of λ — so reading it from any one λ is enough.
    ref_lam = SEMKNN_LAMBDAS[0]
    n_estimate = sum(1 for r in rows if r["semknn"][ref_lam].get("estimate"))
    if n_estimate:
        sample = next(
            r["semknn"][ref_lam] for r in rows
            if r["semknn"][ref_lam].get("estimate")
        )
        print(
            f"\n  estimate flag : {n_estimate}/{n} routings flagged "
            f"(match_quality={sample.get('match_quality')}, "
            f"profile_used={sample.get('profile_used')!r})."
        )

    print(
        "\nReading the λ sweep:\n"
        "  • λ=1   balanced cost/quality.\n"
        "  • λ=40  heavy cost penalty — cheaper techniques get a large boost.\n"
        "\n"
        "  Same technique across both columns → the q-matrix already ranks\n"
        "  that technique top on quality alone. Diverging picks → the cost\n"
        "  penalty is moving the argmax."
    )

    print(
        "\nFor statistically-rigorous A/B over your own production prompts, "
        "use agentcodec.Evaluator (README §Evaluation)."
    )


if __name__ == "__main__":
    main()
