"""
00 — Drop-in replacement for the ``anthropic`` SDK.

Change one import:

    - from anthropic import Anthropic
    + from agentcodec.anthropic import Anthropic

The wrapper preserves the native Anthropic shape: ``client.messages.create(
model, system, messages, max_tokens, …)`` returns a ``Message`` with
``content`` blocks, ``stop_reason``, and ``usage.input_tokens`` /
``usage.output_tokens``. With ``reliability=`` set, the call goes through
a :class:`~agentcodec.ReliabilityModule`; the response shape is unchanged
plus a ``.reliability`` escape hatch.

Requires the ``anthropic`` SDK on the passthrough path:
``pip install agentcodec[anthropic]`` (or ``pip install anthropic``).
Reliability presets work even without it because they construct an
OpenAI-compatible channel internally.

Run::

    ANTHROPIC_API_KEY=... python examples/00_drop_in_anthropic.py
"""
from __future__ import annotations

import os

from agentcodec.anthropic import Anthropic

from _common import load_showcase_task

# Three distinct Anthropic models so Layers 2/3 demonstrate real spatial
# diversity (two different generators) plus a cheaper, separate judge.
# Override via env vars if you want different Claude variants.
MODEL_A = os.environ.get("AGENTCODEC_EXAMPLE_ANTHROPIC_MODEL", "claude-sonnet-4-5")
MODEL_B = os.environ.get("AGENTCODEC_EXAMPLE_ANTHROPIC_MODEL_B", "claude-haiku-4-5")
JUDGE = os.environ.get("AGENTCODEC_EXAMPLE_ANTHROPIC_JUDGE", "claude-haiku-4-5")
# Layer-1 passthrough is a single vanilla SDK call → only needs one model.
MODEL = MODEL_A

# Pull the demo prompt from showcase_tasks.json. The library auto-infers
# `score_mode` from the task's `source` field where applicable, so the
# final quality blends a deterministic check with the judge's score.
try:
    TASK = load_showcase_task("gsm8k_hard_0021")
    PROMPT = TASK["prompt"]
    REFERENCE: str | None = TASK["reference"]
    CATEGORY: str | None = TASK["category"]
    SOURCE: str | None = TASK.get("source")
except (KeyError, FileNotFoundError) as e:
    print(f"[showcase task not found — falling back to a generic prompt: {e}]")
    PROMPT = "In one sentence, what is QUIC?"
    REFERENCE = None
    CATEGORY = None
    SOURCE = None

# The demo prompt is a numeric word problem. The system prompt asks for
# step-by-step reasoning and a clean ``#### <answer>`` final line — that
# format matches what ``score_mode="numeric"`` (auto-inferred from
# source="gsm8k") looks for when scoring.
SYSTEM_PROMPT = (
    "You are a careful problem-solving assistant. Read the problem, "
    "work through the solution step by step, then write a final line "
    "in the form '#### <answer>' containing only the final number "
    "(no units, no extra commentary)."
)


def _fmt_quality(q: float | None) -> str:
    """Quality is None when the technique has no judge (e.g. pure passthrough)."""
    return f"{q:.3f}" if q is not None else "n/a"


def layer_1_passthrough() -> None:
    """No `reliability=` → identical to vanilla anthropic.Anthropic."""
    print("\n=== Layer 1 — passthrough ===")
    client = Anthropic()       # picks up ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=512,
    )
    # Native Anthropic shape: response.content is a list of typed blocks.
    text = resp.content[0].text if resp.content else ""
    print(f"  text         : {text[:120].strip()}…")
    print(f"  stop_reason  : {resp.stop_reason}")
    print(f"  input_tokens : {resp.usage.input_tokens}")
    print(f"  output_tokens: {resp.usage.output_tokens}")


def layer_2_preset() -> None:
    """``from_preset("diversity_sc", models=[A, B], judge=J)`` → multi-model
    selection combining over Anthropic (pick the best-judged branch; no
    synthesis, so the final text is whichever branch the judge scored
    highest).
    """
    print("\n=== Layer 2 — preset shortcut ===")
    from agentcodec import ReliabilityModule
    mod = ReliabilityModule.from_preset(
        "diversity_sc",
        models=[MODEL_A, MODEL_B],
        judge=JUDGE,
    )
    try:
        client = Anthropic(reliability=mod)
        resp = client.messages.create(
            model=MODEL_A,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=512,
        )
        text = resp.content[0].text if resp.content else ""
        rel = resp.reliability
        print(f"  text       : {text[:120].strip()}…")
        print(f"  technique  : {rel.technique_used}")
        print(f"  cost (USD) : {rel.cost_usd:.6f}")
        print(f"  latency_s  : {rel.latency_s:.2f}")
        print(f"  quality    : {_fmt_quality(rel.final_quality)}  (final score after combining)")
        # The wrapper now forces `return_trace=True` on every dispatch, so
        # the breakdown reads from the SAME run — no second dispatch.
        show_quality_breakdown(
            rel.trace,
            label='Layer 2 (from_preset "diversity_sc")',
        )
    finally:
        mod.close()


def layer_3_full_module():
    """Power-user path: ReliabilityModule with custom judge / critic.

    Returns the constructed module so the quality-breakdown helper below
    can reuse it without rebuilding.
    """
    print("\n=== Layer 3 — full ReliabilityModule ===")
    from agentcodec import ReliabilityModule
    mod = ReliabilityModule.from_dict({
        "models": [
            {"model": MODEL_A, "temperature": 0.7},
            {"model": MODEL_B, "temperature": 0.7},
        ],
        "judge": {"model": JUDGE},
        "critic": {"same": True},
        "strategy": {"type": "fixed", "technique": "diversity_sc"},
    })
    client = Anthropic(reliability=mod)
    resp = client.messages.create(
        model=MODEL_A,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=512,
    )
    text = resp.content[0].text if resp.content else ""
    rel = resp.reliability
    print(f"  text       : {text[:120].strip()}…")
    print(f"  technique  : {rel.technique_used}")
    print(f"  cost (USD) : {rel.cost_usd:.6f}")
    print(f"  quality    : {_fmt_quality(rel.final_quality)}  (final score after combining)")
    # Trace comes from the same call thanks to the wrapper's auto-trace.
    show_quality_breakdown(
        rel.trace,
        label="Layer 3 (hand-built ReliabilityModule)",
    )
    return mod


def show_quality_breakdown(trace: dict, label: str = "") -> None:
    """Pretty-print the trace from a single reliability run.

    The wrapper auto-passes ``return_trace=True`` into ``ReliabilityModule.run``
    now, so ``resp.reliability.trace`` is populated on every call — no
    second dispatch needed. This function reads from that trace and prints
    final/per-branch quality and the per-judge-call checklist.
    """
    suffix = f" — {label}" if label else ""
    print(f"\n=== Quality score breakdown{suffix} ===")
    if not trace:
        print("  (no trace on this response — wrapper skipped return_trace=True)")
        return
    print(f"  final_quality          : {_fmt_quality(trace.get('final_quality'))}")
    print(f"  best_individual_quality: {_fmt_quality(trace.get('best_individual_quality'))}")
    print(f"  diversity_gain         : {_fmt_quality(trace.get('diversity_gain'))}  "
          f"(final - best_individual)")

    calls = trace.get("calls", [])
    branch_calls = [c for c in calls if c.get("role") == "channel"]
    if branch_calls:
        print("  per-branch scores (pre-combine):")
        for i, c in enumerate(branch_calls, 1):
            print(f"    branch {i} model={c.get('model')!r:<24} "
                  f"quality={_fmt_quality(c.get('quality_score'))}")

    judge_calls = [c for c in calls if c.get("role") == "judge"]
    if not judge_calls:
        print("  (no judge calls in trace — the technique runs without a "
              "judge, or score_strategy='exact' short-circuited it)")
        return
    with_checklist = [c for c in judge_calls if c.get("checklist")]
    if not with_checklist:
        # Judge ran (that's where the quality score came from), but the
        # judge's reply wasn't parseable as the 15-key JSON checklist, so
        # QualityScorer fell back to single-number parsing. Common with
        # small local models that return prose like "Score: 0.7" instead
        # of strict JSON. The quality value is still real — it just came
        # from a single-number parse rather than a weighted checklist.
        print(f"  {len(judge_calls)} judge call(s) ran, but none returned a "
              f"parseable 15-criterion checklist — QualityScorer fell back "
              f"to single-number parsing. Use a stronger / JSON-mode judge "
              f"for the structured breakdown.")
        for i, c in enumerate(judge_calls, 1):
            preview = (c.get("text_preview") or "").replace("\n", " ")[:80]
            print(f"    Judge call {i}: model={c.get('model')}, "
                  f"cost=${c.get('cost_usd', 0):.6f}")
            print(f"      reply preview: {preview!r}…")
        return
    for i, c in enumerate(with_checklist, 1):
        cl = c["checklist"]
        breakdown = cl["breakdown"]
        weights = cl["weights"]
        passed = cl["passed"]
        total = cl["total"]
        print(f"\n  Judge call {i} — model={c.get('model')}, "
              f"cost=${c.get('cost_usd', 0):.6f}")
        print(f"    {passed}/{total} criteria passed → "
              f"weighted score {cl['weighted_score']:.3f}")
        ordered = sorted(breakdown.items(),
                         key=lambda kv: -weights.get(kv[0], 0.0))
        for name, ok in ordered:
            w = weights.get(name, 0.0)
            contrib = w if ok else 0.0
            mark = "pass" if ok else "FAIL"
            print(f"      [{mark}] {name:<42} "
                  f"weight={w:.2f}  contributes {contrib:.2f}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example.")
        return
    layer_1_passthrough()
    layer_2_preset()
    mod = layer_3_full_module()
    mod.close()


if __name__ == "__main__":
    main()
