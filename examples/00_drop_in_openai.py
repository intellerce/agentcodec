"""
00 — Drop-in replacement for the ``openai`` SDK.

The fastest way to get reliability into existing OpenAI code: change
**one import**. Everything else — ``messages=[...]``, ``tools=[...]``,
``stream=True``, async / sync, ``client.chat.completions.create(...)``
— continues to work because the wrapper preserves the native shape.

This script walks the four layers of adoption:

  Layer 1 — zero config. Acts identically to ``openai.OpenAI``.
  Layer 2 — preset string. One kwarg, picks a reliability technique.
  Layer 3 — full ReliabilityModule, configured from YAML or a dict.
  Layer 4 — per-call reliability override (or bypass).

The defaults talk to the local Ollama OpenAI-compat endpoint configured
in ``_common.py``; override via the standard ``AGENTCODEC_EXAMPLE_*``
env vars or your ``.env``.

Run::

    python examples/00_drop_in_openai.py
"""
from __future__ import annotations

# The headline import. Same surface as the real openai SDK.
from agentcodec.openai import OpenAI

from _common import BASE_URL, API_KEY, JUDGE, MODEL_A, MODEL_B, load_showcase_task

# Pull the demo prompt from showcase_tasks.json. The library auto-infers
# `score_mode` from the task's `source` field where applicable, so the
# final quality blends a deterministic check with the judge's score.
try:
    TASK = load_showcase_task("gsm8k_hard_0094")
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
# format is exactly what ``score_mode="numeric"`` (auto-inferred from
# source="gsm8k") looks for when scoring, so a correctly-reasoned answer
# also scores cleanly on the deterministic half of the rubric.
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
    """No `reliability=` → identical to vanilla openai.OpenAI."""
    print("\n=== Layer 1 — passthrough (no reliability) ===")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL_A,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
    )
    # Native OpenAI shape: resp.choices[0].message.content, resp.usage, etc.
    print(f"  text   : {resp.choices[0].message.content[:120].strip()}…")
    print(f"  model  : {resp.model}")
    print(f"  tokens : in={resp.usage.prompt_tokens}, out={resp.usage.completion_tokens}")


def layer_2_preset() -> None:
    """``from_preset("diversity_sc", models=[A, B], judge=J)`` → multi-model
    selection combining (pick the best-judged branch — no synthesis, so the
    final text is whichever branch the judge scored highest).

    The "preset" pattern is still one helper call; we pass it a list of
    models so the demo actually uses MODEL_A + MODEL_B for spatial diversity
    rather than two temperature samples of one model.
    """
    print("\n=== Layer 2 — preset shortcut ===")
    from agentcodec import ReliabilityModule
    mod = ReliabilityModule.from_preset(
        "diversity_sc",
        models=[MODEL_A, MODEL_B],
        api_key=API_KEY, base_url=BASE_URL,
        judge=JUDGE,
    )
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, reliability=mod)
        resp = client.chat.completions.create(
            model=MODEL_A,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
        )
        # Same OpenAI shape — plus a `.reliability` escape hatch carrying
        # the full ReliabilityResult, including the trace.
        rel = resp.reliability
        print(f"  text       : {resp.choices[0].message.content[:120].strip()}…")
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
    """For power users: build a ReliabilityModule directly and inject it.

    Lets you configure judge / critic / score_strategy / routed strategies
    (SemKNN, ACM table) without leaving the OpenAI client shape. Returns the
    constructed module so the quality-breakdown helper below can reuse it.
    """
    print("\n=== Layer 3 — full ReliabilityModule ===")
    from agentcodec import ReliabilityModule
    mod = ReliabilityModule.from_dict({
        "models": [
            {"model": MODEL_A, "base_url": BASE_URL, "api_key": API_KEY,
             "temperature": 0.7},
            {"model": MODEL_B, "base_url": BASE_URL, "api_key": API_KEY,
             "temperature": 0.7},
        ],
        "judge": {"model": JUDGE, "base_url": BASE_URL, "api_key": API_KEY},
        "critic": {"same": True},
        "strategy": {"type": "fixed", "technique": "diversity_sc"},
    })
    client = OpenAI(
        api_key=API_KEY, base_url=BASE_URL, reliability=mod,
    )
    # `model=` is required by the OpenAI SDK signature (this is a drop-in
    # wrapper), but with a full ReliabilityModule injected it does NOT pick
    # the channel models — the module's own `models=[MODEL_A, MODEL_B]` above
    # drives the calls. Here `model` is inert: it's only stamped onto
    # `resp.model` as a label. (It *does* matter in passthrough / preset-string
    # layers, which is why every layer keeps the same call shape.)
    resp = client.chat.completions.create(
        model=MODEL_A,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
    )
    rel = resp.reliability
    print(f"  text       : {resp.choices[0].message.content[:120].strip()}…")
    print(f"  technique  : {rel.technique_used}")
    print(f"  cost (USD) : {rel.cost_usd:.6f}")
    print(f"  quality    : {_fmt_quality(rel.final_quality)}  (final score after combining)")
    # Trace comes from the same call thanks to the wrapper's auto-trace.
    show_quality_breakdown(
        rel.trace,
        label="Layer 3 (hand-built ReliabilityModule)",
    )
    return mod


def layer_4_per_call_override() -> None:
    """Per-call `reliability=` overrides the client default.

    Pass ``reliability=False`` for raw passthrough on one call. Pass a
    different preset, dict, or module to use a different technique just
    for one call.
    """
    print("\n=== Layer 4 — per-call override ===")
    client = OpenAI(
        api_key=API_KEY, base_url=BASE_URL,
        reliability="diversity_sc",           # default: diversity_sc for every call
    )
    # This one call bypasses reliability (raw passthrough).
    resp = client.chat.completions.create(
        model=MODEL_A,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
        reliability=False,
    )
    has_reliability = getattr(resp, "reliability", None) is not None
    print(f"  bypass call carries reliability attr: {has_reliability}")
    print("  (this call ran with reliability=False → no trace to inspect)")


def show_quality_breakdown(trace: dict, label: str = "") -> None:
    """Pretty-print the trace from a single reliability run.

    The wrapper auto-passes ``return_trace=True`` into ``ReliabilityModule.run``
    now, so ``resp.reliability.trace`` is populated on every call — no
    second dispatch needed. This function reads from that trace and prints:

    1. Top-level quality signals (final, best branch, diversity gain).
    2. Per-branch scores (one row per channel call).
    3. Per-judge-call checklist (which yes/no criteria the judge answered
       and how each one contributed to the weighted score).

    Final quality is the judge's verdict, built from a binary checklist:
    the judge answers ~15 weighted yes/no criteria, the weighted sum is
    normalized to [0, 1]. ``label`` tags the section header so multiple
    per-layer breakdowns in one script are easy to tell apart.
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
        # Sort by weight descending so the heaviest checks show first.
        ordered = sorted(breakdown.items(),
                         key=lambda kv: -weights.get(kv[0], 0.0))
        for name, ok in ordered:
            w = weights.get(name, 0.0)
            contrib = w if ok else 0.0
            mark = "pass" if ok else "FAIL"
            print(f"      [{mark}] {name:<42} "
                  f"weight={w:.2f}  contributes {contrib:.2f}")


def main() -> None:
    layer_1_passthrough()
    layer_2_preset()
    mod = layer_3_full_module()
    layer_4_per_call_override()
    mod.close()


if __name__ == "__main__":
    main()
