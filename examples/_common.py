"""
Shared helpers for the example scripts.

The examples default to a local Ollama setup (matching ``configs/quick.yaml``)
because that's what runs without any cloud API key. Override via env:

    AGENTCODEC_EXAMPLE_BASE_URL    OpenAI-compatible endpoint
                                   (default: http://localhost:11434/v1)
    AGENTCODEC_EXAMPLE_API_KEY     API key for that endpoint
                                   (default: "ollama")
    AGENTCODEC_EXAMPLE_MODEL_A     primary generator model
                                   (default: qwen3:8b)
    AGENTCODEC_EXAMPLE_MODEL_B     secondary generator (used by diversity)
                                   (default: llama3.1:8b)
    AGENTCODEC_EXAMPLE_JUDGE       judge / quality scorer
                                   (default: gemma3:12b)

Quick switch to OpenAI:

    export AGENTCODEC_EXAMPLE_BASE_URL=https://api.openai.com/v1
    export AGENTCODEC_EXAMPLE_API_KEY=$OPENAI_API_KEY
    export AGENTCODEC_EXAMPLE_MODEL_A=gpt-4o-mini
    export AGENTCODEC_EXAMPLE_MODEL_B=gpt-4o
    export AGENTCODEC_EXAMPLE_JUDGE=gpt-4o-mini
"""
from __future__ import annotations

import os
from typing import Any

# The library itself does NOT auto-load `.env` (OpenAI SDK convention —
# embedders shouldn't have their process env silently mutated on import).
# The examples DO, because they're end-user entry points where the DX
# of "edit .env, then python examples/01_hello.py" matters. Shell
# exports still win over `.env`; opt out entirely with
# `AGENTCODEC_DISABLE_DOTENV=1`.
from agentcodec import load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


BASE_URL = env("AGENTCODEC_EXAMPLE_BASE_URL", "http://localhost:11434/v1")
API_KEY = env("AGENTCODEC_EXAMPLE_API_KEY", "ollama")
MODEL_A = env("AGENTCODEC_EXAMPLE_MODEL_A", "qwen3:8b")
MODEL_B = env("AGENTCODEC_EXAMPLE_MODEL_B", "llama3.1:8b")
JUDGE = env("AGENTCODEC_EXAMPLE_JUDGE", "gemma3:12b")


def model_block(
    model: str,
    temperature: float = 0.7,
    thinking: bool | str | dict[str, Any] | None = None,
    request_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Build a single ModelConfig dict matching the YAML schema.

    ``thinking`` is omitted by default (the library disables reasoning unless
    asked). Pass ``thinking=True`` for local Ollama/vLLM thinking models
    (e.g. qwen3), ``"auto"``, or a ``{"enabled": True, "budget_tokens": N}``
    dict for Anthropic / OpenAI o-series. See ``12_thinking_capture.py`` for
    the per-backend forms.

    ``request_timeout_s`` overrides the per-call HTTP timeout (default 300s for
    Ollama, 240s otherwise) — raise it for slow local / reasoning models.
    """
    block: dict[str, Any] = {
        "model": model,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "temperature": temperature,
    }
    if thinking is not None:
        block["thinking"] = thinking
    if request_timeout_s is not None:
        block["request_timeout_s"] = request_timeout_s
    return block


def judge_block() -> dict[str, Any]:
    return {
        "model": JUDGE,
        "base_url": BASE_URL,
        "api_key": API_KEY,
    }


def critic_same() -> dict[str, Any]:
    """Critic config that reuses the primary channel (recommended for HARQ/Turbo)."""
    return {"same": True}


def load_showcase_task(task_id: str) -> dict[str, Any]:
    """Load one task from ``examples/showcase_tasks.json``.

    The JSON ships next to this file and contains a small, curated set of
    hard tasks where the AgentCodec techniques featured in each example
    actually win. Each entry has ``id``, ``category``, ``prompt``,
    ``reference``, and a ``best_technique`` hint.

    Raises ``KeyError`` if the requested ``task_id`` is not in the file —
    the example scripts catch this and fall back to a generic prompt so
    the demo still runs.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "showcase_tasks.json"
    payload = json.loads(path.read_text())
    for item in payload.get("tasks", []):
        if item.get("id") == task_id:
            return item
    raise KeyError(f"Task {task_id!r} not found in {path}")


# --- Score explanation helpers -------------------------------------------
#
# `final_quality` is a number in [0, 1], but on its own it tells you nothing
# about *how* it was produced. These helpers turn it into something a reader
# can reason about. There are two layers of detail:
#
#   print_result(..., score_mode=...)  — one-line "scoring path" note
#                                        (lightweight; no return_trace needed)
#   explain_score(result, score_mode=) — full per-criterion judge checklist
#                                        (rich; needs return_trace=True)
#
# Both assume the library default `score_strategy="blended"`: a task with a
# deterministic `score_mode` blends that signal with the judge as
# `final_quality = 0.6 × deterministic + 0.4 × judge`; a task without one is
# scored purely by the judge's 15-criterion yes/no checklist.

_SCORE_MODE_DESC = {
    "code":            "sandbox execution (pass/fail asserts)",
    "code_complexity": "sandbox execution + empirical Big-O fit",
    "exact_match":     "exact normalized string match",
    "exact_letter":    "multiple-choice letter match (A/B/C/D)",
    "numeric":         "numeric-answer match",
    "relaxed":         "numeric match within tolerance",
    "yes_no":          "yes/no answer match",
}

# Techniques that fan out into independently-scored *parallel candidates*,
# for which "branch scores" and "combining gain" are meaningful. Sequential
# refine/iterate techniques (self_refine, harq_*, turbo, chain_of_verification)
# are deliberately excluded: their channel-role calls include unscored
# intermediate steps (e.g. the critique in self_refine, which defaults to
# quality_score=0.0), so reporting them as "branches" is misleading.
_PARALLEL_TECHNIQUES = frozenset({
    "diversity_sc", "diversity_mrc", "diversity_egc", "diversity_sc_N",
    "diversity_mrc_discrete_N", "diversity_spatial", "diversity_frequency",
    "diversity_time", "diversity_mrc_soft", "diversity_mrc_discrete_N_soft",
    "best_of_n", "weighted_bon", "self_consistency", "cisc",
    "mixture_of_agents", "fountain", "fountain_soft",
    "fec_0.75", "fec_0.50", "fec_0.33",
})


def _is_parallel(result: Any) -> bool:
    """True if the run's technique produces parallel, independently-scored
    branches (so per-branch scores / combining gain are meaningful)."""
    return (getattr(result, "technique_used", "") or "") in _PARALLEL_TECHNIQUES


def scoring_path_note(score_mode: str | None) -> str:
    """One-line description of how `final_quality` was computed.

    See the module-level note above for the blend formula. Pass the same
    `score_mode` you handed to `mod.run(...)`; `None` (or "judge") means the
    task was judge-only.
    """
    if not score_mode or score_mode == "judge":
        return "judge checklist (15 weighted yes/no criteria)"
    det = _SCORE_MODE_DESC.get(score_mode, score_mode)
    return f"blended: 0.6 × {det} + 0.4 × judge checklist"


def print_result(
    result: Any, label: str = "Result", score_mode: str | None = None,
) -> None:
    """Pretty-print the minimal-mode fields of a ReliabilityResult.

    Now always surfaces `final_quality` (the number every technique is
    optimizing) and a one-line explanation of how it was scored. Pass
    `score_mode` if the task used deterministic scoring so the note reflects
    the blend; omit it for judge-only tasks. For the full per-criterion
    breakdown, call `explain_score(result, score_mode=...)` on a result run
    with `return_trace=True`.
    """
    print(f"\n=== {label} ===")
    print(f"  technique  : {result.technique_used}")
    print(f"  cost (USD) : {result.cost_usd:.6f}  ({result.cost_source})")
    print(f"  latency_s  : {result.latency_s:.2f}")
    fq = getattr(result, "final_quality", None)
    if fq is not None:
        print(f"  quality    : {fq:.3f}  ({scoring_path_note(score_mode)})")
    # For combining techniques (diversity / MoA / fountain), show how much
    # the synthesizer added over the single best branch. Skipped for
    # sequential techniques, which have no parallel branches to combine.
    bi = getattr(result, "best_individual_quality", None)
    dg = getattr(result, "diversity_gain", None)
    if _is_parallel(result) and bi is not None and dg is not None:
        print(f"  combining  : best branch {bi:.3f} → final {fq:.3f}  (gain {dg:+.3f})")
    if result.error:
        print(f"  error      : {result.error}")
    # Truncate long answers so the terminal stays readable.
    text = (result.text or "").strip()
    preview = text if len(text) < 400 else text[:400] + " …"
    print(f"  text       : {preview}")


def explain_score(result: Any, score_mode: str | None = None) -> None:
    """Render a detailed, human-readable breakdown of `final_quality`.

    Shows the headline number, the scoring path, per-branch quality (for
    parallel techniques), and the judge's 15-criterion checklist for the
    final scored answer — which criteria passed, their weights, and the
    weighted sum. Requires the result to have been produced with
    `return_trace=True`; without a trace it prints just the headline and a
    hint.
    """
    fq = getattr(result, "final_quality", None)
    print("\n  ── score breakdown ──")
    print(f"  final_quality : {fq:.3f}" if fq is not None else "  final_quality : (none)")
    print(f"  scoring path  : {scoring_path_note(score_mode)}")
    parallel = _is_parallel(result)
    bi = getattr(result, "best_individual_quality", None)
    dg = getattr(result, "diversity_gain", None)
    # "best branch / combining gain" only makes sense when branches were
    # combined; for sequential techniques the answer evolves over rounds
    # instead (reported as `rounds` below).
    if parallel and bi is not None:
        gain = f"   combining gain: {dg:+.3f}" if dg is not None else ""
        print(f"  best branch   : {bi:.3f}{gain}")
    print(f"  rounds        : {getattr(result, 'rounds', 0)}   "
          f"llm_calls: {getattr(result, 'num_llm_calls', 0)}")

    trace = getattr(result, "trace", None) or {}
    calls = trace.get("calls") or []
    if not calls:
        print("  (run with return_trace=True to see the per-criterion judge checklist)")
        return

    # Per-branch quality — only for parallel techniques, where every
    # channel call is an independently-scored candidate answer. (Sequential
    # techniques interleave unscored critique/verification calls here, so
    # listing them as "branches" would be misleading.)
    if parallel:
        branch_scores = [
            c.get("quality_score") for c in calls
            if c.get("role") == "channel" and c.get("quality_score") is not None
        ]
        if len(branch_scores) > 1:
            print(f"  branch scores : {[round(s, 3) for s in branch_scores]}")

    # The final-answer checklist is the last judge call that carries one
    # (branches are scored first, the combined/final answer last).
    judge_checklists = [
        c["checklist"] for c in calls
        if c.get("role") == "judge" and c.get("checklist")
    ]
    if not judge_checklists:
        print("  (no judge checklist in trace — pure-deterministic "
              "score_strategy='exact', or a non-judge scoring path)")
        return
    cl = judge_checklists[-1]
    breakdown = cl.get("breakdown", {})
    weights = cl.get("weights", {})
    passed = cl.get("passed", sum(1 for v in breakdown.values() if v))
    total = cl.get("total", len(breakdown))
    print(f"  judge checklist: {passed}/{total} criteria passed  "
          f"(weighted = {cl.get('weighted_score', 0.0):.3f})")
    # Heaviest criteria first — that's where the score is won or lost.
    for name, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        mark = "✓" if breakdown.get(name) else "✗"
        print(f"      {mark} {name:<38s} {w:.2f}")
