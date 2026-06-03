"""
09 — Routed strategy via the remote SemKNN service.

SemKNN ("Semantic K-Nearest-Neighbour, cost-aware") picks the best technique
per-prompt based on a learned model of which technique works best for which
kind of prompt. The trained q-matrix lives on a backend service; your prompt
never leaves the client — only a unit-norm BGE embedding does.

The single knob ``lambda`` slides the operating point on the quality / cost
Pareto frontier:

    lambda = 0   pure quality, ignore cost
    lambda = 1   balanced
    lambda = 5   ~10% cheaper picks
    lambda = 10  ~30% cheaper
    lambda = 20  ~45% cheaper

This example posts a few prompts and prints the recommended technique +
match quality so you can see when the backend hits an "exact" profile
match (your channel pool matches a trained lineup) vs. an estimate
(the backend falls back to a partial match and flags the response).

Where it points
---------------
By default the client talks to the public hosted backend at
``https://agentcodec.intellerce.com`` (set by the library). Override
without editing this script::

    # In .env (autoloaded by examples/_common.py):
    AGENTCODEC_SEMKNN_SERVER_URL=http://127.0.0.1:18765

    # ...or from the shell, one-shot:
    AGENTCODEC_SEMKNN_SERVER_URL=http://127.0.0.1:18765 \\
        python examples/09_routed_semknn.py

The library auto-uses the BGE-small ONNX encoder from fastembed (~130 MB
first-run download). No extra installs needed for SemKNN client-side
encoding.

Run::

    python examples/09_routed_semknn.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule, RemoteSemKNNRouter

# Importing _common auto-loads .env, so anything you set there
# (AGENTCODEC_SEMKNN_SERVER_URL, AGENTCODEC_TELEMETRY_ENDPOINT, etc.)
# is in os.environ by the time we build the module below.
from _common import critic_same, model_block, print_result


def main() -> None:
    mod = ReliabilityModule.from_dict({
        # Your real channel pool — the backend matches the canonical model
        # families against trained profiles to pick the right q-matrix.
        "models": [
            model_block("qwen3:8b", temperature=0.7),
            model_block("llama3.1:8b", temperature=0.7),
        ],
        "judge": {
            "model": "gemma3:12b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
        "critic": critic_same(),
        "strategy": {
            "type": "routed",
            "router": {
                "type": "semknn",
                # `server_url` is OPTIONAL — the library defaults to the
                # public hosted backend (https://agentcodec.intellerce.com).
                # Set AGENTCODEC_SEMKNN_SERVER_URL in .env to override
                # for local dev / self-hosted deployments.
                "lambda": 1.0,
                # Optional knobs:
                # "api_key": "...",          # or set $AGENTCODEC_API_KEY
                # "knn_k_override": 20,
                # "strict_match": False,     # accept estimate fallbacks
                # "fallback": "linear",      # offline degradation
                # "fallback_cache": "weights/linear.json",
            },
            "dispatch": {
                "harq_ir": {"max_rounds": 4},
            },
        },
        "defaults": {
            "category": "auto",
            "on_error": "fallback_baseline",
        },
        # Tasks WITH a score_mode + reference get deterministic scoring
        # (no judge call). Tasks WITHOUT one still fall through to the
        # judge — so this switch is safe even when only some prompts
        # have a ground truth.
        "score_strategy": "exact",
    })

    assert isinstance(mod.router, RemoteSemKNNRouter)
    print(f"SemKNN backend:  {mod.router.server_url}")
    print(f"BGE model:       {mod.router.bge_model}")
    print(f"Lambda:          {mod.router.lambda_}")
    print()

    # (category, prompt, reference, score_mode)
    # `category` and `reference`/`score_mode` are all optional — pass None
    # to let the router auto-classify or to keep judge-based scoring.
    prompts = [
        ("qa",        "What is the capital of France?",
                      "Paris", "exact_match"),
        (None,        "If a train leaves Boston at 60 mph and another leaves "
                      "New York at 80 mph, when do they meet?",
                      None, None),
        # No test fixture here, so this is judge-scored, not executed — the
        # point of this example is SemKNN *routing*, not code correctness.
        # For execution-based code scoring see examples 02 and 16.
        (None,        "Write a Python function that returns the nth Fibonacci "
                      "number using memoization.",
                      None, None),
    ]

    for category, prompt, reference, score_mode in prompts:
        result = mod.run(
            prompt,
            category=category,
            reference=reference,
            score_mode=score_mode,
            return_trace=True,
        )
        label = category or "auto"
        print_result(result, label=f"{label}: {prompt[:60]}", score_mode=score_mode)
        # Surface SemKNN-specific provenance so you can see what /route returned.
        router_extra = result.trace.get("router", {}).get("extra", {})
        print(f"  match_quality : {router_extra.get('match_quality')}")
        print(f"  estimate       : {router_extra.get('estimate')}")
        print(f"  predicted q    : {router_extra.get('predicted_quality_for_chosen')}")
        print(f"  observed q     : {result.final_quality}")
        print()


if __name__ == "__main__":
    main()
