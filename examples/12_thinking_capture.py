"""
12 — Thinking-text capture across all backends.

Models that emit a separate reasoning channel (Anthropic ThinkingBlock,
OpenAI o-series / GPT-5 reasoning_content, Ollama msg.thinking) — and
models that emit inline ``<think>...</think>`` tags (DeepSeek-R1, Qwen3,
GLM-4.5+, Nemotron, Phi-4-reasoning) — all populate
``ReliabilityResult.thinking_text`` and per-call ``trace["calls"][*]["thinking"]["text"]``.

Cost is split between thinking and answer tokens via
``thinking_cost_usd`` / ``answer_cost_usd``.

This example runs a thinking-capable model on a math task that benefits
from reasoning and prints the captured chain-of-thought separately from
the user-facing answer.

Run:
    # Local thinking model via Ollama:
    python examples/12_thinking_capture.py

    # OpenAI o-series:
    AGENTCODEC_EXAMPLE_BASE_URL=https://api.openai.com/v1 \\
    AGENTCODEC_EXAMPLE_API_KEY=$OPENAI_API_KEY \\
    AGENTCODEC_EXAMPLE_MODEL_A=o4-mini \\
    AGENTCODEC_EXAMPLE_JUDGE=gpt-4o-mini \\
    python examples/12_thinking_capture.py

    # Anthropic with extended thinking:
    AGENTCODEC_EXAMPLE_MODEL_A=claude-sonnet-4-5 \\
    AGENTCODEC_EXAMPLE_JUDGE=claude-haiku-4-5 \\
    python examples/12_thinking_capture.py
"""
from __future__ import annotations

import asyncio

from agentcodec import (
    FinalEvent,
    ProgressEvent,
    ReliabilityModule,
    TokenEvent,
)

from _common import MODEL_A, BASE_URL, API_KEY, JUDGE


def _build_model_block() -> dict:
    """Build a model block that enables thinking for the configured model.

    Different backends translate ``thinking`` differently — the library
    handles this; we just pass the high-level flag.
    """
    model = MODEL_A.lower()
    block = {
        "model": MODEL_A,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "temperature": 0.7,
    }
    if model.startswith("claude-"):
        # Anthropic: extended-thinking with a 4096-token budget.
        block["thinking"] = {"enabled": True, "budget_tokens": 4096}
    elif model.startswith(("o1", "o3", "o4", "gpt-5")):
        # OpenAI o-series / GPT-5: reasoning_effort medium (~4k tokens).
        block["thinking"] = {"enabled": True, "budget_tokens": 4096}
    else:
        # Ollama / vLLM local thinking models: just on.
        block["thinking"] = True
    return block


async def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [_build_model_block()],
        "judge": {"model": JUDGE, "base_url": BASE_URL, "api_key": API_KEY},
        "critic": {"same": True},
        "strategy": {"type": "fixed", "technique": "baseline"},
        "defaults": {
            "category": "auto",
            "streaming": {"events": "all", "emit_thinking_tokens": True},
        },
    })

    prompt = (
        "If a train leaves station A at 12:00 traveling 60 km/h, and "
        "another leaves station B at 12:30 traveling 80 km/h toward A "
        "(stations are 200 km apart), when do they meet? Show your work."
    )

    print(f"Streaming with thinking captured. Model: {MODEL_A}")
    print(f"Prompt: {prompt}\n")

    thinking_chars = 0
    answer_chars = 0
    final_result = None

    with mod:
        async for ev in mod.astream(prompt):
            if isinstance(ev, TokenEvent):
                if ev.role == "thinking":
                    if thinking_chars == 0:
                        print("[THINKING]", end=" ", flush=True)
                    print(ev.text, end="", flush=True)
                    thinking_chars += len(ev.text)
                elif ev.role == "answer":
                    if answer_chars == 0:
                        print("\n\n[ANSWER]", end=" ", flush=True)
                    print(ev.text, end="", flush=True)
                    answer_chars += len(ev.text)
            elif isinstance(ev, ProgressEvent) and ev.stage in (
                "channel_start", "channel_complete",
            ):
                # Don't disrupt the stream; just record.
                pass
            elif isinstance(ev, FinalEvent):
                final_result = ev.result

    print("\n")
    if final_result is not None:
        print("=" * 70)
        print(f"  thinking_used     : {final_result.thinking_used}")
        print(f"  thinking_text len : {len(final_result.thinking_text or '')} chars")
        print(f"  total cost_usd    : ${final_result.cost_usd:.6f} "
              f"({final_result.cost_source})")
        print(f"  thinking_cost_usd : ${final_result.thinking_cost_usd:.6f}")
        print(f"  thinking_tokens   : {final_result.thinking_tokens}")
        print(f"  output_tokens     : {final_result.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
