"""
10 — Async streaming with native per-token deltas.

This is the canonical async-streaming entry point. Unlike the older
``05_streaming.py`` (which used the sync ``mod.stream()`` worker-thread
bridge), this example drives ``mod.astream()`` natively in an event loop
and renders per-token deltas as the model produces them.

The native astream path covers 16 techniques as of v0.4:

  Sequential — baseline, harq_ir, harq_cc, turbo, self_refine, chain_of_verification
  Parallel   — best_of_n, weighted_bon, self_consistency, mixture_of_agents,
               diversity_sc / mrc / egc / spatial / frequency / time

For any technique not in the above set, ``astream()`` still works — it
falls back to sync ``dispatch()`` in an executor and emits only the
terminal ``FinalEvent``. Use ``agentcodec.dispatch.is_streamable(name)``
to introspect.

Run:
    python examples/10_async_streaming.py
"""
from __future__ import annotations

import asyncio
import sys

from agentcodec import (
    FinalEvent,
    ProgressEvent,
    ReliabilityModule,
    TokenEvent,
    WarningEvent,
)

from _common import MODEL_A, MODEL_B, critic_same, judge_block, model_block, print_result


# ANSI color helpers — make role-tagged tokens easy to spot in the terminal.
RESET = "\x1b[0m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
GREEN = "\x1b[32m"
MAGENTA = "\x1b[35m"
GRAY = "\x1b[90m"


ROLE_COLOR = {
    "answer": GREEN,
    "draft": CYAN,
    "critique": YELLOW,
    "verification": MAGENTA,
    "synthesis": GREEN,
    "candidate": GRAY,
    "thinking": DIM,
}


def _color_for(role: str) -> str:
    return ROLE_COLOR.get(role, "")


async def stream_one(mod: ReliabilityModule, prompt: str, *, label: str) -> None:
    """Drive mod.astream() and pretty-print every event role-by-role."""
    print(f"\n{'=' * 70}")
    print(f"Streaming: {label!r}")
    print(f"Prompt: {prompt!r}")
    print("=" * 70)

    current_role: str | None = None
    final_result = None
    async for event in mod.astream(prompt, category="qa"):
        if isinstance(event, ProgressEvent):
            # End any in-progress token line cleanly before a progress event.
            if current_role is not None:
                print(RESET)
                current_role = None
            print(f"  {GRAY}[{event.elapsed_s:5.2f}s] {event.stage:24s} "
                  f"{event.detail}{RESET}")

        elif isinstance(event, TokenEvent):
            if current_role != event.role:
                if current_role is not None:
                    print(RESET)
                # Show a fresh role banner so the user can demux the stream.
                print(f"  {_color_for(event.role)}[{event.role}] ", end="",
                      flush=True)
                current_role = event.role
            print(event.text, end="", flush=True)

        elif isinstance(event, WarningEvent):
            if current_role is not None:
                print(RESET)
                current_role = None
            print(f"  {YELLOW}[WARN {event.severity}] {event.code}: "
                  f"{event.message}{RESET}")

        elif isinstance(event, FinalEvent):
            if current_role is not None:
                print(RESET)
                current_role = None
            final_result = event.result

    print()
    if final_result is not None:
        print_result(final_result, label=label)
        if final_result.thinking_text:
            print(f"  thinking   : {final_result.thinking_text[:200]}{'…' if len(final_result.thinking_text) > 200 else ''}")
            print(f"  thinking $ : {final_result.thinking_cost_usd:.6f}")


async def main() -> None:
    # Pick a technique from argv, default to harq_ir which produces lots of
    # interesting events (round drafts + critiques + final).
    technique = sys.argv[1] if len(sys.argv) > 1 else "harq_ir"

    mod = ReliabilityModule.from_dict({
        # Two models so diversity techniques have spatial diversity.
        "models": [model_block(MODEL_A), model_block(MODEL_B)],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {"type": "fixed", "technique": technique},
        "defaults": {
            "category": "auto",
            "streaming": {"events": "all", "emit_thinking_tokens": True},
        },
    })

    prompts = [
        "Why is JSON not a good wire format for sub-microsecond RPC?",
    ]
    with mod:
        for p in prompts:
            await stream_one(mod, p, label=f"{technique}")


if __name__ == "__main__":
    asyncio.run(main())
