"""
05 — Streaming, the simplest version: watch tokens arrive live.

This is the "hello world" of streaming. ``mod.astream()`` drives the
backend natively in an event loop and emits ``TokenEvent``s as the model
produces text — so the answer appears token-by-token in your terminal,
not all at once at the end.

Two things to know:

  * Per-token streaming lives on the **async** API (``astream``). The sync
    ``mod.stream()`` only emits stage-level ``ProgressEvent``s plus one
    final ``TokenEvent`` with the whole answer — useful for progress bars,
    but it does not stream tokens. If you want live tokens, use ``astream``
    (this file).
  * Every ``TokenEvent`` carries a ``role``. With a multi-pass technique
    like HARQ-IR you'll see the round-by-round ``draft`` and ``critique``
    stream first, then the final ``answer`` — each tagged so a UI can route
    them to different panes. (``thinking`` tokens are captured to
    ``result.thinking_text`` rather than printed here.)

For the fuller treatment — ANSI-colored roles, a technique selectable from
``argv``, and parallel-branch streaming — see ``10_async_streaming.py`` and
``11_async_diversity_mrc.py``.

Run:
    python examples/05_streaming.py
"""
from __future__ import annotations

import asyncio

from agentcodec import (
    FinalEvent,
    ProgressEvent,
    ReliabilityModule,
    TokenEvent,
    WarningEvent,
)

from _common import critic_same, judge_block, model_block


async def main() -> None:
    mod = ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.7)],
        "judge": judge_block(),
        "critic": critic_same(),
        "strategy": {
            "type": "fixed",
            "technique": "harq_ir",
            "params": {"max_rounds": 3},
        },
        "defaults": {"category": "auto"},
    })

    prompt = "Why is JSON not a good wire format for sub-microsecond RPC?"
    print(f"Streaming: {prompt!r}\n")

    current_role: str | None = None
    final_result = None
    with mod:  # ReliabilityModule is a sync context manager; astream runs inside it
        async for event in mod.astream(prompt, category="qa"):
            if isinstance(event, ProgressEvent):
                # Close any open token line before printing a stage update.
                if current_role is not None:
                    print()
                    current_role = None
                print(f"  [{event.elapsed_s:5.2f}s] {event.stage}: {event.detail}")

            elif isinstance(event, TokenEvent):
                if event.role == "thinking":
                    continue  # captured to result.thinking_text, not shown
                # Print a role banner when the role changes, then stream the
                # tokens of that segment inline as they arrive.
                if event.role != current_role:
                    print(f"\n  [{event.role}] ", end="", flush=True)
                    current_role = event.role
                print(event.text, end="", flush=True)

            elif isinstance(event, WarningEvent):
                if current_role is not None:
                    print()
                    current_role = None
                print(f"  [WARN {event.severity}] {event.code}: {event.message}")

            elif isinstance(event, FinalEvent):
                final_result = event.result

    if current_role is not None:
        print()
    if final_result is not None:
        fq = final_result.final_quality
        quality = f"{fq:.3f}" if fq is not None else "n/a"
        print(
            f"\n  done — technique={final_result.technique_used}, "
            f"rounds={final_result.rounds}, quality={quality}"
        )
        print(f"  cost=${final_result.cost_usd:.6f} ({final_result.cost_source})")


if __name__ == "__main__":
    asyncio.run(main())
