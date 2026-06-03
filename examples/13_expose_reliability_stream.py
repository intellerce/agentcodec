"""
13 — `expose_reliability_stream=True` across all three compat shims.

By default the drop-in shims (`agentcodec.openai.OpenAI` /
`agentcodec.anthropic.Anthropic` / `agentcodec.ollama.Client`) look
exactly like the native SDK: streaming chunks carry only the final
``answer`` text and (when the model emits it) ``thinking``. Intermediate
reliability-layer roles — ``draft``, ``critique``, ``verification``,
``candidate`` — are hidden so the answer stream stays clean.

Setting ``expose_reliability_stream=True`` surfaces those internal
roles **with sentinel fields** that existing consumers ignore:

  OpenAI    → ``delta.agentcodec_role`` and ``delta.agentcodec_call_id``
  Anthropic → same names on ``content_block_delta`` events
  Ollama    → ``message.agentcodec_role`` and ``message.agentcodec_call_id``

This example runs the same HARQ-IR pipeline through all three shims with
the flag set, and prints role-tagged chunks as they arrive so you can
see how a UI would render drafts vs critiques vs the final answer.

Run::

    # Local Ollama (default — only the Ollama section runs).
    python examples/13_expose_reliability_stream.py

    # OpenAI (cloud):
    AGENTCODEC_EXAMPLE_BASE_URL=https://api.openai.com/v1 \\
    AGENTCODEC_EXAMPLE_API_KEY=$OPENAI_API_KEY \\
    AGENTCODEC_EXAMPLE_MODEL_A=gpt-4o-mini \\
    AGENTCODEC_EXAMPLE_JUDGE=gpt-4o-mini \\
    python examples/13_expose_reliability_stream.py openai

    # Anthropic:
    AGENTCODEC_EXAMPLE_MODEL_A=claude-sonnet-4-5 \\
    AGENTCODEC_EXAMPLE_JUDGE=claude-haiku-4-5 \\
    python examples/13_expose_reliability_stream.py anthropic
"""
from __future__ import annotations

import asyncio
import sys

from _common import API_KEY, BASE_URL, JUDGE, MODEL_A


# ---------------------------------------------------------------------------
# Role colors — same scheme as 10_async_streaming.py for visual continuity
# ---------------------------------------------------------------------------
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


def _print_chunk(role: str | None, text: str) -> None:
    if not text:
        return
    color = ROLE_COLOR.get(role or "answer", "")
    print(f"{color}[{role or 'answer':>12}]{RESET} {text}")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


async def demo_openai() -> None:
    """OpenAI-compat path. Works against api.openai.com OR Ollama's /v1.

    With ``expose_reliability_stream=True``, the shim emits drafts,
    critiques, etc. as ``delta.content`` chunks with a
    ``delta.agentcodec_role`` sentinel. Thinking still goes via
    ``delta.reasoning_content`` (matches OpenAI's o-series shape).
    """
    from agentcodec.openai import AsyncOpenAI

    print(f"\n{'=' * 70}\nOpenAI shim (model={MODEL_A})\n{'=' * 70}")
    client = AsyncOpenAI(
        api_key=API_KEY, base_url=BASE_URL,
        reliability="harq_ir",
        expose_reliability_stream=True,
    )
    stream = await client.chat.completions.create(
        model=MODEL_A,
        messages=[{"role": "user", "content": "In one line: what is QUIC?"}],
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        role = getattr(delta, "agentcodec_role", None)
        text = getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
        if text:
            # role is None on raw passthrough chunks (e.g. role="assistant" header);
            # the very first chunk carries `role="assistant"` and no agentcodec_role —
            # treat that as the regular answer stream.
            _print_chunk(role or "answer", text)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


async def demo_anthropic() -> None:
    """Anthropic-compat path. Same flag, different chunk shape.

    Anthropic streams ``content_block_delta`` events. Drafts/critiques
    surface as ``text_delta`` with ``delta.agentcodec_role`` set;
    thinking lands on its own ``content_block`` (index 1) with
    ``thinking_delta`` deltas.
    """
    from agentcodec.anthropic import AsyncAnthropic

    print(f"\n{'=' * 70}\nAnthropic shim (model={MODEL_A})\n{'=' * 70}")
    client = AsyncAnthropic(
        api_key=API_KEY,
        reliability="harq_ir",
        expose_reliability_stream=True,
    )
    async with client.messages.stream(
        model=MODEL_A,
        messages=[{"role": "user", "content": "In one line: what is QUIC?"}],
        max_tokens=512,
    ) as stream:
        async for event in stream:
            etype = getattr(event, "type", None)
            if etype != "content_block_delta":
                continue
            delta = event.delta
            if getattr(delta, "type", None) == "text_delta":
                role = getattr(delta, "agentcodec_role", None) or "answer"
                _print_chunk(role, getattr(delta, "text", ""))
            elif getattr(delta, "type", None) == "thinking_delta":
                _print_chunk("thinking", getattr(delta, "thinking", ""))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


async def demo_ollama() -> None:
    """Ollama-compat path. Dict-based shape; sentinels live in
    ``chunk["message"]["agentcodec_role"]``."""
    from agentcodec.ollama import AsyncClient

    print(f"\n{'=' * 70}\nOllama shim (model={MODEL_A})\n{'=' * 70}")
    client = AsyncClient(
        host=BASE_URL.rstrip("/v1"),  # AsyncClient wants the bare host
        reliability="harq_ir",
        expose_reliability_stream=True,
    )
    stream = await client.chat(
        model=MODEL_A,
        messages=[{"role": "user", "content": "In one line: what is QUIC?"}],
        stream=True,
    )
    async for chunk in stream:
        msg = chunk.get("message") or {}
        text = msg.get("content") or msg.get("thinking") or ""
        role = msg.get("agentcodec_role") or (
            "thinking" if msg.get("thinking") else "answer"
        )
        if text:
            _print_chunk(role, text)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "openai":
        await demo_openai()
    elif arg == "anthropic":
        await demo_anthropic()
    elif arg == "ollama":
        await demo_ollama()
    elif arg == "all":
        await demo_openai()
        await demo_anthropic()
        await demo_ollama()
    else:
        # Default: just the Ollama-compat path so the script runs against
        # the local Ollama default without any API key.
        await demo_ollama()


if __name__ == "__main__":
    asyncio.run(main())
