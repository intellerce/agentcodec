"""
Drop-in replacement for ``openai.OpenAI`` with optional reliability layer.

Change one import:

    - from openai import OpenAI
    + from agentcodec.openai import OpenAI

No further code changes. With no ``reliability=`` kwarg the wrapper is a
true passthrough — the native ``openai.OpenAI`` is instantiated lazily
and every call goes through unchanged. To opt into the reliability
layer, pass a preset name, a ``ReliabilityModule``, or a config dict:

    client = OpenAI(api_key=KEY, reliability="harq_ir")

    client = OpenAI(api_key=KEY, reliability=ReliabilityModule.from_yaml(...))

Per-call override:

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        reliability={"technique": "harq_ir", "max_rounds": 2},
    )
"""
from __future__ import annotations

from .client import AsyncOpenAI, OpenAI

__all__ = ["AsyncOpenAI", "OpenAI"]
