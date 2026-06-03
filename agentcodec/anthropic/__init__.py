"""
Drop-in replacement for ``anthropic.Anthropic`` with optional reliability layer.

Change one import:

    - from anthropic import Anthropic
    + from agentcodec.anthropic import Anthropic

Without ``reliability=`` the wrapper is a true passthrough: the native
``anthropic.Anthropic`` client is constructed lazily, every call goes
through unchanged. To opt into the reliability layer, pass a preset,
a ``ReliabilityModule``, or a config dict:

    client = Anthropic(api_key=KEY, reliability="harq_ir")
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        system="You are a librarian.",
        messages=[{"role": "user", "content": "Recommend a book."}],
        max_tokens=1024,
    )
    print(resp.content[0].text)         # native shape preserved
    print(resp.reliability.technique_used)   # power-user escape hatch
"""
from __future__ import annotations

from .client import Anthropic, AsyncAnthropic

__all__ = ["Anthropic", "AsyncAnthropic"]
