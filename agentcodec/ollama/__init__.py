"""
Drop-in replacement for ``ollama.Client`` with optional reliability layer.

Change one import:

    - from ollama import Client
    + from agentcodec.ollama import Client

Without ``reliability=`` the wrapper is a true passthrough. Otherwise:

    client = Client(host="http://localhost:11434", reliability="harq_ir")
    resp = client.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(resp["message"]["content"])          # native dict shape preserved
    print(resp["reliability"]["technique_used"])  # power-user escape hatch

Module-level helpers ``chat()``, ``generate()``, ``embed()`` are also
exported as wrappers around an implicit default Client, matching the
``ollama`` library's top-level API.
"""
from __future__ import annotations

from .client import AsyncClient, Client

__all__ = ["AsyncClient", "Client"]
