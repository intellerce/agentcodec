# Contributing to AgentCodec

Thanks for your interest. AgentCodec is **source-available under the
PolyForm Noncommercial License 1.0.0**. Contributions are welcome —
the rest of this document is the legal and practical machinery that
makes them stick.

## Licensing of contributions

By submitting a pull request you agree that:

1. Your contribution is your own work (or you have the right to submit
   it), and
2. You **license your contribution to INTELLERCE LLC under both** the
   PolyForm Noncommercial License 1.0.0 (matching the public release)
   **and** under any commercial license INTELLERCE may grant to third
   parties for the rest of the codebase.

In plain English: we need to be able to keep selling commercial
licenses to companies that can't use PolyForm-NC. We can't do that if
parts of the code are under a license that forbids it. Every other
open-core project (Sentry, Grafana, ClickHouse, Cal.com, …) has the
same requirement; we are no different.

We use the **Developer Certificate of Origin (DCO)** to capture this.
Sign every commit with `git commit -s`. That appends a `Signed-off-by:`
trailer that asserts you have the right to submit the patch under
these terms. Full DCO text: <https://developercertificate.org>.

If your employer asserts ownership of code you write in your spare
time, please clear contributions with them in writing before
submitting.

## Getting set up

```bash
# 1. Fork + clone
git clone https://github.com/<your-fork>/agentcodec.git
cd agentcodec

# 2. uv-managed virtualenv with dev extras
./setup.sh                  # or `./setup.sh --remote` for SemKNN client
source .venv/bin/activate

# 3. Verify the install
pytest tests/ -q            # unit tests; no network calls
ruff check agentcodec/      # lint
mypy agentcodec/            # types
```

The library targets Python ≥ 3.10. CI runs on 3.10 / 3.11 / 3.12.

## Running with a real LLM

The examples expect a local Ollama with `qwen2.5:7b`, `llama3.1:8b`,
and `gemma3:12b`. One-liner:

```bash
ollama pull qwen2.5:7b llama3.1:8b gemma3:12b
```

For cloud providers, set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and point
your model configs at the provider — see the fully-annotated
[`examples/example_config.yaml`](examples/example_config.yaml) for every knob.

## What we accept

In rough priority order:

- **Bug fixes**, especially anything that breaks `cost_source`
  accuracy, telemetry privacy, or the strict-config validator.
- **New techniques** that fit the communication-theoretic taxonomy
  (HARQ / diversity / turbo / fountain / FEC / ACM). Add them in
  `agentcodec/techniques/`, wire into `dispatch.py`, write at least
  one dispatch smoke test against `MockChannel`.
- **Provider/backend adapters** — vLLM, SGLang, OpenRouter quirks,
  Together, DeepSeek, Bedrock, Vertex. Keep them inside the
  OpenAI-compat path when possible; add a per-backend transmit only if
  the dialect actually diverges.
- **Documentation improvements** — README clarity, more examples,
  tutorial notebooks, better docstrings.
- **Integration shims** for LangChain / LlamaIndex / Mirascope /
  others.

What we generally do **not** accept:

- Hard dependencies on additional heavy packages (torch beyond the
  `[remote-semknn]` extra, pandas, scikit-learn). Move them to an
  extras-group and lazy-import.
- Vendored copies of third-party SDKs.
- Anything that ships LLM API keys or trained SemKNN artifacts.
- Renames or restructures of the public surface without a migration
  story.

## Pull-request hygiene

- One concern per PR. Rebase to a clean history; squash trivial
  fixups.
- New behavior gets a test. Privacy- or cost-affecting behavior gets a
  *targeted* test (see `tests/test_telemetry_schema_pinned.py` for the
  pattern).
- `ruff check agentcodec/` + `pytest tests/` must be green.
- The PR description should explain **why** the change matters; the
  diff already explains what.

## Reporting bugs

File an issue with:

- AgentCodec version (`python -c "import agentcodec; print(agentcodec.__version__)"`)
- Your YAML config (redact API keys)
- Minimal Python snippet that reproduces
- The full stack trace or the relevant section of the trace dict

For privacy- or security-sensitive issues, see
[SECURITY.md](./SECURITY.md) instead.

## Coordinating bigger changes

For anything more than a few hundred lines, open an issue first to
agree on the shape. We don't want to send anyone down a path that ends
in a rejected PR.

Email for design discussions, academic collaborations, or anything
that doesn't fit a public issue: **research@intellerce.com**.
