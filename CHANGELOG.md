# Changelog

All notable changes to the AgentCodec library are documented here. The
first tagged release starts a normal [Keep a Changelog](https://keepachangelog.com)
cadence; the pre-1.0 history below is a single "Unreleased" block because the
internal iteration history (benchmark runs, router retrains, profile rebuilds)
isn't meaningful to public users.

## Unreleased — initial public release

First public, source-available push. Not on PyPI yet — install from Git
(see the README).

### Added

- License: **PolyForm Noncommercial 1.0.0**. Source-available; free for
  research, teaching, and personal / internal evaluation. Commercial use
  requires a separate license — see [COMMERCIAL.md](COMMERCIAL.md).
- Public client is **SemKNN-remote**: the package ships only the client
  surface; the trained q-matrix lives on a backend. The client sends a
  unit-norm BGE embedding (`bge-small-en-v1.5` by default) plus a small
  channel-pool fingerprint — never the prompt text. Self-hosting the
  backend (and the trained artifacts) is available under a separate
  license.
- Anonymous usage telemetry is on by default for **every router**. Sends
  the prompt embedding (already going to `/route` for SemKNN; lazy-encoded
  client-side for fixed / ACMTable / ACMLinear), `lambda`, technique
  chosen, latency, predicted vs. observed quality, token counts, and a
  canonical model-family fingerprint. **Never sends:** prompt text, model
  outputs, reference answers, concrete model identifiers, API keys, or
  any user identifier. Master kill switch: `AGENTCODEC_TELEMETRY=0`.
  Endpoint override: `AGENTCODEC_TELEMETRY_ENDPOINT=...` or
  `telemetry.endpoint:` in YAML. Defense in depth: client-side `_scrub` +
  server-side `_scrub_event` pin a forbidden-keys denylist; a schema-pin
  test fails if a new field ever lands in the payload.
- `RouterConfig` schema (Pydantic, strict — rejects unknown keys):
  - `type: semknn` requires `server_url` + `lambda`; `cache:` is rejected
    with a migration error.
  - `type: acm_linear` requires `cache:`; SemKNN-only fields are rejected.
  - `type: acm_table` requires `table:` or `category_tables:`.
  - `fallback: linear` requires `fallback_cache:`.
- Cost transparency: every `cost_usd` carries a `cost_source` tier
  (`exact_user_rate` → `exact_table_rate` → `inferred_table_rate` →
  `default_fallback` → `tokens_estimated_*`) with explicit caveats.
  Construction-time pricing log + loud warning on `default_fallback`.
- Thinking control: per-channel
  `thinking: bool | "auto" | {enabled, budget_tokens}`. Per-backend
  translation (Anthropic, OpenAI o-series / GPT-5, Ollama / vLLM via
  `enable_thinking`). `budget_seconds` is rejected at config-load with a
  clear error.
- `Evaluator` for multi-config head-to-head: bootstrap CIs, paired
  Wilcoxon + Benjamini-Hochberg, Cohen's d, Pareto frontiers, per-config
  cost/latency/quality table. Optional `scipy` via the `[eval]` extra.
- Drop-in provider shims: `agentcodec.openai`, `agentcodec.anthropic`,
  `agentcodec.ollama` — swap one import, get reliability with an optional
  `reliability=` kwarg. Pure passthrough when unused.
- 29 dispatchable techniques across six families plus 7 prior-method
  baselines (Self-Consistency, Self-Refine, Chain-of-Verification,
  Best-of-N, Weighted Best-of-N, CISC, Mixture-of-Agents).
- Examples suite under `examples/` (00–16) covering drop-in usage,
  per-technique demos, async streaming, thinking capture, code scoring,
  and the showcase lift table.
- Packaging: `pip install agentcodec[...]` with extras `openai`,
  `anthropic`, `ollama`, `remote-semknn`, `eval`, `benchmark`, `all`,
  `dev`. Core install is torch-free (fastembed / ONNX BGE encoder).

### Note on the SemKNN backend

The SemKNN routing service is **not** part of this package. It serves the
trained q-matrix, training-prompt embeddings, and cost vector produced from
paid benchmark runs, which are not redistributed. The public client talks to
it over HTTPS, sending only a unit-norm embedding (never the prompt).
Self-hosting requires a license — see [COMMERCIAL.md](COMMERCIAL.md).
