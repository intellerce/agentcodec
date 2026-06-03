---
name: "Cost / pricing"
about: A model fell back to default_fallback, or cost_usd looks wrong
title: "[cost] "
labels: pricing
---

**Which tier did the channel land on?**

`exact_user_rate` / `exact_table_rate` / `inferred_table_rate` /
`default_fallback` / `tokens_estimated_from_chars` /
`tokens_estimated_with_thinking_chars`

You can check via `result.cost_source` or the per-call `cost_source`
in the trace.

**Model name and provider**

e.g. `mycorp/finetuned-llm-13b` via OpenRouter.

**What rate should it be?**

Per-million-token input / output, with a source link if it's a public
catalog. If it's an internal price, just state it — we'll happily land
a `cost_per_1m` override example without exposing the number.

**Suggested fix**

- [ ] Add to `MODEL_COSTS` in `agentcodec/channel.py`
- [ ] Document `cost_per_1m: { input: ..., output: ... }` for this
      model in a config example
- [ ] Improve the `inferred_table_rate` heuristic
- [ ] Something else (explain)
