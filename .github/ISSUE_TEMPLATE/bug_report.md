---
name: Bug report
about: Something works less reliably than the docs claim
title: "[bug] "
labels: bug
---

**What you did**

```python
# Minimal reproducer
```

**What you expected**

…

**What happened instead**

…

**Environment**

- `agentcodec` version: `python -c "import agentcodec; print(agentcodec.__version__)"`
- Python:
- OS:
- Provider(s) involved: e.g. Ollama (qwen2.5:7b), Anthropic, OpenAI

**Your config**

```yaml
# Redact API keys.
```

**Trace (if available)**

```python
result = mod.run(prompt, return_trace=True)
print(json.dumps(result.to_dict(), indent=2))
```

Paste the relevant section here (don't paste your whole prompt or the
model output unless they're necessary to understand the bug).
