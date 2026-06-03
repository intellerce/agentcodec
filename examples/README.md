# AgentCodec examples

Each script is self-contained and runnable. They all share `_common.py`,
which builds the model/judge config from environment variables — by
default targeting a local Ollama server (matches `configs/quick.yaml`).

`_common.py` also provides the score-reporting helpers every example uses:
`print_result(result, score_mode=...)` prints `final_quality` plus a
one-line note on how it was scored, and `explain_score(result, score_mode=...)`
(on a result run with `return_trace=True`) renders the judge's full
15-criterion checklist — which criteria passed, their weights, and the
weighted sum — so you can see exactly where a score came from.

```bash
# from the package source root:
pip install -e '.[openai]'             # core + the OpenAI-compat SDK (Ollama uses it)

# Pull the three models the examples assume (one-time, ~12 GB total).
ollama pull qwen3:8b llama3.1:8b gemma3:12b
ollama serve &

python examples/01_hello_baseline.py
```

## Index

| #  | File                          | Technique         | What it shows |
| -- | ----------------------------- | ----------------- | -------------- |
| 00 | `00_drop_in_openai.py`, `00_drop_in_anthropic.py`, `00_drop_in_ollama.py` | (passthrough) | Drop-in shims: change one import, add an optional `reliability=` kwarg, keep the provider-native response shape. One script per provider (OpenAI / Anthropic / Ollama). |
| 01 | `01_hello_baseline.py`        | `baseline`        | Minimal setup. The `ReliabilityResult` shape every other example returns. |
| 02 | `02_self_refine.py`           | `self_refine`     | One model loops `draft → critique → revise`, on a code task with a test fixture so each draft is **executed** in the sandbox; `explain_score` shows the blended sandbox + judge breakdown. |
| 03 | `03_diversity_mrc.py`         | `diversity_mrc`   | Two different models answer in parallel; judge produces a quality-weighted blend. |
| 04 | `04_harq_ir.py`               | `harq_ir`         | Critic emits structured deltas → iterative incremental redundancy. |
| 05 | `05_streaming.py`             | (any)             | Consume `ProgressEvent` / `TokenEvent` / `FinalEvent` from `mod.stream()`. |
| 06 | `06_routed_acm_table.py`      | routed            | Per-prompt routing via a hand-coded difficulty table — easy prompts get baseline, hard ones get diversity. |
| 07 | `07_compare_techniques.py`    | comparison        | Same prompt, five techniques, one line of summary each. |
| 08 | `08_from_yaml.py` + [`example_config.yaml`](example_config.yaml) | (any) | Load a fully-annotated YAML config via `ReliabilityModule.from_yaml`. Covers every public knob. |
| 09 | `09_routed_semknn.py`         | routed (remote)   | Per-prompt routing via the remote SemKNN service. Sends a unit-norm BGE embedding (never the prompt) and prints predicted vs. observed quality. Set `AGENTCODEC_SEMKNN_SERVER_URL` to point at your backend. |
| 10 | `10_async_streaming.py`       | (any)             | Drive `mod.astream()` natively in `asyncio` and render per-token deltas as the model produces them. Pass a technique name as `argv[1]` to compare (e.g. `python 10_async_streaming.py turbo`). |
| 11 | `11_async_diversity_mrc.py`   | `diversity_mrc`   | Parallel-branch streaming: branches run concurrently via `asyncio.gather`; per-branch `ProgressEvent`s show real concurrency, then the judge synthesizer emits the combined answer. |
| 12 | `12_thinking_capture.py`      | (any)             | Captures the model's internal reasoning (Anthropic `ThinkingBlock`, OpenAI `reasoning_content`, Ollama `msg.thinking`, inline `<think>` tags) into `result.thinking_text` with cost split between thinking and answer tokens. |
| 13 | `13_expose_reliability_stream.py` | (any technique) | Demonstrates `expose_reliability_stream=True` on all three drop-in shims (OpenAI / Anthropic / Ollama) side-by-side. Drafts, critiques, verifications surface as native stream chunks with `agentcodec_role` sentinel fields. Pass `openai` / `anthropic` / `ollama` / `all` as argv[1] to pick which shim to drive. |
| 14 | `14_showcase_lift.py`         | comparison        | Loads pre-curated hard tasks from `showcase_tasks.json` (same directory) and runs each through five configs: `baseline`, fixed `harq_ir`, and a SemKNN λ-sweep at `λ=0`, `λ=3`, `λ=40`. Per-row table prints the technique each SemKNN column actually picked (with a `*` tag when the backend flagged `estimate`); aggregate footer summarizes per-λ technique counts so you can read whether routing converges from cost bias (e.g. all-`self_refine` at λ=40) or from the q-matrix quality argmax alone (same pick at λ=0). Also demonstrates that `strategy.technique="baseline"` produces the same `cost_usd` / `final_quality` traces as the other techniques. |
| 15 | `15_code_scoring.py`          | code scoring      | Scoring layer in isolation, no LLM round-trip: hand-written `correct` / `buggy` / `syntax-error` candidates passed directly to `score_deterministic("code", …)`. Also demos `score_mode="code_complexity"` with a fast O(n) vs. a real O(n²) implementation of `find_max`. Default sandbox = subprocess; flip to Docker via `AGENTCODEC_CODE_SANDBOX=docker`. |
| 16 | `16_code_scoring_end_to_end.py` | code scoring + LLM | Full pipeline: prompts the LLM to implement a HumanEval-style function, runs the model's actual output through the sandbox, scores blended `0.6 × sandbox + 0.4 × judge`. Two configurations on the same prompt — `baseline` (one shot) and `harq_ir` (iterative critic-and-refine) — so you can see the score drive harq_ir's early-exit decision. INFO logging surfaces the per-round `[SCORE …]` breakdown. |
| 17 | `17_prior_baselines.py`       | prior methods     | The six other prior-method baselines head-to-head on a numeric task: `self_consistency`, `best_of_n`, `weighted_bon`, `cisc`, `mixture_of_agents`, `chain_of_verification` (+ a `baseline` control), scored with `score_mode="numeric"`. Prints a quality/cost/calls leaderboard. |
| 18 | `18_score_modes.py`           | deterministic scoring | The non-code `score_mode`s — `exact_letter` (MMLU), `numeric` (math), `yes_no` — showing the `0.6 × deterministic + 0.4 × judge` blend and the extracted answer alongside the raw `{0,1}` deterministic signal. |
| 19 | `19_fountain_fec.py`          | `fountain`, `fec_*` | Rateless vs. fixed-rate redundancy: `fountain` draws samples adaptively until the judge is satisfied; `fec_0.75/0.50/0.33` commit a fixed redundancy budget by code rate. Compares adaptive vs. fixed call counts. |
| 20 | `20_turbo_harqcc.py`          | `turbo`, `harq_cc` | The rest of the iterative family alongside `harq_ir` as control: `harq_cc` (Chase combining) and `turbo` (generator/critic extrinsic-info exchange). Watch `rounds` vs. `quality`. |
| 21 | `21_diversity_family.py`      | `diversity_*`     | Full diversity sweep: SC/EGC/MRC combining rules, spatial/frequency/time axes, the wider-pool `*_N` variants, and the logprob-weighted `*_soft` variants (skipped gracefully on backends without logprobs). Ends with an `explain_score` breakdown of one run. |
| 22 | `22_evaluator.py`             | `Evaluator`       | Rigorous A/B: runs `baseline` / `harq_ir` / `diversity_mrc` over a shared prompt set with repeats, reporting per-config means with bootstrap 95% CIs, paired Wilcoxon significance vs. baseline, and a quality/cost/latency Pareto frontier. |

## Switching provider

The defaults assume Ollama at `http://localhost:11434/v1`. To run against
OpenAI / Anthropic / any OpenAI-compatible endpoint, export:

```bash
export AGENTCODEC_EXAMPLE_BASE_URL=https://api.openai.com/v1
export AGENTCODEC_EXAMPLE_API_KEY=$OPENAI_API_KEY
export AGENTCODEC_EXAMPLE_MODEL_A=gpt-4o-mini
export AGENTCODEC_EXAMPLE_MODEL_B=gpt-4o
export AGENTCODEC_EXAMPLE_JUDGE=gpt-4o-mini
```

See the top of [`_common.py`](_common.py) for the full env-var list.

## Where to go next

- Production deployment configs (YAML, not Python dicts): [`configs/lib/`](../configs/lib/).
- Full library API documentation: [`README.md`](../README.md) (sections "Library API" and "Streaming").
- Benchmark mode (paper reproduction): [`run_benchmark.py`](../run_benchmark.py).
- Per-technique design notes: docstrings in
  [`agentcodec/techniques/`](../agentcodec/techniques/) — `diversity.py`,
  `harq.py`, `turbo.py`, `fec.py`, `fountain.py`, `baselines.py`.
