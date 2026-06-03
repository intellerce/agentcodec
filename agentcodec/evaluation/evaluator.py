"""
Evaluator — runs N candidate configurations on a shared prompt set and
produces an EvalReport.

Per-(config, prompt, repeat) caching means a kill/resume is safe; results
already on disk are loaded and skipped.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..api import ReliabilityModule
from ..config import JudgeConfig, LibraryConfig
from .report import (
    ConfigStats,
    EvalReport,
    build_pairwise,
)
from .stats import bootstrap_ci, pareto_frontier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_config(c: str | Path | LibraryConfig | dict, name: str) -> LibraryConfig:
    if isinstance(c, LibraryConfig):
        return c
    if isinstance(c, dict):
        return LibraryConfig.from_dict(c)
    if isinstance(c, (str, Path)):
        return LibraryConfig.from_yaml(c)
    raise TypeError(
        f"configs[{name!r}]: expected str/Path/dict/LibraryConfig, got {type(c).__name__}"
    )


def _override_judge(cfg: LibraryConfig, eval_judge: JudgeConfig | dict | None) -> LibraryConfig:
    """Return a new LibraryConfig with the judge replaced (when eval_judge set)."""
    if eval_judge is None:
        return cfg
    if isinstance(eval_judge, dict):
        eval_judge = JudgeConfig.model_validate(eval_judge)
    # pydantic v2 model_copy update path
    return cfg.model_copy(update={"judge": eval_judge})


def _load_prompts(
    prompts: list[dict[str, Any]] | None,
    prompts_file: str | Path | None,
) -> list[dict[str, Any]]:
    if prompts is not None and prompts_file is not None:
        raise ValueError("Pass exactly one of `prompts` or `prompts_file`")
    if prompts is not None:
        out = list(prompts)
    elif prompts_file is not None:
        out = []
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(json.loads(line))
    else:
        raise ValueError("Either `prompts` or `prompts_file` is required")

    # Auto-fill missing IDs.
    for i, r in enumerate(out):
        if "id" not in r:
            r["id"] = f"eval-{i:06d}"
        if "prompt" not in r:
            raise ValueError(f"Prompt record {i}: missing 'prompt' field")
    return out


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Compare 2-N deployment configurations on a shared evaluation prompt set."""

    def __init__(
        self,
        configs: dict[str, str | Path | LibraryConfig | dict],
        *,
        prompts: list[dict[str, Any]] | None = None,
        prompts_file: str | Path | None = None,
        repeats: int = 1,
        parallel_prompts: int = 1,
        eval_judge: JudgeConfig | dict | None = None,
        cache_dir: str | Path = "results/eval",
        on_error: str = "continue",          # "continue" | "raise" | "fallback_baseline"
        max_retries: int = 0,                 # retry a failed run this many times before giving up
        warm: bool = True,                    # construct ReliabilityModules eagerly
    ) -> None:
        if len(configs) < 2:
            raise ValueError(
                f"Evaluator needs at least 2 configs to compare; got {len(configs)}"
            )
        self.config_names = list(configs.keys())
        self.raw_configs = {n: _coerce_config(c, n) for n, c in configs.items()}

        # If eval_judge set, override every config's judge for fair comparison.
        if eval_judge is not None:
            self.configs = {
                n: _override_judge(c, eval_judge) for n, c in self.raw_configs.items()
            }
            self._judge_overridden = True
        else:
            self.configs = dict(self.raw_configs)
            self._judge_overridden = False

        self.prompts = _load_prompts(prompts, prompts_file)
        self.repeats = max(1, int(repeats))
        self.parallel_prompts = max(1, int(parallel_prompts))
        self.on_error = on_error
        self.max_retries = max(0, int(max_retries))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cross-config-judge warning
        self._warnings: list[str] = []
        if not self._judge_overridden:
            judges = {n: c.judge.model for n, c in self.configs.items()}
            distinct = set(judges.values())
            if len(distinct) > 1:
                self._warnings.append(
                    f"Configs use different judges {judges} — quality scores are not "
                    f"directly comparable across configs. Pass `eval_judge` to fix."
                )

        # Warn when cost-source tiers will mismatch (pricing comparison is unsound).
        from ..cost import CostSource, resolve_rate
        weakest_per_config: dict[str, str] = {}
        for n, c in self.configs.items():
            worst = CostSource.EXACT_USER_RATE
            for m in c.models:
                override = (m.cost_per_1m.input, m.cost_per_1m.output) if m.cost_per_1m else None
                _, src = resolve_rate(m.model, user_override=override)
                if src.rank > worst.rank:
                    worst = src
            weakest_per_config[n] = worst.value
        if len(set(weakest_per_config.values())) > 1:
            self._warnings.append(
                f"Configs have different cost-source tiers {weakest_per_config} — "
                f"cost comparison is sound only within a tier. Set `cost_per_1m` "
                f"for every model to make all tiers `exact_user_rate`."
            )

        # Lazy-warm modules (heavy: BGE encoders eager-load).
        self._modules: dict[str, ReliabilityModule | None] = {n: None for n in self.config_names}
        if warm:
            self._warm_modules()

    def _warm_modules(self) -> None:
        for n, cfg in self.configs.items():
            if self._modules[n] is None:
                logger.info(f"[evaluator] warming module: {n}")
                self._modules[n] = ReliabilityModule(cfg)

    # ----- main entry point -----

    def run(
        self,
        *,
        baseline: str | None = None,
        alpha: float = 0.05,
        progress_callback: "Callable[[dict[str, Any]], None] | None" = None,
    ) -> EvalReport:
        """Evaluate every config × prompt × repeat and return an EvalReport.

        ``progress_callback``, when given, is invoked after each completed run
        (success, error, or cache-hit during warm-up replay) with a small dict::

            {"completed": int, "total": int, "config": str,
             "record": dict, "cached": bool}

        ``record`` is the per-run record (carries ``quality``, ``cost_usd``,
        ``error``, …), so a caller can render a live progress bar with running
        intermediate results. Exceptions raised by the callback are swallowed
        so a flaky UI never aborts the eval.
        """
        self._warm_modules()
        records_by_config: dict[str, list[dict[str, Any]]] = {n: [] for n in self.config_names}

        total = len(self.config_names) * len(self.prompts) * self.repeats
        completed = 0

        for cfg_name in self.config_names:
            cache_path = self.cache_dir / f"{cfg_name}.jsonl"
            existing = self._load_cached(cache_path)
            done_keys = {(r["prompt_id"], r["repeat_idx"]) for r in existing}
            records_by_config[cfg_name] = list(existing)

            mod = self._modules[cfg_name]
            if mod is None:
                raise RuntimeError(
                    f"config {cfg_name!r} was not warmed before run(); "
                    f"_warm_modules() should have populated self._modules"
                )

            todo = []
            for p in self.prompts:
                for rep in range(self.repeats):
                    if (p["id"], rep) not in done_keys:
                        todo.append((p, rep))

            logger.info(
                f"[evaluator] config={cfg_name}: "
                f"{len(existing)} cached, {len(todo)} remaining"
            )
            completed += len(existing)

            def _emit(record: dict[str, Any] | None, cached: bool) -> None:
                """Fire the user's progress_callback; never let it abort the run."""
                if progress_callback is None:
                    return
                try:
                    progress_callback({
                        "completed": completed,
                        "total": total,
                        "config": cfg_name,
                        "record": record,
                        "cached": cached,
                    })
                except Exception:  # pragma: no cover - UI must not break eval
                    logger.debug("progress_callback raised; ignoring", exc_info=True)

            # Reflect any cache-replayed runs in the progress position up front.
            if existing:
                _emit(existing[-1], cached=True)

            def _one(prompt_record: dict, repeat_idx: int) -> dict[str, Any] | None:
                # Retry the whole run up to max_retries on any exception (covers
                # timeouts the channel's own backoff couldn't recover) before
                # falling back to the on_error policy.
                for attempt in range(self.max_retries + 1):
                    try:
                        res = mod.run(
                            prompt_record["prompt"],
                            category=prompt_record.get("category"),
                            reference=prompt_record.get("reference"),
                            score_mode=prompt_record.get("score_mode"),
                            metadata=prompt_record.get("metadata"),
                            task_id=str(prompt_record["id"]),
                            return_trace=True,
                        )
                        return self._record_from_result(prompt_record, repeat_idx, res)
                    except Exception as e:
                        if attempt < self.max_retries:
                            logger.warning(
                                "[evaluator] %s rep%d failed (attempt %d/%d): %r; retrying",
                                prompt_record["id"], repeat_idx,
                                attempt + 1, self.max_retries + 1, e,
                            )
                            continue
                        if self.on_error == "raise":
                            raise
                        return self._error_record(prompt_record, repeat_idx, e)
                return None  # unreachable

            if self.parallel_prompts == 1:
                for p, rep in todo:
                    rec = _one(p, rep)
                    if rec is not None:
                        records_by_config[cfg_name].append(rec)
                        self._append_cache(cache_path, rec)
                        completed += 1
                        _emit(rec, cached=False)
                        if completed % 10 == 0 or completed == total:
                            logger.info(f"[evaluator] {completed}/{total} runs complete")
            else:
                with ThreadPoolExecutor(
                    max_workers=self.parallel_prompts,
                    thread_name_prefix=f"eval-{cfg_name}",
                ) as ex:
                    futs = {ex.submit(_one, p, rep): (p, rep) for p, rep in todo}
                    for fut in as_completed(futs):
                        rec = fut.result()
                        if rec is not None:
                            records_by_config[cfg_name].append(rec)
                            self._append_cache(cache_path, rec)
                            completed += 1
                            _emit(rec, cached=False)
                            if completed % 10 == 0 or completed == total:
                                logger.info(f"[evaluator] {completed}/{total} runs complete")

        # Build per-config stats
        config_stats = [
            self._stats_for(name, records_by_config[name]) for name in self.config_names
        ]

        # Pairwise tests (paired Wilcoxon + BH per metric family)
        pairwise = build_pairwise(
            records_by_config,
            metrics=["quality", "cost_usd", "latency_s"],
            baseline=baseline,
            alpha=alpha,
        )

        # Pareto frontier
        pareto: dict[str, list[str]] = {}
        points = [
            {
                "name": c.name,
                "quality": c.quality_mean,
                "cost_usd": c.cost_usd_mean,
                "latency_s": c.latency_s_mean,
            }
            for c in config_stats
        ]
        pareto["quality vs cost"] = pareto_frontier(
            points, objectives={"quality": "max", "cost_usd": "min"},
        )
        pareto["quality vs latency"] = pareto_frontier(
            points, objectives={"quality": "max", "latency_s": "min"},
        )
        pareto["cost vs latency"] = pareto_frontier(
            points, objectives={"cost_usd": "min", "latency_s": "min"},
        )

        # Recommendation
        recommendation = self._build_recommendation(config_stats, pairwise, pareto)

        # Methodology footer
        methodology = {
            "n_prompts": len(self.prompts),
            "repeats": self.repeats,
            "judge_overridden": self._judge_overridden,
            "alpha": alpha,
            "test": "paired Wilcoxon signed-rank with Benjamini-Hochberg correction",
            "ci": "percentile bootstrap, 95%, 2000 samples",
            "pareto_objectives": list(pareto.keys()),
            "cache_dir": str(self.cache_dir),
        }

        # Flatten raw records (all configs)
        raw_records = [
            dict(r, config=name)
            for name, recs in records_by_config.items() for r in recs
        ]

        return EvalReport(
            configs=config_stats,
            pairwise=pairwise,
            pareto=pareto,
            recommendation=recommendation,
            methodology=methodology,
            warnings=list(self._warnings),
            raw_records=raw_records,
        )

    # ----- record / cache helpers -----

    def _record_from_result(
        self, prompt_record: dict, repeat_idx: int, res
    ) -> dict[str, Any]:
        trace = res.trace or {}
        totals = trace.get("totals", {})
        thinking_calls = sum(
            1 for c in trace.get("calls", []) if c.get("thinking", {}).get("emitted")
        )
        n_calls = max(1, totals.get("num_llm_calls", 0))
        return {
            "prompt_id": str(prompt_record["id"]),
            "repeat_idx": repeat_idx,
            "category": prompt_record.get("category", "qa"),
            "technique_used": res.technique_used,
            "quality": float(trace.get("final_quality") or 0.0),
            "cost_usd": float(res.cost_usd),
            "cost_source": res.cost_source,
            "latency_s": float(res.latency_s),
            "cumulative_latency_s": float(res.cumulative_latency_s),
            "thinking_used": bool(res.thinking_used),
            "thinking_call_share": thinking_calls / n_calls,
            "num_llm_calls": int(totals.get("num_llm_calls", 0)),
            "input_tokens": int(totals.get("input_tokens", 0)),
            "output_tokens": int(totals.get("output_tokens", 0)),
            "thinking_tokens": int(totals.get("thinking_tokens", 0)),
            "judge_cost_usd": float(totals.get("judge_cost_usd", 0.0)),
            "router_chosen": (trace.get("router") or {}).get("chosen"),
            "error": res.error,
            "answer_text_preview": (res.text or "")[:200],
        }

    def _error_record(self, prompt_record: dict, repeat_idx: int, e: Exception) -> dict[str, Any]:
        return {
            "prompt_id": str(prompt_record["id"]),
            "repeat_idx": repeat_idx,
            "category": prompt_record.get("category", "qa"),
            "technique_used": None,
            "quality": 0.0,
            "cost_usd": 0.0,
            "cost_source": None,
            "latency_s": 0.0,
            "cumulative_latency_s": 0.0,
            "thinking_used": False,
            "thinking_call_share": 0.0,
            "num_llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "judge_cost_usd": 0.0,
            "router_chosen": None,
            "error": f"{type(e).__name__}: {e}",
            "answer_text_preview": "",
        }

    def _stats_for(self, name: str, records: list[dict[str, Any]]) -> ConfigStats:
        ok = [r for r in records if not r.get("error")]
        n_runs = len(records)
        n_errors = n_runs - len(ok)
        if not ok:
            return ConfigStats(
                name=name, n_runs=n_runs, n_errors=n_errors,
                quality_mean=0.0, quality_ci95=(0.0, 0.0),
                cost_usd_mean=0.0, cost_usd_p50=0.0, cost_usd_p95=0.0,
                latency_s_mean=0.0, latency_s_p50=0.0, latency_s_p95=0.0,
                thinking_call_rate=0.0,
                cost_source_breakdown={},
                weakest_cost_tier="n/a",
                judge_model=self.configs[name].judge.model,
                technique_distribution={},
                per_category_quality={},
            )
        qs = np.array([r["quality"] for r in ok])
        cs = np.array([r["cost_usd"] for r in ok])
        ls = np.array([r["latency_s"] for r in ok])
        # Aggregated cost-source breakdown across runs
        breakdown: dict[str, float] = defaultdict(float)
        worst_tier = "exact_user_rate"
        from ..cost import CostSource
        for r in ok:
            src = r.get("cost_source") or "exact_table_rate"
            breakdown[src] += r["cost_usd"]
            try:
                if CostSource(src).rank > CostSource(worst_tier).rank:
                    worst_tier = src
            except ValueError:
                pass
        # Technique distribution (when routed)
        tech_counts = Counter(r["technique_used"] for r in ok if r["technique_used"])
        # Per-category quality
        per_cat: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            per_cat[r["category"]].append(r["quality"])
        per_cat_mean = {c: float(np.mean(v)) for c, v in per_cat.items()}

        return ConfigStats(
            name=name,
            n_runs=n_runs,
            n_errors=n_errors,
            quality_mean=float(qs.mean()),
            quality_ci95=bootstrap_ci(qs.tolist()),
            cost_usd_mean=float(cs.mean()),
            cost_usd_p50=float(np.percentile(cs, 50)),
            cost_usd_p95=float(np.percentile(cs, 95)),
            latency_s_mean=float(ls.mean()),
            latency_s_p50=float(np.percentile(ls, 50)),
            latency_s_p95=float(np.percentile(ls, 95)),
            thinking_call_rate=float(np.mean([r["thinking_call_share"] for r in ok])),
            cost_source_breakdown=dict(breakdown),
            weakest_cost_tier=worst_tier,
            judge_model=self.configs[name].judge.model,
            technique_distribution=dict(tech_counts),
            per_category_quality=per_cat_mean,
        )

    def _build_recommendation(
        self,
        configs: list[ConfigStats],
        pairwise: list,
        pareto: dict[str, list[str]],
    ) -> str:
        if not configs:
            return "(no configs evaluated)"
        best_q = max(configs, key=lambda c: c.quality_mean)
        cheap = min(configs, key=lambda c: c.cost_usd_mean)
        fastest = min(configs, key=lambda c: c.latency_s_mean)

        lines = []
        lines.append(
            f"Highest quality: **{best_q.name}** "
            f"(q={best_q.quality_mean:.3f}, ${best_q.cost_usd_mean:.5f}/call, "
            f"{best_q.latency_s_p50:.2f}s p50)"
        )
        if cheap.name != best_q.name:
            lines.append(
                f"Cheapest:        **{cheap.name}** "
                f"(q={cheap.quality_mean:.3f}, ${cheap.cost_usd_mean:.5f}/call)"
            )
        if fastest.name != best_q.name and fastest.name != cheap.name:
            lines.append(
                f"Fastest:         **{fastest.name}** "
                f"(p50={fastest.latency_s_p50:.2f}s, q={fastest.quality_mean:.3f})"
            )

        # Pareto frontier — which configs are dominated?
        all_names = {c.name for c in configs}
        all_pareto = set()
        for f in pareto.values():
            all_pareto.update(f)
        dominated = sorted(all_names - all_pareto)
        if dominated:
            lines.append(f"Dominated (avoid): {', '.join(dominated)}")

        # Significance flag if winner is significantly better than runner-up.
        if len(configs) >= 2 and self._judge_overridden:
            sorted_q = sorted(configs, key=lambda c: -c.quality_mean)
            top, second = sorted_q[0], sorted_q[1]
            for row in pairwise:
                if (
                    row.metric == "quality"
                    and {row.config_a, row.config_b} == {top.name, second.name}
                ):
                    sig = "significant" if row.significant else "not significant"
                    lines.append(
                        f"Winner significance: {top.name} vs {second.name} "
                        f"Δq={top.quality_mean - second.quality_mean:+.3f} "
                        f"(p_BH={row.p_value_bh:.3f}, {sig})"
                    )
                    break
        return "\n".join(lines)

    # ----- cache I/O (JSONL, one line per record) -----

    def _load_cached(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
        except Exception as e:
            logger.warning(f"[evaluator] failed to load cache {path}: {e}")
        return out

    def _append_cache(self, path: Path, record: dict[str, Any]) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"[evaluator] failed to append cache {path}: {e}")
