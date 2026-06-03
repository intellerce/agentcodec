"""
EvalReport — what `Evaluator.run()` returns.

Carries per-config statistics, pairwise comparisons, the Pareto frontier,
and a one-paragraph recommendation. Serializable to JSON for CI gating
and Markdown for review.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .stats import (
    benjamini_hochberg,
    paired_compare,
)

# Direction of "better" per metric — used by Pareto + paired tests.
_OBJECTIVES = {
    "quality": "max",
    "cost_usd": "min",
    "latency_s": "min",
}


@dataclass
class ConfigStats:
    """Aggregated per-config metrics."""
    name: str
    n_runs: int
    n_errors: int
    quality_mean: float
    quality_ci95: tuple[float, float]
    cost_usd_mean: float
    cost_usd_p50: float
    cost_usd_p95: float
    latency_s_mean: float
    latency_s_p50: float
    latency_s_p95: float
    thinking_call_rate: float          # fraction of calls where thinking emitted
    cost_source_breakdown: dict[str, float]
    weakest_cost_tier: str
    judge_model: str
    technique_distribution: dict[str, int]   # for routed configs: how often each technique was picked
    per_category_quality: dict[str, float]   # mean quality per task category

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quality_ci95"] = list(self.quality_ci95)
        return d


@dataclass
class PairwiseComparison:
    """One row of the pairwise-significance table."""
    config_a: str
    config_b: str                # b - a is the delta
    metric: str                  # "quality" | "cost_usd" | "latency_s"
    n_pairs: int
    delta: float
    p_value: float
    p_value_bh: float            # BH-corrected p across the comparison set
    cohen_d: float
    significant: bool            # True if p_value_bh < alpha
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """The output of an Evaluator run."""
    configs: list[ConfigStats]
    pairwise: list[PairwiseComparison] = field(default_factory=list)
    pareto: dict[str, list[str]] = field(default_factory=dict)   # objective_set → frontier names
    recommendation: str = ""
    methodology: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    # Raw per-(config, prompt, repeat) records — kept for downstream tools
    # (e.g., training SemKNN from this report). Stripped from console summary.
    raw_records: list[dict[str, Any]] = field(default_factory=list)

    # ----- Decision helpers -----

    def winner(self, metric: str = "quality") -> str | None:
        """Return the name of the config with the best mean for `metric`.

        For "quality", higher = better. For "cost_usd" / "latency_s", lower.
        Returns None if no configs have data.
        """
        if not self.configs:
            return None
        if metric == "quality":
            return max(self.configs, key=lambda c: c.quality_mean).name
        if metric == "cost_usd":
            return min(self.configs, key=lambda c: c.cost_usd_mean).name
        if metric == "latency_s":
            return min(self.configs, key=lambda c: c.latency_s_mean).name
        if metric == "cost_per_quality_unit":
            def ratio(c: ConfigStats) -> float:
                if c.quality_mean <= 0:
                    return float("inf")
                return c.cost_usd_mean / c.quality_mean
            return min(self.configs, key=ratio).name
        raise ValueError(f"Unknown metric: {metric!r}")

    def is_significant(
        self,
        config_b: str,
        config_a: str,
        *,
        metric: str = "quality",
        alpha: float = 0.05,
    ) -> bool:
        """True if the BH-corrected paired test rejects H0 (no difference)."""
        for row in self.pairwise:
            if row.config_a == config_a and row.config_b == config_b and row.metric == metric:
                return row.significant and row.p_value_bh < alpha
        return False

    def gate(
        self,
        baseline: str,
        candidate: str,
        *,
        metric: str = "quality",
        max_regression: float = 0.0,
        alpha: float = 0.05,
    ) -> tuple[bool, str]:
        """CI gating: should `candidate` be allowed to ship vs `baseline`?

        Returns (passed, reason). Pass = no significant regression worse than
        max_regression on the chosen metric.
        """
        b_stats = next((c for c in self.configs if c.name == candidate), None)
        a_stats = next((c for c in self.configs if c.name == baseline), None)
        if not b_stats or not a_stats:
            return False, f"missing config(s): baseline={baseline!r}, candidate={candidate!r}"

        if metric == "quality":
            delta = b_stats.quality_mean - a_stats.quality_mean
            sig = self.is_significant(candidate, baseline, metric=metric, alpha=alpha)
            if delta < -abs(max_regression) and sig:
                return False, (
                    f"{candidate} quality {b_stats.quality_mean:.3f} regressed "
                    f"vs {baseline} {a_stats.quality_mean:.3f} (Δ={delta:+.3f}, "
                    f"max allowed={-abs(max_regression):+.3f}, p<{alpha}) — block"
                )
            return True, f"OK: Δ quality = {delta:+.3f} (max regression allowed = {-abs(max_regression):+.3f})"
        # cost / latency: regression = increase
        attr = {"cost_usd": "cost_usd_mean", "latency_s": "latency_s_mean"}.get(metric)
        if not attr:
            return False, f"unknown metric: {metric!r}"
        delta = getattr(b_stats, attr) - getattr(a_stats, attr)
        sig = self.is_significant(candidate, baseline, metric=metric, alpha=alpha)
        if delta > abs(max_regression) and sig:
            return False, (
                f"{candidate} {metric} regressed vs {baseline} (Δ={delta:+.4f}, "
                f"max allowed=+{abs(max_regression):.4f}, p<{alpha}) — block"
            )
        return True, f"OK: Δ {metric} = {delta:+.4f}"

    # ----- Output formats -----

    def summary(self) -> str:
        """Render the console summary. Prints to stdout AND returns the text."""
        lines: list[str] = []
        lines.append("=" * 88)
        lines.append("AgentCodec Evaluation Report")
        lines.append("=" * 88)
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        if self.warnings:
            lines.append("")

        # Per-config table
        header = f"{'Config':<20s} {'Quality (95% CI)':<24s} {'Cost':>10s} {'Lat p50/p95':>14s} {'Errors':>8s} {'N':>6s}"
        lines.append(header)
        lines.append("-" * len(header))
        for c in self.configs:
            ci = f"{c.quality_mean:.3f} [{c.quality_ci95[0]:.3f}, {c.quality_ci95[1]:.3f}]"
            lat = f"{c.latency_s_p50:.2f}/{c.latency_s_p95:.2f}s"
            lines.append(
                f"{c.name:<20s} {ci:<24s} ${c.cost_usd_mean:>9.4f} {lat:>14s} "
                f"{c.n_errors:>4d}/{c.n_runs:<3d} {c.n_runs:>6d}"
            )
        lines.append("")

        # Pairwise (if any)
        if self.pairwise:
            lines.append("Pairwise comparisons (paired Wilcoxon, BH-corrected):")
            lines.append(
                f"  {'A':<14s} → {'B':<14s} {'Metric':<12s} {'Δ':>10s} {'p (BH)':>10s} {'sig':>5s}"
            )
            for row in self.pairwise:
                sig = "**" if row.significant else "  "
                lines.append(
                    f"  {row.config_a:<14s} → {row.config_b:<14s} "
                    f"{row.metric:<12s} {row.delta:>+10.4f} "
                    f"{row.p_value_bh:>10.4f} {sig:>5s}"
                )
            lines.append("")

        # Pareto
        if self.pareto:
            for objs, frontier in self.pareto.items():
                lines.append(f"Pareto frontier across [{objs}]:  {', '.join(frontier)}")
            lines.append("")

        # Recommendation
        if self.recommendation:
            lines.append("Recommendation:")
            for line in self.recommendation.splitlines():
                lines.append(f"  {line}")
            lines.append("")

        # Methodology footer
        lines.append("-" * 88)
        if self.methodology:
            for k, v in self.methodology.items():
                lines.append(f"  {k}: {v}")
        lines.append("=" * 88)
        text = "\n".join(lines)
        print(text)
        return text

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON. Writes to file when `path` is given."""
        d = {
            "generated_at": self.generated_at,
            "configs": [c.to_dict() for c in self.configs],
            "pairwise": [r.to_dict() for r in self.pairwise],
            "pareto": dict(self.pareto),
            "recommendation": self.recommendation,
            "methodology": self.methodology,
            "warnings": list(self.warnings),
        }
        text = json.dumps(d, indent=2)
        if path:
            Path(path).write_text(text)
        return text

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a Markdown report (always includes raw stats; methodology footer)."""
        md: list[str] = []
        md.append("# AgentCodec Evaluation Report")
        md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.generated_at))}_")
        md.append("")
        if self.warnings:
            md.append("## Warnings")
            for w in self.warnings:
                md.append(f"- ⚠️ {w}")
            md.append("")

        md.append("## Per-config summary")
        md.append("")
        md.append("| Config | Quality (95% CI) | Cost ($) | Latency p50/p95 (s) | Errors | N | Judge | Weakest cost tier |")
        md.append("|---|---|---:|---:|---:|---:|---|---|")
        for c in self.configs:
            md.append(
                f"| `{c.name}` "
                f"| {c.quality_mean:.3f} [{c.quality_ci95[0]:.3f}, {c.quality_ci95[1]:.3f}] "
                f"| {c.cost_usd_mean:.5f} "
                f"| {c.latency_s_p50:.2f} / {c.latency_s_p95:.2f} "
                f"| {c.n_errors}/{c.n_runs} "
                f"| {c.n_runs} "
                f"| {c.judge_model} "
                f"| {c.weakest_cost_tier} |"
            )
        md.append("")

        # Per-category breakdown if available
        if any(c.per_category_quality for c in self.configs):
            md.append("### Quality by category")
            md.append("")
            cats = sorted({k for c in self.configs for k in c.per_category_quality.keys()})
            md.append("| Config | " + " | ".join(cats) + " |")
            md.append("|---|" + "|".join("---:" for _ in cats) + "|")
            for c in self.configs:
                row = [f"`{c.name}`"]
                for cat in cats:
                    v = c.per_category_quality.get(cat)
                    row.append(f"{v:.3f}" if v is not None else "—")
                md.append("| " + " | ".join(row) + " |")
            md.append("")

        # Technique distribution (if any routed configs)
        if any(c.technique_distribution for c in self.configs):
            md.append("### Technique distribution (routed configs)")
            md.append("")
            for c in self.configs:
                if c.technique_distribution:
                    md.append(f"- `{c.name}`:")
                    total = sum(c.technique_distribution.values())
                    for tech, cnt in sorted(c.technique_distribution.items(), key=lambda kv: -kv[1]):
                        pct = 100 * cnt / total if total else 0
                        md.append(f"  - `{tech}`: {cnt} ({pct:.1f}%)")
            md.append("")

        # Pairwise
        if self.pairwise:
            md.append("## Pairwise significance (paired Wilcoxon, BH-corrected)")
            md.append("")
            md.append("| A | B | Metric | Δ (B-A) | p (BH) | Cohen's d | Significant |")
            md.append("|---|---|---|---:|---:|---:|:---:|")
            for r in self.pairwise:
                sig = "✓" if r.significant else ""
                md.append(
                    f"| `{r.config_a}` | `{r.config_b}` | {r.metric} "
                    f"| {r.delta:+.4f} | {r.p_value_bh:.4f} | {r.cohen_d:+.2f} | {sig} |"
                )
            md.append("")

        # Pareto
        if self.pareto:
            md.append("## Pareto frontier")
            md.append("")
            for objs, frontier in self.pareto.items():
                md.append(f"- **{objs}**: " + ", ".join(f"`{n}`" for n in frontier))
            md.append("")

        # Recommendation
        if self.recommendation:
            md.append("## Recommendation")
            md.append("")
            md.append(self.recommendation)
            md.append("")

        # Methodology
        if self.methodology:
            md.append("## Methodology")
            md.append("")
            for k, v in self.methodology.items():
                md.append(f"- **{k}**: {v}")
            md.append("")

        md.append("---")
        md.append("_Generated by [AgentCodec](https://github.com) `agentcodec eval`._")

        text = "\n".join(md)
        if path:
            Path(path).write_text(text)
        return text


# ---------------------------------------------------------------------------
# Builder helpers (used by Evaluator)
# ---------------------------------------------------------------------------

def build_pairwise(
    records_by_config: dict[str, list[dict]],
    *,
    metrics: list[str] = ("quality", "cost_usd", "latency_s"),
    baseline: str | None = None,
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Build the pairwise comparison list with BH correction.

    When baseline is set, only baseline-vs-others rows are produced.
    Otherwise, all-vs-all (capped to keep the table readable).
    """
    names = list(records_by_config.keys())
    if not baseline:
        if len(names) > 4:
            # Avoid overwhelming the user; pick the first as implicit baseline
            baseline = names[0]
    pairs: list[tuple[str, str]] = []
    if baseline:
        pairs = [(baseline, b) for b in names if b != baseline]
    else:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.append((a, b))

    rows: list[PairwiseComparison] = []
    raw_p_by_metric: dict[str, list[float]] = {m: [] for m in metrics}
    rows_by_metric: dict[str, list[PairwiseComparison]] = {m: [] for m in metrics}

    for (a, b) in pairs:
        ra = records_by_config.get(a, [])
        rb = records_by_config.get(b, [])
        # Pair by (prompt_id, repeat_idx)
        ra_idx = {(r["prompt_id"], r["repeat_idx"]): r for r in ra if not r.get("error")}
        rb_idx = {(r["prompt_id"], r["repeat_idx"]): r for r in rb if not r.get("error")}
        common = sorted(ra_idx.keys() & rb_idx.keys())
        if not common:
            continue
        for metric in metrics:
            higher_is_better = (metric == "quality")
            a_vals = [ra_idx[k][metric] for k in common]
            b_vals = [rb_idx[k][metric] for k in common]
            res = paired_compare(a_vals, b_vals, higher_is_better=higher_is_better)
            row = PairwiseComparison(
                config_a=a, config_b=b, metric=metric,
                n_pairs=res.n_pairs, delta=res.delta,
                p_value=res.p_value, p_value_bh=res.p_value,  # corrected below
                cohen_d=res.cohen_d, significant=False,
                higher_is_better=higher_is_better,
            )
            raw_p_by_metric[metric].append(res.p_value)
            rows_by_metric[metric].append(row)
            rows.append(row)

    # BH-correct per metric (independent families).
    for metric, ps in raw_p_by_metric.items():
        if not ps:
            continue
        ps_clean = [0.5 if p != p else p for p in ps]   # NaN→0.5 if scipy missing
        rejected = benjamini_hochberg(ps_clean, alpha=alpha)
        for i, row in enumerate(rows_by_metric[metric]):
            row.p_value_bh = ps_clean[i]
            row.significant = rejected[i]
    return rows
