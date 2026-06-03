"""
Statistics for the evaluation framework.

Kept small and explicit — paired Wilcoxon for the significance test
(no normality assumption, matches the paired-by-prompt design),
percentile bootstrap for confidence intervals, Cohen's d for effect
size, Benjamini-Hochberg for multiple-comparison correction, and a
manual Pareto frontier for cost/quality/latency trade-offs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PairedTestResult:
    """Result of comparing two configs on the same prompt set."""
    n_pairs: int
    mean_a: float
    mean_b: float
    delta: float                 # mean(b) - mean(a)
    p_value: float               # one-sided "b > a" by default; flipped for cost/latency
    cohen_d: float               # standardized effect size on the deltas
    test_name: str               # "wilcoxon" — currently the only one
    higher_is_better: bool


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean. Returns (low, high)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized effect size for paired samples.

    d = mean(diffs) / std(diffs). Sign matches mean(b) - mean(a).
    """
    diffs = b - a
    if len(diffs) < 2:
        return 0.0
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        # No variance — return inf with the right sign, or 0 if no diff.
        m = float(diffs.mean())
        if m == 0:
            return 0.0
        return math.inf if m > 0 else -math.inf
    return float(diffs.mean() / sd)


def paired_compare(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    *,
    higher_is_better: bool = True,
) -> PairedTestResult:
    """Paired Wilcoxon signed-rank test.

    Tests whether b is *better* than a (one-sided). For metrics where
    smaller is better (cost, latency), pass higher_is_better=False;
    we flip the alternative internally.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired_compare: a/b shape mismatch ({a_arr.shape} vs {b_arr.shape})"
        )
    n = len(a_arr)
    mean_a = float(a_arr.mean()) if n else 0.0
    mean_b = float(b_arr.mean()) if n else 0.0
    delta = mean_b - mean_a

    # Need scipy for the test; degrade gracefully if absent.
    p_value = 1.0
    try:
        from scipy.stats import wilcoxon
        # Drop zero-diff pairs (Wilcoxon's default 'wilcox' handler does
        # this); scipy>=1.9 supports zero_method.
        diffs = b_arr - a_arr
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 1:
            alt = "greater" if higher_is_better else "less"
            try:
                stat = wilcoxon(b_arr, a_arr, alternative=alt, zero_method="wilcox")
                p_value = float(stat.pvalue)
            except ValueError:
                # All differences zero, or otherwise undefined. Leave p=1.0.
                p_value = 1.0
    except ImportError:
        # scipy unavailable — return p=1.0 with a marker
        p_value = float("nan")

    d = cohen_d_paired(a_arr, b_arr) * (1 if higher_is_better else -1)
    return PairedTestResult(
        n_pairs=n,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        p_value=p_value,
        cohen_d=d,
        test_name="wilcoxon",
        higher_is_better=higher_is_better,
    )


def benjamini_hochberg(p_values: list[float], *, alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg correction. Returns a list of bools — True = reject H0.

    Order-preserving wrt the input.
    """
    n = len(p_values)
    if n == 0:
        return []
    idx_sorted = sorted(range(n), key=lambda i: p_values[i])
    sorted_ps = [p_values[i] for i in idx_sorted]
    threshold_idx = -1
    for k in range(n - 1, -1, -1):
        if sorted_ps[k] <= alpha * (k + 1) / n:
            threshold_idx = k
            break
    rejected_sorted = [i <= threshold_idx for i in range(n)]
    rejected = [False] * n
    for sorted_pos, original_pos in enumerate(idx_sorted):
        rejected[original_pos] = rejected_sorted[sorted_pos]
    return rejected


def pareto_frontier(
    points: list[dict],
    *,
    objectives: dict[str, str],
) -> list[str]:
    """Compute the Pareto frontier across a set of points.

    Args:
        points:     list of {"name": str, **metric_values} dicts.
        objectives: {metric_name: "max"|"min"} — direction per metric.

    Returns:
        Names of points on the frontier (no point dominates them).
    """
    if not points:
        return []
    names = [p["name"] for p in points]
    metrics = list(objectives.keys())

    def dominates(p_better, p_worse) -> bool:
        """True if p_better dominates p_worse: better-or-equal on every
        metric, strictly better on at least one."""
        any_strictly_better = False
        for m in metrics:
            direction = objectives[m]
            a = p_better.get(m, float("nan"))
            b = p_worse.get(m, float("nan"))
            if math.isnan(a) or math.isnan(b):
                return False
            if direction == "max":
                if a < b:
                    return False
                if a > b:
                    any_strictly_better = True
            else:
                if a > b:
                    return False
                if a < b:
                    any_strictly_better = True
        return any_strictly_better

    frontier = []
    for i, p in enumerate(points):
        is_dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(names[i])
    return frontier
