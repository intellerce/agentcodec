"""
Plotting module — generates all key figures for the paper.

Produces publication-quality plots with:
- 95% bootstrap confidence intervals on all means
- Significance annotations (Wilcoxon signed-rank vs baseline)
- Consistent professional styling (Nature/NeurIPS conventions)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-GUI backend — TkAgg's Tcl objects crash when
# garbage-collected from worker threads (parallel_tasks > 1)
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Professional matplotlib style (NeurIPS / Nature conventions)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Figure
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Axes
    "axes.facecolor": "white",
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Grid
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    # Lines
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

# ---------------------------------------------------------------------------
# Colour palette — perceptually uniform, colourblind-friendly
# Inspired by Tol's "bright" palette + a few additions
# ---------------------------------------------------------------------------
TECHNIQUE_COLORS = {
    "baseline":       "#BBBBBB",
    "baseline_thinking": "#777777",
    "diversity_sc":   "#4477AA",
    "diversity_mrc":  "#228833",
    "diversity_egc":  "#EE6677",
    "harq_cc":        "#CCBB44",
    "harq_ir":        "#AA3377",
    "turbo":          "#66CCEE",
    "fountain":       "#0077BB",
    "fec":            "#CC6633",
    "fec_1.0":        "#BBBBBB",
    "fec_0.75":       "#EE9933",
    "fec_0.5":        "#CC6633",
    "fec_0.50":       "#CC6633",
    "fec_0.33":       "#882255",
    "fec_0.25":       "#332288",
    "acm":            "#117733",
    # ACM sub-techniques
    "acm_uncoded":    "#BBBBBB",
    "acm_fec_r0.75":  "#EE9933",
    "acm_fec_r0.5":   "#CC6633",
    "acm_harq_ir":    "#AA3377",
    "acm_diversity_mrc": "#228833",
    # Soft-output techniques (logprob-based)
    "diversity_mrc_soft": "#4477AA",
    "fountain_soft":      "#005599",
    "acm_soft":           "#119955",
    # Our wider-pool multi-model diversity operators (not prior methods)
    "diversity_sc_N":                  "#AACCEE",
    "diversity_mrc_discrete_N":        "#6699CC",
    # Prior-method reproductions (matched-budget, canonical single-model)
    "self_consistency":      "#DDAA33",
    "self_refine":           "#CC99CC",
    "chain_of_verification": "#88CCAA",
    "best_of_n":             "#77AADD",
    "weighted_bon":          "#446699",
    "mixture_of_agents":     "#EEBB77",
    "cisc":                  "#994455",
    "acm_learned":           "#CC7788",
}

TECHNIQUE_LABELS = {
    "baseline":       "Baseline (uncoded)",
    "baseline_thinking": "Baseline (thinking)",
    "diversity_sc":   "Diversity-SC",
    "diversity_mrc":  "Diversity-MRC",
    "diversity_egc":  "Diversity-EGC",
    "harq_cc":        "HARQ-CC",
    "harq_ir":        "HARQ-IR",
    "turbo":          "Turbo decoder",
    "fountain":       "Fountain code",
    "fec":            "FEC",
    "fec_1.0":        "FEC (r=1, uncoded)",
    "fec_0.75":       "FEC (r=3/4)",
    "fec_0.5":        "FEC (r=1/2)",
    "fec_0.50":       "FEC (r=1/2)",
    "fec_0.33":       "FEC (r=1/3)",
    "fec_0.25":       "FEC (r=1/4)",
    "acm":            "ACM (adaptive)",
    "diversity_mrc_soft": "Diversity-MRC (soft)",
    "fountain_soft":      "Fountain (soft)",
    "acm_soft":           "ACM (soft CQI)",
    "diversity_sc_N":                  "Diversity-SC-N (multi-model, N=5)",
    "diversity_mrc_discrete_N":        "Diversity-MRC-Discrete-N (multi-model, N=5)",
    "self_consistency":      "Self-Consistency",
    "self_refine":           "Self-Refine",
    "chain_of_verification": "Chain-of-Verification",
    "best_of_n":             "Best-of-N (single-model, N=5)",
    "weighted_bon":          "Weighted BoN (single-model)",
    "mixture_of_agents":     "Mixture-of-Agents",
    "cisc":                  "CISC",
    "acm_learned":           "ACM (learned)",
}

TECHNIQUE_MARKERS = {
    "baseline":       "X",
    "baseline_thinking": "x",
    "diversity_sc":   "o",
    "diversity_mrc":  "s",
    "diversity_egc":  "^",
    "harq_cc":        "D",
    "harq_ir":        "v",
    "turbo":          "P",
    "fountain":       "*",
    "fec":            "H",
    "fec_1.0":        "X",
    "fec_0.75":       "h",
    "fec_0.5":        "H",
    "fec_0.50":       "H",
    "fec_0.33":       "p",
    "fec_0.25":       "8",
    "acm":            "d",
    "diversity_mrc_soft": "s",
    "fountain_soft":      "*",
    "acm_soft":           "d",
    "diversity_sc_N":                  "8",
    "diversity_mrc_discrete_N":        "p",
    "self_consistency":      "<",
    "self_refine":           ">",
    "chain_of_verification": "X",
    "best_of_n":             "D",
    "weighted_bon":          "d",
    "mixture_of_agents":     "H",
    "cisc":                  "p",
    "acm_learned":           "d",
}

# Consistent ordering for legends / bar charts
TECHNIQUE_ORDER = [
    "baseline",
    "baseline_thinking",
    "diversity_sc", "diversity_mrc", "diversity_egc",
    "harq_cc", "harq_ir",
    "turbo",
    "fountain",
    "fec_0.75", "fec_0.50", "fec_0.33",
    "acm", "acm_learned",
    # Soft-output (logprob-based) variants
    "diversity_mrc_soft", "fountain_soft", "acm_soft",
    # Our wider-pool multi-model diversity operators
    "diversity_sc_N", "diversity_mrc_discrete_N",
    # Prior-method reproductions (matched-budget, canonical single-model)
    "self_consistency", "self_refine", "chain_of_verification",
    "best_of_n", "weighted_bon", "mixture_of_agents", "cisc",
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _model_info_text(config: dict | None) -> str:
    """Build a one-line model info string from benchmark config."""
    if not config:
        return ""
    models = config.get("models", [])
    model_names = [m["model"] if isinstance(m, dict) else m for m in models]
    judge = config.get("judge_model", "")
    critic = config.get("critic_model", "")
    parts = []
    if model_names:
        parts.append("Models: " + ", ".join(model_names))
    if judge:
        parts.append(f"Judge: {judge}")
    if critic and critic not in ("same", "judge"):
        parts.append(f"Critic: {critic}")
    elif critic == "judge" and judge:
        parts.append(f"Critic: {judge} (=judge)")
    return "  |  ".join(parts)


def _add_model_info(fig, model_info: str):
    """Add model info as a small annotation at the top of a figure (above the title)."""
    if model_info:
        fig.suptitle(
            model_info,
            fontsize=7, fontstyle="italic", color="0.4",
            y=0.995,
        )


def _bootstrap_ci(values, n_boot=2000, ci=0.95):
    """Bootstrap 95 % CI for the mean. Returns (lower, upper)."""
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) else 0.0
        return m, m
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    means = np.array(
        [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def _wilcoxon_p(tech_scores: list[float], baseline_scores: list[float]) -> float | None:
    """Paired Wilcoxon signed-rank test (one-sided: tech > baseline). Returns p or None."""
    if len(tech_scores) < 5 or len(tech_scores) != len(baseline_scores):
        return None
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(tech_scores, baseline_scores, alternative="greater")
        return float(p)
    except Exception:
        return None


def _significance_star(p: float | None) -> str:
    """Convert p-value to significance star annotation."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Loader / dispatcher
# ---------------------------------------------------------------------------

def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def _ensure_normalized(results: list[dict]):
    """
    Compute cost normalization in-place.

    ALWAYS recomputes from raw data. Normalization is cheap (O(N) dict lookup).

    Handles three edge cases that previously caused cost_overhead=0:
    1. Cache files saved before normalization (no cost_overhead field)
    2. total_cost_usd=0 despite real LLM calls (Ollama usage reporting gaps)
    3. Baseline has zero cost (fallback to num_llm_calls ratio)
    """
    # Canonicalize technique names (e.g. "fec_0.5" → "fec_0.50")
    _TECHNIQUE_ALIASES = {"fec_0.5": "fec_0.50"}
    for r in results:
        t = r.get("technique", "")
        if t in _TECHNIQUE_ALIASES:
            r["technique"] = _TECHNIQUE_ALIASES[t]

    # Build per-task baseline lookup
    baselines: dict[str, dict[str, float]] = {}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_task[r["task_id"]].append(r)

    if not by_task:
        return

    for task_id, runs in by_task.items():
        bl_runs = [r for r in runs if r["technique"] == "baseline"]
        if bl_runs:
            bl = bl_runs[0]
        else:
            bl = min(runs, key=lambda r: r.get("total_cost_usd", float("inf")))
        baselines[task_id] = {
            "cost": bl.get("total_cost_usd", 0),
            "quality": bl.get("final_quality", 0),
            "calls": bl.get("num_llm_calls", 1),
        }

    normalized_count = 0
    zero_cost_estimated = 0
    for r in results:
        if "error" in r:
            continue
        bl = baselines.get(r["task_id"], {"cost": 0, "quality": 0, "calls": 1})
        bl_cost = bl["cost"]
        r["baseline_quality"] = bl["quality"]
        r["baseline_cost_usd"] = bl_cost

        r_cost = r.get("total_cost_usd", 0)
        r_calls = r.get("num_llm_calls", 0)

        # Cost overhead: how many multiples of baseline cost this technique uses.
        # Three tiers of estimation:
        #   1. Both have real costs → direct ratio (most accurate)
        #   2. One/both have zero cost but real call counts → ratio of calls
        #      (Ollama doesn't always report usage; call count is a fair proxy)
        #   3. Neither has data → default to 1.0
        if bl_cost > 0 and r_cost > 0:
            r["cost_overhead"] = r_cost / bl_cost
        elif r_calls > 0 and bl.get("calls", 0) > 0:
            # Fallback: estimate overhead from LLM call ratio.
            # This handles the case where Ollama returns usage=None or
            # cost data is missing from cached results.
            r["cost_overhead"] = r_calls / bl["calls"]
            zero_cost_estimated += 1
        else:
            r["cost_overhead"] = 1.0

        quality_delta = r["final_quality"] - bl["quality"]
        cost_delta = r_cost - bl_cost
        r["quality_gain_per_cost"] = quality_delta / cost_delta if cost_delta > 0 else 0.0
        normalized_count += 1

    if zero_cost_estimated > 0:
        logger.warning(
            f"Estimated cost_overhead from call counts for {zero_cost_estimated} "
            f"results (total_cost_usd was 0 despite having LLM calls)"
        )
    logger.info(
        f"Computed cost normalization for {normalized_count} results "
        f"across {len(baselines)} tasks"
    )


def plot_all_from_results(results: list[dict], output_dir: str | Path = "plots",
                          model_info: str = ""):
    """Generate all paper plots from in-memory results (for live updates)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate ACM sub-technique variants under their parent label.
    # acm_uncoded, acm_harq_ir, acm_fec_r0.5, ... → "acm"
    # acm_soft_fec, acm_soft_harq_ir, ...          → "acm_soft"
    # Shallow-copy rewritten dicts so callers' input (e.g. self.results on the
    # runner) is not mutated — previous in-place rewrite caused acm_soft_*
    # entries to be saved under technique="acm", polluting the acm cache.
    rewritten: list[dict] = []
    for r in results:
        tech = r.get("technique", "")
        # Check soft variants FIRST: "acm_soft" itself also matches
        # "acm_".startswith, so the generic acm_* rule would otherwise swallow
        # soft runs into the hard-ACM pool and bias the gap analysis.
        if tech == "acm_soft" or tech.startswith("acm_soft_"):
            r = {**r, "technique": "acm_soft"}
        elif tech == "acm_learned" or tech.startswith("acm_learned_"):
            # Keep acm_learned distinct from hand-coded ACM — it is a
            # separately learned policy, not an ACM sub-technique variant.
            r = {**r, "technique": "acm_learned"}
        elif tech == "acm" or tech.startswith("acm_"):
            r = {**r, "technique": "acm"}
        rewritten.append(r)
    results = rewritten

    _ensure_normalized(results)

    plot_quality_vs_cost(results, output_dir, model_info=model_info)
    plot_matched_budget(results, output_dir, model_info=model_info)
    plot_diversity_gain(results, output_dir, model_info=model_info)
    plot_harq_convergence(results, output_dir, model_info=model_info)
    plot_turbo_waterfall(results, output_dir, model_info=model_info)
    plot_fec_rate_distortion(results, output_dir, model_info=model_info)
    plot_technique_comparison_heatmap(results, output_dir, model_info=model_info)
    plot_quality_distribution(results, output_dir, model_info=model_info)
    plot_acm_oracle_gap(results, output_dir, model_info=model_info)
    # Hard-benchmark panels — both no-op silently when no *_hard_* task ids
    # are present, so it's safe to call on every run.
    plot_hard_benchmark_bars(results, output_dir, model_info=model_info)
    plot_hard_benchmark_pareto(results, output_dir, model_info=model_info)
    # Per-category panels for the curated 69-task set; no-op on standard-
    # datasets-only runs (every task carries a *_hard_* id and is filtered out).
    plot_curated_category_bars(results, output_dir, model_info=model_info)
    plot_curated_category_pareto(results, output_dir, model_info=model_info)

    logger.info(f"All plots saved to {output_dir}/")


def plot_all(results_path: str | Path, output_dir: str | Path = "plots"):
    """Generate all paper plots from a results file."""
    data = load_results(results_path)
    results = data["results"]
    model_info = _model_info_text(data.get("config"))
    plot_all_from_results(results, output_dir, model_info=model_info)


def plot_all_from_cache(cache_dir: str | Path, output_dir: str | Path = "plots"):
    """
    Generate all paper plots by loading per-technique JSON files from a cache
    directory (e.g. results/cache/).

    This is the most common use case: each technique is cached independently
    as <technique>.json. We merge all results, compute cost normalization
    (which requires cross-technique baseline data), and generate plots.
    """
    import json as _json

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    config_info: dict | None = None

    for path in sorted(cache_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = _json.load(f)
            results = data.get("results", [])
            all_results.extend(results)
            if config_info is None and "config" in data:
                config_info = data["config"]
            logger.info(f"Loaded {len(results)} results from {path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    if not all_results:
        logger.error(f"No results found in {cache_dir}")
        return

    logger.info(f"Total: {len(all_results)} results from {cache_dir}")
    model_info = _model_info_text(config_info) if config_info else ""
    plot_all_from_results(all_results, output_dir, model_info=model_info)


# ===================================================================
# Plot 1: Quality vs. Cost  (BER vs Eb/N0 — the money plot)
# ===================================================================

def plot_quality_vs_cost(results: list[dict], output_dir: Path, model_info: str = ""):
    """
    Three panels:
      1. Raw cost scatter
      2. Quality vs normalised cost overhead (scatter)
      3. Quality gain vs cost overhead (ROI scatter)
    All means carry 95 % bootstrap error bars with improved visibility.
    The standalone line plot (mean quality vs normalised cost with CI bands)
    is still saved separately as quality_vs_overhead_line.{png,pdf}.
    """
    # Check with `is not None` — cost_overhead=0.0 is falsy but valid.
    has_normalised = any(
        r.get("cost_overhead") is not None
        for r in results if "error" not in r
    )
    if has_normalised:
        fig, axes_row = plt.subplots(1, 3, figsize=(16.5, 4.8))
    else:
        fig, single_ax = plt.subplots(1, 1, figsize=(6, 4.8))
        axes_row = np.array([single_ax])

    by_tech = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_tech[r["technique"]].append(r)

    def _scatter_panel(ax, x_key, y_key, y_transform=None, skip_baseline=False):
        """Scatter individual + mean with CI error bar on one axis.

        Returns the list of mean x-values so the caller can set a sensible
        x-axis limit (otherwise one outlier individual point stretches the
        axis and crushes all mean markers near x=0).
        """
        mean_xs: list[float] = []
        for tech in TECHNIQUE_ORDER:
            if tech not in by_tech:
                continue
            if skip_baseline and tech == "baseline":
                continue
            runs = by_tech[tech]
            xs = [r.get(x_key, r.get("total_cost_usd", 0)) for r in runs]
            ys = [y_transform(r) if y_transform else r[y_key] for r in runs]
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            ci_lo, ci_hi = _bootstrap_ci(ys)
            mean_xs.append(mx)

            c = TECHNIQUE_COLORS.get(tech, "#333")
            m = TECHNIQUE_MARKERS.get(tech, "o")
            lab = TECHNIQUE_LABELS.get(tech, tech)

            # Individual points (faded)
            ax.scatter(xs, ys, c=c, marker=m, alpha=0.15, s=12, linewidths=0)
            # Error bar
            ax.errorbar(mx, my, yerr=[[my - ci_lo], [ci_hi - my]],
                        fmt="none", ecolor=c, capsize=4, capthick=1.2,
                        elinewidth=1.6, zorder=4)
            # Mean marker
            ax.scatter([mx], [my], c=c, marker=m, s=60,
                       edgecolors="black", linewidths=0.5, label=lab, zorder=5)
        return mean_xs

    def _clip_xlim(ax, mean_xs, pad_factor=1.35, lo=0.0):
        """Set x-axis limit so mean markers occupy most of the panel width.

        Without this, one outlier individual-scatter point stretches the
        x-axis and crushes all the means near x=0, making the plot look
        like techniques are at rho≈0 when they are actually at rho≈3–5.
        """
        if not mean_xs:
            return
        hi = max(mean_xs) * pad_factor
        ax.set_xlim(lo, max(hi, lo + 1e-6))

    # — Panel 1: raw cost —
    ax = axes_row[0]
    mxs = _scatter_panel(ax, "total_cost_usd", "final_quality")
    ax.set_xlabel("Esimated Cost (USD)")
    ax.set_ylabel("Quality")
    ax.set_title("Quality vs. raw cost")
    ax.set_ylim(0, 1.05)
    _clip_xlim(ax, mxs)

    if has_normalised:
        # — Panel 2: quality vs normalised overhead —
        ax = axes_row[1]
        mxs = _scatter_panel(ax, "cost_overhead", "final_quality")
        ax.set_xlabel(r"Cost overhead $\rho$ ($\times$baseline)")
        ax.set_ylabel("Quality")
        ax.set_title("Quality vs. normalised cost")
        ax.set_ylim(0, 1.05)
        ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
        _clip_xlim(ax, mxs)

        # — Panel 3: quality gain vs overhead (ROI) —
        ax = axes_row[2]
        mxs = _scatter_panel(
            ax, "cost_overhead", None,
            y_transform=lambda r: r["final_quality"] - r.get("baseline_quality", 0),
            skip_baseline=True,
        )
        ax.set_xlabel(r"Cost overhead $\rho$ ($\times$baseline)")
        ax.set_ylabel(r"$\Delta$ Quality (vs. baseline)")
        ax.set_title("Return on investment")
        ax.axhline(0, color="0.5", ls="--", lw=0.8, zorder=0)
        ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
        _clip_xlim(ax, mxs)
        # Also clip y for ROI so we aren't squashed by rare outlier deltas
        ax.set_ylim(-0.15, 0.35)

    # Shared legend below all panels — avoids overlap
    handles, labels = axes_row[0].get_legend_handles_labels()
    n_legend_rows = 0
    if handles:
        ncol = min(len(handles), 4)
        n_legend_rows = math.ceil(len(handles) / ncol)
        fig.legend(handles, labels, loc="lower center",
                   ncol=ncol, fontsize=7,
                   framealpha=0.9, edgecolor="0.7",
                   bbox_to_anchor=(0.5, -0.02))

    _add_model_info(fig, model_info)
    # Reserve enough vertical space at the bottom for the legend so it
    # does not overlap the panels. Each legend row is ~0.04 of fig height.
    bottom_reserve = 0.02 + 0.035 * max(0, n_legend_rows - 1)
    fig.tight_layout(w_pad=2.5, h_pad=2.0, rect=[0, bottom_reserve, 1, 0.97])
    fig.savefig(output_dir / "quality_vs_cost.png")
    fig.savefig(output_dir / "quality_vs_cost.pdf")
    plt.close(fig)
    logger.info("Saved quality_vs_cost plot")

    # Also save standalone line plot for the paper
    if has_normalised:
        _save_standalone_line_plot(by_tech, output_dir)


def _line_panel_quality_vs_overhead(ax, by_tech: dict[str, list[dict]]):
    """Line plot: mean quality vs mean normalised cost overhead, with CI bands."""
    # Collect (mean_overhead, mean_quality, ci_lo, ci_hi) per technique
    points = []
    for tech in TECHNIQUE_ORDER:
        if tech not in by_tech:
            continue
        runs = by_tech[tech]
        overheads = [r.get("cost_overhead", 1.0) for r in runs]
        quals = [r["final_quality"] for r in runs]
        if not overheads:
            continue
        mo = float(np.mean(overheads))
        mq = float(np.mean(quals))
        ci_lo, ci_hi = _bootstrap_ci(quals)
        points.append((tech, mo, mq, ci_lo, ci_hi))

    # Sort by cost overhead for line connectivity
    points.sort(key=lambda p: p[1])

    # Draw CI band + line + markers
    for tech, mo, mq, ci_lo, ci_hi in points:
        c = TECHNIQUE_COLORS.get(tech, "#333")
        m = TECHNIQUE_MARKERS.get(tech, "o")
        lab = TECHNIQUE_LABELS.get(tech, tech)
        # Error bar
        ax.errorbar(mo, mq, yerr=[[mq - ci_lo], [ci_hi - mq]],
                    fmt="none", ecolor=c, capsize=4, capthick=1.2,
                    elinewidth=1.6, zorder=4)
        # Marker
        ax.scatter([mo], [mq], c=c, marker=m, s=55,
                   edgecolors="black", linewidths=0.5, label=lab, zorder=5)

    # Connect points with a thin line to show the frontier
    if len(points) > 1:
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, "k-", alpha=0.15, lw=1.0, zorder=1)

    ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel(r"Cost overhead $\rho$ ($\times$baseline)")
    ax.set_ylabel("Mean quality (95% CI)")
    ax.set_title("Quality vs. normalised cost (line + CI)")
    ax.set_ylim(0, 1.05)


def _save_standalone_line_plot(by_tech: dict[str, list[dict]], output_dir: Path):
    """Save a standalone line plot figure for the paper."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_panel_quality_vs_overhead(ax, by_tech)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=7,
                  ncol=1, framealpha=0.9, edgecolor="0.7")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_vs_normalized_cost_line.png")
    fig.savefig(output_dir / "quality_vs_normalized_cost_line.pdf")
    plt.close(fig)
    logger.info("Saved quality_vs_normalized_cost_line plot")


# ===================================================================
# Plot 1b: Matched-budget evaluation
# ===================================================================

def plot_matched_budget(results: list[dict], output_dir: Path, model_info: str = ""):
    """
    Matched-budget evaluation (cf. Snell et al., 2024).

    Fix a sequence of budget ceilings B (rho ~ {2x, 3x, 4x, 5x, 7x, 10x}).
    For each budget, each technique is represented by its mean quality over
    runs whose cost_overhead falls in the bucket around B. The "winner" at
    each budget is the technique with the highest mean quality, marked with
    a black outline.

    This answers: "at a fixed inference budget, which technique wins?" —
    the question a practitioner actually asks, as opposed to the raw
    quality vs cost cloud.
    """
    runs_ok = [
        r for r in results
        if "error" not in r and r.get("cost_overhead") is not None
    ]
    if not runs_ok:
        logger.info("Skipped matched_budget plot (no cost_overhead data)")
        return

    buckets = [
        (1.5, 2.5, "2x"),
        (2.5, 3.5, "3x"),
        (3.5, 4.5, "4x"),
        (4.5, 6.0, "5x"),
        (6.0, 8.5, "7x"),
        (8.5, 14.0, "10x"),
    ]

    by_tech = defaultdict(list)
    for r in runs_ok:
        by_tech[r["technique"]].append(r)

    techs = [t for t in TECHNIQUE_ORDER if t in by_tech and t != "baseline"]
    if not techs:
        logger.info("Skipped matched_budget plot (no non-baseline techniques)")
        return

    # Build data: per (budget, tech) → (mean_quality, ci_lo, ci_hi, n)
    grid: dict[tuple[int, str], tuple[float, float, float, int]] = {}
    for bi, (lo, hi, _) in enumerate(buckets):
        for tech in techs:
            vals = [
                r["final_quality"] for r in by_tech[tech]
                if lo <= r["cost_overhead"] < hi
            ]
            if len(vals) >= 2:
                m = float(np.mean(vals))
                cl, ch = _bootstrap_ci(vals)
                grid[(bi, tech)] = (m, cl, ch, len(vals))

    # Drop buckets that have fewer than 2 techniques represented.
    live_buckets = [
        bi for bi in range(len(buckets))
        if sum(1 for tech in techs if (bi, tech) in grid) >= 2
    ]
    if not live_buckets:
        logger.info("Skipped matched_budget plot (no budget bucket had >=2 techniques)")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_techs = len(techs)
    group_w = 0.85
    bar_w = group_w / max(n_techs, 1)

    for ti, tech in enumerate(techs):
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for bi in live_buckets:
            key = (bi, tech)
            if key not in grid:
                continue
            m, cl, ch, _ = grid[key]
            xs.append(bi + (ti - (n_techs - 1) / 2.0) * bar_w)
            ys.append(m)
            yerr_lo.append(m - cl)
            yerr_hi.append(ch - m)
        if not xs:
            continue
        c = TECHNIQUE_COLORS.get(tech, "#333")
        lab = TECHNIQUE_LABELS.get(tech, tech)
        ax.bar(xs, ys, width=bar_w * 0.9, color=c, edgecolor="none",
               alpha=0.85, label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="none",
                    ecolor="black", capsize=2, capthick=0.6, elinewidth=0.6,
                    alpha=0.5, zorder=3)

    # Outline the winner per bucket.
    for bi in live_buckets:
        best_tech = max(
            (t for t in techs if (bi, t) in grid),
            key=lambda t: grid[(bi, t)][0],
        )
        ti = techs.index(best_tech)
        m, _, _, _ = grid[(bi, best_tech)]
        x = bi + (ti - (n_techs - 1) / 2.0) * bar_w
        ax.bar([x], [m], width=bar_w * 0.9, color="none",
               edgecolor="black", linewidth=1.6, zorder=4)

    ax.set_xticks(live_buckets)
    ax.set_xticklabels([buckets[bi][2] for bi in live_buckets])
    ax.set_xlabel(r"Matched budget $\rho$ ($\times$baseline cost)")
    ax.set_ylabel("Mean quality (95% CI)")
    ax.set_title("Matched-budget evaluation — winner per budget (black outline)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=7, ncol=2,
                  framealpha=0.9, edgecolor="0.7")

    _add_model_info(fig, model_info)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "matched_budget.png")
    fig.savefig(output_dir / "matched_budget.pdf")
    plt.close(fig)

    # Log the winner table for the paper.
    winner_lines = []
    for bi in live_buckets:
        best_tech = max(
            (t for t in techs if (bi, t) in grid),
            key=lambda t: grid[(bi, t)][0],
        )
        m, _, _, n = grid[(bi, best_tech)]
        winner_lines.append(
            f"  budget={buckets[bi][2]:>4s}  winner={best_tech:<22s}  q={m:.3f}  n={n}"
        )
    logger.info("Saved matched_budget plot. Winner table:\n" + "\n".join(winner_lines))


# ===================================================================
# Plot 2: Diversity gain vs. branch count
# ===================================================================

def plot_diversity_gain(results: list[dict], output_dir: Path, model_info: str = ""):
    """Diversity gain per combining strategy. Mean + 95 % CI."""
    diversity_runs = [r for r in results if r.get("technique", "").startswith("diversity_")]
    if not diversity_runs:
        logger.info("Skipped diversity_gain plot (no diversity runs)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    # Group by technique → branch count
    by_tech: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in diversity_runs:
        tech = r["technique"]
        d = r.get("config", {}).get("num_channels", 1)
        by_tech[tech][d].append(r.get("diversity_gain", 0))

    for tech in ["diversity_sc", "diversity_mrc", "diversity_egc"]:
        if tech not in by_tech:
            continue
        d_data = by_tech[tech]
        branches = sorted(d_data.keys())
        means, ci_los, ci_his = [], [], []
        for d in branches:
            vals = d_data[d]
            m = float(np.mean(vals))
            lo, hi = _bootstrap_ci(vals)
            means.append(m)
            ci_los.append(m - lo)
            ci_his.append(hi - m)

        c = TECHNIQUE_COLORS.get(tech, "#333")
        lab = TECHNIQUE_LABELS.get(tech, tech)
        ax.errorbar(branches, means, yerr=[ci_los, ci_his],
                     fmt="o-", color=c, label=lab, capsize=4, capthick=1,
                     markeredgecolor="black", markeredgewidth=0.3, markersize=7)

    ax.axhline(0, color="0.5", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Number of diversity branches")
    ax.set_ylabel("Diversity gain (combined $-$ best individual)")
    ax.set_title("Diversity gain vs. branch count")
    ax.legend()

    _add_model_info(fig, model_info)
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_gain.png")
    fig.savefig(output_dir / "diversity_gain.pdf")
    plt.close(fig)
    logger.info("Saved diversity_gain plot")


# ===================================================================
# Plot 3: HARQ convergence curves
# ===================================================================

def plot_harq_convergence(results: list[dict], output_dir: Path, model_info: str = ""):
    """Quality vs. round for HARQ-CC / IR with CI shading."""
    harq_runs = [r for r in results if r.get("technique", "").startswith("harq_")]
    if not harq_runs:
        logger.info("Skipped harq_convergence plot (no HARQ runs)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for tech_name in ["harq_cc", "harq_ir"]:
        runs = [r for r in harq_runs if r["technique"] == tech_name]
        if not runs:
            continue

        max_rounds = max(len(r.get("individual_scores", [])) for r in runs)
        means, ci_los, ci_his = [], [], []
        for idx in range(max_rounds):
            # Pad early-exit tasks with their last score so the mean
            # isn't inflated at round 1 by high-quality early exits
            scores = []
            for r in runs:
                s = r.get("individual_scores", [])
                if idx < len(s):
                    scores.append(s[idx])
                elif s:
                    scores.append(s[-1])  # carry forward last score
            if not scores:
                break
            m = float(np.mean(scores))
            lo, hi = _bootstrap_ci(scores)
            means.append(m)
            ci_los.append(lo)
            ci_his.append(hi)

        if means:
            c = TECHNIQUE_COLORS.get(tech_name, "#333")
            lab = TECHNIQUE_LABELS.get(tech_name, tech_name)
            rounds = np.arange(1, len(means) + 1)
            ax.plot(rounds, means, "o-", color=c, label=lab, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.3)
            ax.fill_between(rounds, ci_los, ci_his, color=c, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel("Quality (mean across tasks)")
    ax.set_title("HARQ convergence")
    ax.set_ylim(0.5, 1.05)
    ax.legend()

    _add_model_info(fig, model_info)
    fig.tight_layout()
    fig.savefig(output_dir / "harq_convergence.png")
    fig.savefig(output_dir / "harq_convergence.pdf")
    plt.close(fig)
    logger.info("Saved harq_convergence plot")


# ===================================================================
# Plot 4: Turbo waterfall
# ===================================================================

def plot_turbo_waterfall(results: list[dict], output_dir: Path, model_info: str = ""):
    """Individual task curves (thin) + mean with CI shading.

    X-axis starts at 0 (= baseline, initial generation before any refinement).
    individual_scores[0] is the generator's first output; [1], [2], ... are
    successive critic-driven refinements. Overlaying the x=0 point makes the
    improvement (or regression) of refinement visible at a glance.
    """
    turbo_runs = [r for r in results if r.get("technique") == "turbo"]
    if not turbo_runs:
        logger.info("Skipped turbo_waterfall plot (no turbo runs)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    c = TECHNIQUE_COLORS["turbo"]

    # Individual curves (x starts at 0 = baseline)
    for r in turbo_runs:
        scores = r.get("individual_scores", [])
        if scores:
            ax.plot(range(0, len(scores)), scores, "-", alpha=0.15, color=c, lw=0.8)

    # Average + CI — "survivor" curve: at iter k, averages over tasks that
    # reached iter k. This curve LOOKS like it drops because high-scoring
    # tasks early-exit and disappear from the mean — not because refinement
    # made them worse. We plot it faintly and also show the same-cohort
    # curve (tasks that ran all max_iter iterations) which is the correct
    # signal for "does turbo actually improve a given task?".
    max_iters = max(len(r.get("individual_scores", [])) for r in turbo_runs)
    means, ci_los, ci_his, ns = [], [], [], []
    for idx in range(max_iters):
        scores = [r["individual_scores"][idx]
                  for r in turbo_runs if idx < len(r.get("individual_scores", []))]
        if not scores:
            break
        m = float(np.mean(scores))
        lo, hi = _bootstrap_ci(scores)
        means.append(m)
        ci_los.append(lo)
        ci_his.append(hi)
        ns.append(len(scores))

    if means:
        iters = np.arange(0, len(means))
        ax.plot(iters, means, "o-", color=c, lw=1.5, markersize=6, alpha=0.55,
                markeredgecolor="black", markeredgewidth=0.3,
                label="Survivor mean", zorder=4)
        ax.fill_between(iters, ci_los, ci_his, color=c, alpha=0.15, label="95 % CI")
        # Annotate the per-iteration n so survivorship bias can't hide in the
        # averaging (curves shorten as tasks early-exit).
        for x, y, n in zip(iters, means, ns, strict=False):
            ax.annotate(f"n={n}", (x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="#555")

    # All-tasks mean with running max carried forward for early-exited tasks.
    # This is what the technique *actually delivers* to the user at each
    # iteration: because the best-of-sequence guard returns argmax over all
    # seen scores, a task that early-exits at iter j contributes max(scores[:j+1])
    # at every later iter. Unlike the survivor curve, this keeps n fixed at
    # the full population and therefore makes monotonic improvement visible.
    all_means, all_los, all_his = [], [], []
    for idx in range(max_iters):
        vals = []
        for r in turbo_runs:
            scores = r.get("individual_scores", [])
            if not scores:
                continue
            eff = scores[: idx + 1] if idx < len(scores) else scores
            vals.append(max(eff))
        if not vals:
            break
        all_means.append(float(np.mean(vals)))
        lo, hi = _bootstrap_ci(vals)
        all_los.append(lo)
        all_his.append(hi)
    if all_means:
        iters_all = np.arange(0, len(all_means))
        ax.plot(iters_all, all_means, "^-", color="#228833", lw=2.4,
                markersize=7, markeredgecolor="black", markeredgewidth=0.3,
                label=f"All-tasks running-max mean (n={len(turbo_runs)} fixed)",
                zorder=6)

    # Same-cohort curve: fixed subset of tasks that ran all max_iters. This
    # is the honest "does refinement help per task?" signal.
    full_cohort = [r.get("individual_scores", []) for r in turbo_runs
                   if len(r.get("individual_scores", [])) == max_iters]
    if len(full_cohort) >= 2 and max_iters >= 2:
        sc_means, sc_los, sc_his = [], [], []
        for idx in range(max_iters):
            vals = [t[idx] for t in full_cohort]
            sc_means.append(float(np.mean(vals)))
            lo, hi = _bootstrap_ci(vals)
            sc_los.append(lo)
            sc_his.append(hi)
        iters_sc = np.arange(0, max_iters)
        ax.plot(iters_sc, sc_means, "s--", color="#AA3377", lw=2.2,
                markersize=7, markeredgecolor="black", markeredgewidth=0.3,
                label=f"Same-cohort mean (n={len(full_cohort)}, ran all {max_iters} iters)",
                zorder=5)

    if means:
        # Horizontal reference at the baseline (iter 0, full population)
        ax.axhline(means[0], color="gray", ls=":", lw=1.0, alpha=0.7,
                   label=f"Baseline (iter 0, all n={ns[0]}) = {means[0]:.3f}")

    ax.set_xlabel("Iteration (0 = baseline, no refinement)")
    ax.set_ylabel("Quality")
    ax.set_title("Turbo decoder: quality vs. refinement iteration")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, max(max_iters, 1)))
    ax.legend(loc="lower right", fontsize=8)

    _add_model_info(fig, model_info)
    fig.tight_layout()
    fig.savefig(output_dir / "turbo_waterfall.png")
    fig.savefig(output_dir / "turbo_waterfall.pdf")
    plt.close(fig)
    logger.info("Saved turbo_waterfall plot")


# ===================================================================
# Plot 5: FEC rate-distortion
# ===================================================================

def plot_fec_rate_distortion(results: list[dict], output_dir: Path, model_info: str = ""):
    """Left: quality vs code rate with CI.  Right: quality-cost scatter by rate."""
    fec_runs = [r for r in results if r.get("technique", "").startswith("fec_")]
    if not fec_runs:
        logger.info("Skipped fec_rate_distortion plot (no FEC runs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    by_rate: dict[float, list[dict]] = defaultdict(list)
    for r in fec_runs:
        rate = r.get("config", {}).get("effective_rate",
               r.get("config", {}).get("code_rate", 1.0))
        by_rate[rate].append(r)

    baselines = [r for r in results if r.get("technique") == "baseline"]
    if baselines:
        by_rate[1.0] = baselines

    rates = sorted(by_rate.keys(), reverse=True)
    means, ci_los, ci_his = [], [], []
    for rate in rates:
        quals = [r["final_quality"] for r in by_rate[rate]]
        m = float(np.mean(quals))
        lo, hi = _bootstrap_ci(quals)
        means.append(m)
        ci_los.append(m - lo)
        ci_his.append(hi - m)

    # Left: quality vs rate
    ax = axes[0]
    ax.errorbar(rates, means, yerr=[ci_los, ci_his],
                fmt="o-", color="#228833", capsize=4, capthick=1,
                markeredgecolor="black", markeredgewidth=0.3, markersize=7)
    # Annotate rates
    for rate, m in zip(rates, means, strict=False):
        label = f"r={rate}" if rate < 1.0 else "uncoded"
        ax.annotate(label, (rate, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7, color="0.3")
    ax.set_xlabel("Code rate (r)")
    ax.set_ylabel("Mean quality")
    ax.set_title("FEC: quality vs. code rate")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()

    # Right: mean cost vs mean quality per rate, with CI and connecting line
    ax = axes[1]
    rate_colors = {1.0: "#BBBBBB", 0.75: "#EE9933", 0.5: "#CC6633",
                   0.33: "#882255", 0.25: "#332288"}
    rate_markers = {1.0: "X", 0.75: "h", 0.5: "H",
                    0.33: "p", 0.25: "8"}
    pts = []  # (mean_cost, mean_qual, ci_lo, ci_hi, rate, color)
    for rate in rates:
        runs = by_rate[rate]
        costs = [r.get("total_cost_usd", 0) for r in runs]
        quals = [r["final_quality"] for r in runs]
        mc = float(np.mean(costs))
        mq = float(np.mean(quals))
        ci_lo, ci_hi = _bootstrap_ci(quals)
        c = rate_colors.get(rate, "#333")
        mk = rate_markers.get(rate, "o")
        label = f"r={rate}" if rate < 1.0 else "Uncoded (r=1)"
        pts.append((mc, mq, ci_lo, ci_hi, rate, c, mk, label))

    # Sort by cost for connecting line
    pts.sort(key=lambda p: p[0])

    # Connecting line (thin)
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "-", color="0.6", lw=1.0, zorder=1)

    # Points with error bars
    for mc, mq, ci_lo, ci_hi, rate, c, mk, label in pts:
        ax.errorbar(mc, mq, yerr=[[mq - ci_lo], [ci_hi - mq]],
                    fmt="none", ecolor=c, capsize=4, capthick=1.2,
                    elinewidth=1.6, zorder=4)
        ax.scatter([mc], [mq], c=c, marker=mk, s=70,
                   edgecolors="black", linewidths=0.5, label=label, zorder=5)
        # Annotate rate next to each point
        ax.annotate(f"r={rate}" if rate < 1.0 else "uncoded",
                    (mc, mq), textcoords="offset points",
                    xytext=(8, 4), fontsize=7, color="0.3")

    ax.set_xlabel("Mean cost (USD)")
    ax.set_ylabel("Mean quality (95% CI)")
    ax.set_title("FEC: quality-cost tradeoff by rate")
    ax.legend(fontsize=7, loc="lower right")

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2)
    fig.savefig(output_dir / "fec_rate_distortion.png")
    fig.savefig(output_dir / "fec_rate_distortion.pdf")
    plt.close(fig)
    logger.info("Saved fec_rate_distortion plot")


# ===================================================================
# Plot 6: Technique x Category heatmap  (with significance)
# ===================================================================

def plot_technique_comparison_heatmap(results: list[dict], output_dir: Path, model_info: str = ""):
    """Heatmap with per-cell mean quality and significance stars vs baseline."""
    by_tech_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" not in r:
            by_tech_cat[r["technique"]][r["task_category"]].append(r["final_quality"])

    # Ordered technique / category lists
    techniques = [t for t in TECHNIQUE_ORDER if t in by_tech_cat]
    categories = sorted({r["task_category"] for r in results if "error" not in r})

    # Baseline lookup per category-task for significance
    baseline_by_cat_task: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        if "error" not in r and r["technique"] == "baseline":
            baseline_by_cat_task[r["task_category"]][r["task_id"]] = r["final_quality"]

    nrow, ncol = len(techniques), len(categories)
    matrix = np.zeros((nrow, ncol))
    annotations = [[""]*ncol for _ in range(nrow)]

    for i, tech in enumerate(techniques):
        for j, cat in enumerate(categories):
            scores = by_tech_cat[tech][cat]
            matrix[i, j] = float(np.mean(scores)) if scores else 0

            # Significance test vs baseline for the same category's tasks
            if tech != "baseline" and cat in baseline_by_cat_task:
                tech_by_task = {}
                for r in results:
                    if "error" not in r and r["technique"] == tech and r["task_category"] == cat:
                        tech_by_task[r["task_id"]] = r["final_quality"]
                paired_t, paired_b = [], []
                for tid in tech_by_task:
                    if tid in baseline_by_cat_task[cat]:
                        paired_t.append(tech_by_task[tid])
                        paired_b.append(baseline_by_cat_task[cat][tid])
                p = _wilcoxon_p(paired_t, paired_b)
                star = _significance_star(p)
            else:
                star = ""

            annotations[i][j] = f"{matrix[i, j]:.2f}{star}"

    fig, ax = plt.subplots(figsize=(max(6, ncol * 1.8), max(4, nrow * 0.55)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([c.upper() for c in categories], fontsize=9)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([TECHNIQUE_LABELS.get(t, t) for t in techniques], fontsize=9)

    for i in range(nrow):
        for j in range(ncol):
            txt_color = "white" if matrix[i, j] < 0.35 or matrix[i, j] > 0.85 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean quality", fontsize=9)
    ax.set_title("Technique $\\times$ category\n(* p<.05  ** p<.01  *** p<.001  vs. baseline)")

    _add_model_info(fig, model_info)
    fig.tight_layout()
    fig.savefig(output_dir / "technique_heatmap.png")
    fig.savefig(output_dir / "technique_heatmap.pdf")
    plt.close(fig)
    logger.info("Saved technique_heatmap plot")


# ===================================================================
# Plot 7: Quality distribution (violin + strip)
# ===================================================================

def plot_quality_distribution(results: list[dict], output_dir: Path, model_info: str = ""):
    """Violin plots with individual data points and significance annotations.

    Split into two stacked panels so labels stay legible:
      - Top: AgentCodec techniques (baseline + comm-theoretic operators and
        soft variants).
      - Bottom: prior-method baselines and wider-pool multi-model diversity
        operators (diversity_*_N, self_consistency, mixture_of_agents, ...).
    """
    by_tech = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_tech[r["technique"]].append(r["final_quality"])

    # Split into two roughly equal panels (12 + 11) so violin density matches.
    # Top: core AgentCodec communication-theoretic operators.
    # Bottom: soft variants + wider-pool diversity + prior-method baselines.
    panel_top_set = {
        "baseline",
        "diversity_sc", "diversity_mrc", "diversity_egc",
        "harq_cc", "harq_ir", "turbo", "fountain",
        "fec_0.75", "fec_0.50", "fec_0.33", "acm",
    }
    panel_top = [t for t in TECHNIQUE_ORDER if t in by_tech and t in panel_top_set]
    panel_bot = [t for t in TECHNIQUE_ORDER if t in by_tech and t not in panel_top_set]

    # Per-task baseline lookup (shared across panels for significance stars).
    baseline_by_task = {}
    for r in results:
        if "error" not in r and r["technique"] == "baseline":
            baseline_by_task[r["task_id"]] = r["final_quality"]

    n_top = max(len(panel_top), 1)
    n_bot = max(len(panel_bot), 1)
    fig_w = max(8.0, max(n_top, n_bot) * 0.85)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 8.5),
                             gridspec_kw={"height_ratios": [1, 1]})

    def _draw_panel(ax, techniques: list[str], title: str):
        if not techniques:
            ax.set_visible(False)
            return
        data = [by_tech[t] for t in techniques]
        labels = [TECHNIQUE_LABELS.get(t, t) for t in techniques]
        colors = [TECHNIQUE_COLORS.get(t, "#CCC") for t in techniques]

        parts = ax.violinplot(data, positions=range(len(techniques)),
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("0.3")
            pc.set_alpha(0.35)
            pc.set_linewidth(0.6)

        bp = ax.boxplot(data, positions=range(len(techniques)), widths=0.15,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5),
                        boxprops=dict(lw=0.8),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8))
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i])
            box.set_alpha(0.6)

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=5, c=colors[i], edgecolors="none", alpha=0.25, zorder=0)

        # Significance stars vs baseline
        if baseline_by_task:
            for i, tech in enumerate(techniques):
                if tech == "baseline":
                    continue
                paired_t, paired_b = [], []
                for r in results:
                    if "error" not in r and r["technique"] == tech:
                        bl = baseline_by_task.get(r["task_id"])
                        if bl is not None:
                            paired_t.append(r["final_quality"])
                            paired_b.append(bl)
                p = _wilcoxon_p(paired_t, paired_b)
                star = _significance_star(p)
                if star and star != "ns":
                    ymax = max(by_tech[tech]) + 0.03
                    ax.text(i, min(ymax, 1.02), star, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="#228833")

        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Quality")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.1)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.set_yticks(np.arange(0.0, 1.01, 0.05), minor=True)
        ax.yaxis.grid(True, which="major", linestyle=":", color="0.55",
                      linewidth=0.8, alpha=0.8, zorder=0)
        ax.yaxis.grid(True, which="minor", linestyle=":", color="0.75",
                      linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    _draw_panel(axes[0], panel_top,
                "Quality distribution — AgentCodec core operators")
    _draw_panel(axes[1], panel_bot,
                "Quality distribution — soft variants, wider-pool operators, and prior-method baselines")

    _add_model_info(fig, model_info)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_distribution.png")
    fig.savefig(output_dir / "quality_distribution.pdf")
    plt.close(fig)
    logger.info("Saved quality_distribution plot (two-panel split)")


# ===================================================================
# Plot 8: Diversity Order — Failure Rate vs Branches (log scale)
# ===================================================================

def plot_diversity_order(diversity_scaling_results: list[dict], output_dir: Path):
    """Failure rate vs branches (log) + quality vs branches.  With CI."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    by_combining: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_combining_qual: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in diversity_scaling_results:
        c = r["combining"]
        d = r["num_branches"]
        by_combining[c][d].append(r["failure_rate"])
        by_combining_qual[c][d].append(r["combined_quality"])

    palette = {"sc": "#4477AA", "mrc": "#228833", "egc": "#EE6677"}
    labels = {"sc": "SC", "mrc": "MRC", "egc": "EGC"}

    # Left: failure rate
    ax = axes[0]
    for comb in ["sc", "egc", "mrc"]:
        if comb not in by_combining:
            continue
        d_data = by_combining[comb]
        branches = sorted(d_data.keys())
        means = [max(float(np.mean(d_data[d])), 1e-3) for d in branches]
        ax.semilogy(branches, means, "o-", color=palette.get(comb, "#333"),
                     label=labels.get(comb, comb), markersize=7,
                     markeredgecolor="black", markeredgewidth=0.3)
    ax.set_xlabel("Diversity branches (d)")
    ax.set_ylabel("Failure rate (log scale)")
    ax.set_title("Effective diversity order")
    ax.set_ylim(1e-3, 1.1)
    ax.legend()

    # Right: quality
    ax = axes[1]
    for comb in ["sc", "egc", "mrc"]:
        if comb not in by_combining_qual:
            continue
        d_data = by_combining_qual[comb]
        branches = sorted(d_data.keys())
        means, ci_los, ci_his = [], [], []
        for d in branches:
            vals = d_data[d]
            m = float(np.mean(vals))
            lo, hi = _bootstrap_ci(vals)
            means.append(m)
            ci_los.append(m - lo)
            ci_his.append(hi - m)
        ax.errorbar(branches, means, yerr=[ci_los, ci_his],
                     fmt="s-", color=palette.get(comb, "#333"),
                     label=labels.get(comb, comb), capsize=4, capthick=1,
                     markeredgecolor="black", markeredgewidth=0.3, markersize=7)
    ax.set_xlabel("Diversity branches (d)")
    ax.set_ylabel("Mean quality")
    ax.set_title("Quality vs. branch count")
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout(w_pad=2)
    fig.savefig(output_dir / "diversity_order.png")
    fig.savefig(output_dir / "diversity_order.pdf")
    plt.close(fig)
    logger.info("Saved diversity_order plot")


# ===================================================================
# Plot 9: Judge Ablation
# ===================================================================

def plot_judge_ablation(judge_results: list[dict], output_dir: Path):
    """Inter-judge variance, score range, pairwise agreement."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    variances = [r["score_variance"] for r in judge_results]
    ranges = [r["score_range"] for r in judge_results]

    # Left: variance histogram
    ax = axes[0]
    ax.hist(variances, bins=20, color="#4477AA", alpha=0.7, edgecolor="white", lw=0.5)
    ax.axvline(np.mean(variances), color="#CC3311", ls="--", lw=1.2,
               label=f"Mean: {np.mean(variances):.4f}")
    ax.set_xlabel("Inter-judge score variance")
    ax.set_ylabel("Count")
    ax.set_title("Channel estimator reliability")
    ax.legend(fontsize=8)

    # Middle: range histogram
    ax = axes[1]
    ax.hist(ranges, bins=20, color="#EE6677", alpha=0.7, edgecolor="white", lw=0.5)
    ax.axvline(np.mean(ranges), color="#CC3311", ls="--", lw=1.2,
               label=f"Mean: {np.mean(ranges):.3f}")
    ax.set_xlabel("Score range (max $-$ min)")
    ax.set_ylabel("Count")
    ax.set_title("Score disagreement")
    ax.legend(fontsize=8)

    # Right: pairwise scatter
    ax = axes[2]
    judge_names = list(judge_results[0]["scores"].keys()) if judge_results else []
    if len(judge_names) >= 2:
        j1, j2 = judge_names[0], judge_names[1]
        s1 = [r["scores"][j1] for r in judge_results if j1 in r["scores"]]
        s2 = [r["scores"][j2] for r in judge_results if j2 in r["scores"]]
        ax.scatter(s1, s2, alpha=0.45, s=18, c="#4477AA", edgecolors="0.5", linewidths=0.2)
        ax.plot([0, 1], [0, 1], color="#CC3311", ls="--", lw=1, alpha=0.6, label="y = x")
        ax.set_xlabel(f"Judge: {j1}")
        ax.set_ylabel(f"Judge: {j2}")
        ax.set_title("Pairwise agreement")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        if len(s1) > 1:
            corr = float(np.corrcoef(s1, s2)[0, 1])
            ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                       fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.tight_layout(w_pad=2)
    fig.savefig(output_dir / "judge_ablation.png")
    fig.savefig(output_dir / "judge_ablation.pdf")
    plt.close(fig)
    logger.info("Saved judge_ablation plot")


# ===================================================================
# Plot 10: Pareto Frontier + ACM vs Oracle
# ===================================================================

def plot_pareto_acm(acm_oracle_results: dict, all_results: list[dict], output_dir: Path):
    """Pareto frontier (left) and ACM vs oracle scatter (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    by_tech = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            by_tech[r["technique"]].append(r)

    # Left: Pareto frontier
    ax = axes[0]
    all_points = []
    for tech in TECHNIQUE_ORDER:
        if tech not in by_tech:
            continue
        runs = by_tech[tech]
        mc = float(np.mean([r["total_cost_usd"] for r in runs]))
        mq = float(np.mean([r["final_quality"] for r in runs]))
        all_points.append((mc, mq, tech))
        c = TECHNIQUE_COLORS.get(tech, "#333")
        m = TECHNIQUE_MARKERS.get(tech, "o")
        lab = TECHNIQUE_LABELS.get(tech, tech)
        ax.scatter([mc], [mq], c=c, marker=m, s=90,
                   edgecolors="black", linewidths=0.4, label=lab, zorder=5)

    # Frontier line
    sorted_pts = sorted(all_points, key=lambda p: p[0])
    fx, fy = [], []
    best = -1
    for cost, qual, _ in sorted_pts:
        if qual > best:
            fx.append(cost)
            fy.append(qual)
            best = qual
    if fx:
        ax.plot(fx, fy, "k--", alpha=0.4, lw=1.2, label="Pareto frontier", zorder=4)

    ax.set_xlabel("Mean cost (USD)")
    ax.set_ylabel("Mean quality")
    ax.set_title("Quality-cost Pareto frontier")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_ylim(0, 1.05)

    # Right: ACM vs Oracle
    ax = axes[1]
    per_task = acm_oracle_results.get("per_task", {})
    if per_task:
        aq = [v["acm"]["quality"] for v in per_task.values()]
        oq = [v["oracle"]["quality"] for v in per_task.values()]
        ax.scatter(oq, aq, alpha=0.5, s=30, c="#228833",
                   edgecolors="black", linewidths=0.2)
        ax.plot([0, 1], [0, 1], color="#CC3311", ls="--", lw=1, alpha=0.6, label="ACM = Oracle")
        ax.set_xlabel("Oracle quality")
        ax.set_ylabel("ACM quality")
        ax.set_title("ACM vs. oracle routing")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        gap = acm_oracle_results.get("quality_gap", 0)
        savings = acm_oracle_results.get("cost_savings_vs_always_best", 0)
        ax.annotate(f"Gap: {gap:.3f}\nSavings: {savings:.1%}",
                    xy=(0.05, 0.85), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    fig.tight_layout(w_pad=2)
    fig.savefig(output_dir / "pareto_acm.png")
    fig.savefig(output_dir / "pareto_acm.pdf")
    plt.close(fig)
    logger.info("Saved pareto_acm plot")


# ===================================================================
# Plot 10b: ACM oracle-gap analysis (cache-only, no re-run)
# ===================================================================

AGENTCODEC_TECHNIQUES: set[str] = {
    "baseline",
    "diversity_sc", "diversity_mrc", "diversity_egc",
    "diversity_sc_N", "diversity_mrc_discrete_N",
    "harq_cc", "harq_ir",
    "turbo",
    "fountain",
    "fec_0.75", "fec_0.50", "fec_0.33", "fec_0.25",
    "diversity_mrc_soft", "fountain_soft",
}
# Prior-method reproductions excluded from the ACM oracle pool: ACM cannot
# route to these (they are comparison baselines, not techniques in our
# framework), so including them would overstate the achievable gap.
PRIOR_BASELINES: set[str] = {
    "self_consistency", "self_refine", "chain_of_verification",
    "best_of_n", "weighted_bon", "mixture_of_agents",
}


def plot_acm_oracle_gap(results: list[dict], output_dir: Path, model_info: str = ""):
    """
    Per-task ACM vs oracle gap, using ONLY existing cached results.

    The "oracle" is restricted to AgentCodec techniques (AGENTCODEC_TECHNIQUES
    above) -- the set ACM could conceivably route to. Prior-method baselines
    (self_consistency, best_of_n, mixture_of_agents, ...) are EXCLUDED from
    the oracle pool because they are outside ACM's candidate set; including
    them would overstate the achievable routing gap.

    Outputs:
      - plots/acm_oracle_gap.png / .pdf: histogram of per-task gaps,
        with best-fixed / top-3 / per-task oracle reference lines.
      - plots/acm_oracle_gap.txt: headline table for the paper, with a
        separate "prior-baseline oracle" row for context (shows how much
        of the gap could be closed by adopting a prior method).

    Crucially, this uses cache results that were ALREADY produced for the
    quality-vs-cost plot, so no extra LLM calls are needed.
    """
    runs_ok = [r for r in results if "error" not in r]
    if not runs_ok:
        logger.info("Skipped acm_oracle_gap (no results)")
        return

    # Oracle pool: one representative quality per (task_id, technique).
    # If a technique ran the same task multiple times, average.
    pool: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs_ok:
        tid = r.get("task_id")
        tech = r.get("technique", "")
        q = r.get("final_quality")
        if tid is None or q is None:
            continue
        pool[(tid, tech)].append(float(q))

    by_task_tech: dict[str, dict[str, float]] = defaultdict(dict)
    for (tid, tech), qs in pool.items():
        by_task_tech[tid][tech] = float(np.mean(qs))

    # Need ACM on the task to compute the gap.
    acm_tasks = [tid for tid, d in by_task_tech.items() if "acm" in d]
    if not acm_tasks:
        logger.info("Skipped acm_oracle_gap (no ACM results in cache)")
        return

    gaps: list[float] = []
    per_task_rows: list[tuple[str, str, float, str, float]] = []  # tid, acm_tech, acm_q, oracle_tech, oracle_q
    # Oracle pool: AgentCodec techniques only (ACM's candidate set). Prior
    # baselines are tracked separately below for context.
    prior_oracle_qs: list[float] = []  # per-task oracle including priors (context)
    for tid in acm_tasks:
        row = by_task_tech[tid]
        acm_q = row["acm"]
        candidates = {
            t: q for t, q in row.items()
            if t in AGENTCODEC_TECHNIQUES and t != "acm"
        }
        if not candidates:
            continue
        oracle_tech = max(candidates, key=lambda t: candidates[t])
        oracle_q = candidates[oracle_tech]
        gap = oracle_q - acm_q
        gaps.append(gap)
        per_task_rows.append((tid, "acm", acm_q, oracle_tech, oracle_q))
        # Prior-inclusive oracle (for context annotation).
        prior_inclusive = {
            t: q for t, q in row.items() if t not in ("acm", "acm_soft")
        }
        prior_oracle_qs.append(max(prior_inclusive.values()))

    if not gaps:
        logger.info("Skipped acm_oracle_gap (no comparable tasks)")
        return

    arr = np.array(gaps, dtype=float)
    mean_gap = float(arr.mean())
    median_gap = float(np.median(arr))
    mean_acm = float(np.mean([r[2] for r in per_task_rows]))
    mean_oracle = float(np.mean([r[4] for r in per_task_rows]))
    frac_of_oracle = mean_acm / mean_oracle if mean_oracle > 0 else 0.0
    frac_optimal = float(np.mean(arr <= 1e-6))  # ACM matched oracle exactly
    frac_within_5 = float(np.mean(arr <= 0.05))

    # Best-fixed technique: the single AgentCodec technique with the highest
    # mean quality on the comparison set. This is the relevant routing
    # baseline -- any routing algorithm must beat always-picking-this to
    # justify its complexity.
    tech_means: dict[str, float] = {}
    agentcodec_present = set(
        t for tid in acm_tasks for t in by_task_tech[tid]
        if t in AGENTCODEC_TECHNIQUES and t != "acm"
    )
    for tech in agentcodec_present:
        vals = [by_task_tech[tid][tech] for tid in acm_tasks if tech in by_task_tech[tid]]
        if len(vals) == len(acm_tasks):  # only techniques that ran on every task
            tech_means[tech] = float(np.mean(vals))
    best_fixed_tech = max(tech_means, key=lambda t: tech_means[t]) if tech_means else None
    mean_best_fixed = tech_means[best_fixed_tech] if best_fixed_tech else 0.0

    # Top-3 restricted oracle: a practically-achievable upper bound for any
    # routing algorithm that only has to choose among the 3 strongest AgentCodec
    # techniques (what a well-tuned ACM router actually has to decide between).
    top3 = sorted(tech_means, key=lambda t: -tech_means[t])[:3]
    mean_top3_oracle = float(np.mean([
        max(by_task_tech[tid][t] for t in top3 if t in by_task_tech[tid])
        for tid in acm_tasks if any(t in by_task_tech[tid] for t in top3)
    ])) if top3 else 0.0

    # Prior-inclusive oracle (context only -- ACM cannot route to priors).
    mean_prior_oracle = float(np.mean(prior_oracle_qs)) if prior_oracle_qs else 0.0

    # Oracle-winner tally: which technique is the oracle on how many tasks?
    winner_tally: dict[str, int] = defaultdict(int)
    for row in per_task_rows:
        winner_tally[row[3]] += 1

    # Plot: histogram of gaps.
    fig, (ax_hist, ax_tally) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_hist.hist(arr, bins=20, color="#117733", edgecolor="black",
                 linewidth=0.4, alpha=0.85)
    ax_hist.axvline(mean_gap, color="#CC3311", ls="--", lw=1.2,
                    label=f"mean gap = {mean_gap:.3f}")
    ax_hist.axvline(0, color="0.5", ls=":", lw=0.8)
    ax_hist.set_xlabel("Oracle quality $-$ ACM quality")
    ax_hist.set_ylabel("Number of tasks")
    ax_hist.set_title("Per-task ACM vs. oracle-best gap")
    ax_hist.legend(fontsize=8)
    bf_lbl = TECHNIQUE_LABELS.get(best_fixed_tech, best_fixed_tech) if best_fixed_tech else "n/a"
    ax_hist.annotate(
        f"Oracle pool: AgentCodec only (priors excluded)\n"
        f"n = {len(arr)}\n"
        f"ACM = {mean_acm:.3f}\n"
        f"Best-fixed = {mean_best_fixed:.3f}  ({bf_lbl})\n"
        f"Top-3 oracle = {mean_top3_oracle:.3f}\n"
        f"AgentCodec oracle = {mean_oracle:.3f}\n"
        f"(Incl. priors oracle = {mean_prior_oracle:.3f})\n"
        f"ACM / oracle = {frac_of_oracle:.1%}\n"
        f"ACM matches oracle: {frac_optimal:.1%}\n"
        f"ACM within 0.05: {frac_within_5:.1%}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.6),
    )
    # Reference lines at ACM / best-fixed / top-3 / per-task oracle means,
    # placed on a twin axis in the same histogram panel so the reader sees
    # the three-way story (ACM vs best-fixed vs oracle ceiling) at a glance.
    for val, col in [
        (mean_best_fixed, "#0077BB"),
        (mean_top3_oracle, "#EE9933"),
    ]:
        if val > 0:
            ax_hist.axvline(val - mean_acm, color=col, ls=":", lw=1.0, alpha=0.7)

    # Tally: oracle-winner distribution.
    tally_items = sorted(winner_tally.items(), key=lambda x: -x[1])
    techs_t = [t for t, _ in tally_items]
    counts_t = [c for _, c in tally_items]
    cols = [TECHNIQUE_COLORS.get(t, "#999") for t in techs_t]
    labs = [TECHNIQUE_LABELS.get(t, t) for t in techs_t]
    ax_tally.barh(range(len(techs_t)), counts_t, color=cols, edgecolor="black",
                  linewidth=0.4)
    ax_tally.set_yticks(range(len(techs_t)))
    ax_tally.set_yticklabels(labs, fontsize=8)
    ax_tally.invert_yaxis()
    ax_tally.set_xlabel("Number of tasks where this technique was the oracle")
    ax_tally.set_title("Oracle-winner distribution (AgentCodec pool, excl. ACM)")

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "acm_oracle_gap.png")
    fig.savefig(output_dir / "acm_oracle_gap.pdf")
    plt.close(fig)

    # Write a headline text file for the paper.
    summary_path = output_dir / "acm_oracle_gap.txt"
    with open(summary_path, "w") as f:
        f.write("ACM oracle-gap analysis (cache-only, no re-run)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Oracle pool: AgentCodec techniques only (priors excluded).\n\n")
        f.write(f"Tasks analysed:           {len(arr)}\n")
        f.write(f"ACM mean quality:         {mean_acm:.4f}\n")
        f.write(f"Best-fixed quality:       {mean_best_fixed:.4f}  ({best_fixed_tech})\n")
        f.write(f"Top-3 oracle quality:     {mean_top3_oracle:.4f}  ({', '.join(top3)})\n")
        f.write(f"AgentCodec oracle:         {mean_oracle:.4f}\n")
        f.write(f"Incl.-priors oracle:      {mean_prior_oracle:.4f}  (context; ACM cannot route here)\n")
        f.write(f"Mean gap (oracle-ACM):    {mean_gap:.4f}\n")
        f.write(f"Median gap:               {median_gap:.4f}\n")
        f.write(f"ACM - best-fixed:         {mean_acm - mean_best_fixed:+.4f}  "
                f"(routing benefit; positive = routing adds value)\n")
        f.write(f"ACM / oracle:             {frac_of_oracle:.4f}  ({frac_of_oracle:.1%})\n")
        f.write(f"Tasks ACM = oracle:       {frac_optimal:.4f}  ({frac_optimal:.1%})\n")
        f.write(f"Tasks gap <= 0.05:        {frac_within_5:.4f}  ({frac_within_5:.1%})\n\n")
        f.write("Oracle-winner distribution (AgentCodec pool only):\n")
        for t, c in tally_items:
            f.write(f"  {t:<24s}  {c:>4d}  ({c / len(arr):.1%})\n")

    logger.info(
        f"Saved acm_oracle_gap plot. ACM={mean_acm:.3f}  "
        f"best-fixed={mean_best_fixed:.3f} ({best_fixed_tech})  "
        f"top3-oracle={mean_top3_oracle:.3f}  per-task-oracle={mean_oracle:.3f}  "
        f"ACM-vs-best-fixed={mean_acm - mean_best_fixed:+.3f}  (n={len(arr)})"
    )


# ===================================================================
# Plot 11: Prediction Validation
# ===================================================================

def plot_prediction_validation(validation_results: dict, output_dir: Path):
    """Horizontal bar chart: green = pass, red = fail."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    predictions = {k: v for k, v in validation_results.items() if not k.startswith("_")}
    names = list(predictions.keys())
    labels = [v["prediction"][:55] + "..." if len(v["prediction"]) > 55 else v["prediction"]
              for v in predictions.values()]
    passed = [v["pass"] for v in predictions.values()]

    colors = ["#228833" if p else "#CC3311" for p in passed]
    y_pos = range(len(names))

    ax.barh(y_pos, [1] * len(names), color=colors, alpha=0.7,
            edgecolor="0.3", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("Communication-theory prediction validation")

    for i, p in enumerate(passed):
        status = "PASS" if p else "FAIL"
        ax.text(1.05, i, status, va="center", fontsize=9, fontweight="bold",
                color="#228833" if p else "#CC3311")

    summary = validation_results.get("_summary", {})
    n_pass = summary.get("passed", sum(passed))
    n_total = summary.get("total_predictions", len(passed))
    ax.annotate(f"{n_pass}/{n_total} predictions validated",
                xy=(0.5, -0.18), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "prediction_validation.png")
    fig.savefig(output_dir / "prediction_validation.pdf")
    plt.close(fig)
    logger.info("Saved prediction_validation plot")


# ===================================================================
# Plot 12: Hard-Benchmark Bar Charts (main paper — one panel per dataset)
# ===================================================================

DATASET_LABELS = {
    "gsm8k_hard": "GSM8K-Hard",
    "mmlu_hard": "MMLU-Hard",
    "humaneval_hard": "HumanEval-Hard",
}

_HARD_DATASET_PREFIXES = (
    ("gsm8k_hard", "gsm8k_hard_"),
    ("mmlu_hard", "mmlu_hard_"),
    ("humaneval_hard", "humaneval_hard_"),
)


def _split_results_by_hard_dataset(
    results: list[dict],
) -> dict[str, list[dict]]:
    """Bin results by the hard-benchmark prefix carried in their task_id.

    Standard-dataset runs in this repo mix gsm8k_hard / mmlu_hard /
    humaneval_hard task ids in one cache, so the dataset is encoded in the
    id prefix rather than in a separate field. Returns only the keys that
    actually have results, so callers can early-out when the run is on
    curated tasks (no panels to draw).
    """
    out: dict[str, list[dict]] = {k: [] for k, _ in _HARD_DATASET_PREFIXES}
    for r in results:
        tid = r.get("task_id", "") or ""
        for key, pref in _HARD_DATASET_PREFIXES:
            if tid.startswith(pref):
                out[key].append(r)
                break
    return {k: v for k, v in out.items() if v}


def plot_hard_benchmark_bars(
    results: list[dict],
    output_dir: str | Path = "plots",
    model_info: str = "",
):
    """
    Main-paper figure: one bar-chart panel per hard-benchmark dataset.

    Splits ``results`` by ``task_id`` prefix (``gsm8k_hard_*``, ``mmlu_hard_*``,
    ``humaneval_hard_*``) and draws one bar-chart panel per non-empty split.
    Returns silently without writing any file if none of those prefixes are
    present, so it is safe to call unconditionally from ``plot_all_from_results``.

    Each panel: per-technique mean quality with 95% bootstrap CI, dashed line
    at the baseline mean, paired-Wilcoxon-vs-baseline significance stars.
    """
    by_dataset = _split_results_by_hard_dataset(results)
    if not by_dataset:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(by_dataset.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), squeeze=False)
    axes = axes[0]

    for ax, ds_name in zip(axes, datasets, strict=False):
        ds_results = by_dataset[ds_name]

        # Aggregate ACM sub-techniques (idempotent — plot_all_from_results
        # already does this upstream; harmless when called standalone).
        for r in ds_results:
            tech = r.get("technique", "")
            if tech.startswith("acm_soft_"):
                r["technique"] = "acm_soft"
            elif tech.startswith("acm_"):
                r["technique"] = "acm"

        # Group by technique
        by_tech: dict[str, list[float]] = defaultdict(list)
        for r in ds_results:
            if "error" not in r:
                by_tech[r["technique"]].append(r["final_quality"])

        # Order techniques
        techniques = [t for t in TECHNIQUE_ORDER if t in by_tech]
        if "baseline" in by_tech and "baseline" not in techniques:
            techniques.insert(0, "baseline")

        means, ci_los, ci_his, colors, labels = [], [], [], [], []
        for tech in techniques:
            scores = by_tech[tech]
            m = float(np.mean(scores))
            lo, hi = _bootstrap_ci(scores)
            means.append(m)
            ci_los.append(m - lo)
            ci_his.append(hi - m)
            colors.append(TECHNIQUE_COLORS.get(tech, "#999"))
            labels.append(TECHNIQUE_LABELS.get(tech, tech))

        x = np.arange(len(techniques))
        ax.bar(x, means, yerr=[ci_los, ci_his], capsize=3,
                      color=colors, edgecolor="0.3", linewidth=0.5,
                      error_kw=dict(lw=1.2, capthick=1))

        # Significance stars vs baseline (paired Wilcoxon, within this split)
        baseline_scores_all: dict[str, float] = {}
        for r in ds_results:
            if "error" not in r and r["technique"] == "baseline":
                baseline_scores_all[r["task_id"]] = r["final_quality"]

        if baseline_scores_all:
            for i, tech in enumerate(techniques):
                if tech == "baseline":
                    continue
                paired_t, paired_b = [], []
                for r in ds_results:
                    if "error" not in r and r["technique"] == tech:
                        bl = baseline_scores_all.get(r["task_id"])
                        if bl is not None:
                            paired_t.append(r["final_quality"])
                            paired_b.append(bl)
                p = _wilcoxon_p(paired_t, paired_b)
                star = _significance_star(p)
                if star and star != "ns":
                    ax.text(i, means[i] + ci_his[i] + 0.015, star,
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#228833")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean quality (95% CI)")
        ax.set_title(DATASET_LABELS.get(ds_name, ds_name))
        ax.set_ylim(0, 1.05)

        # Baseline reference line
        if "baseline" in by_tech:
            bl_mean = float(np.mean(by_tech["baseline"]))
            ax.axhline(bl_mean, color="#BBBBBB", ls="--", lw=1, zorder=0)

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "hard_benchmark_bars.png")
    fig.savefig(output_dir / "hard_benchmark_bars.pdf")
    plt.close(fig)
    logger.info("Saved hard_benchmark_bars plot")


# ===================================================================
# Plot 12b: Curated-Task Category Bar Charts (4 panels: qa / reasoning /
# creative / code) — same layout as Plot 12 but split by task_category over
# the 69-task curated set rather than by hard-benchmark id prefix.
# ===================================================================

CURATED_CATEGORIES = ("qa", "reasoning", "creative", "code")
CATEGORY_LABELS = {
    "qa": "QA",
    "reasoning": "Reasoning",
    "creative": "Creative",
    "code": "Code",
}


def _split_curated_results_by_category(
    results: list[dict],
) -> dict[str, list[dict]]:
    """Bin curated-task results by ``task_category``.

    Excludes task ids that belong to the hard-benchmark splits (handled by the
    Plot-12 figure), so on a mixed cache this view stays scoped to curated
    tasks. Returns only categories that actually have results.
    """
    out: dict[str, list[dict]] = {c: [] for c in CURATED_CATEGORIES}
    for r in results:
        tid = r.get("task_id", "") or ""
        if any(tid.startswith(pref) for _, pref in _HARD_DATASET_PREFIXES):
            continue
        cat = r.get("task_category")
        if cat in out:
            out[cat].append(r)
    return {k: v for k, v in out.items() if v}


def plot_curated_category_bars(
    results: list[dict],
    output_dir: str | Path = "plots",
    model_info: str = "",
):
    """
    Per-category bar chart for the curated-task set: one panel per
    ``task_category`` (qa / reasoning / creative / code), with the same
    layout as ``plot_hard_benchmark_bars``: per-technique mean quality with
    95% bootstrap CI, dashed line at the baseline mean, paired-Wilcoxon-vs-
    baseline significance stars.

    Returns silently without writing any file if no curated results are
    present, so it is safe to call unconditionally from
    ``plot_all_from_results``.
    """
    by_cat = _split_curated_results_by_category(results)
    if not by_cat:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cats = [c for c in CURATED_CATEGORIES if c in by_cat]
    len(cats)
    # Create a 2x2 grid. Adjusting figsize for 2 rows and 2 columns (approx 5.5*2 by 4.5*2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Flatten the 2D axes array to a 1D array for easy iteration
    axes_flat = axes.flatten()

    # for i, ax in enumerate(axes_flat):

    for ax, cat in zip(axes_flat, cats, strict=False):
        cat_results = by_cat[cat]

        # Aggregate ACM sub-techniques (idempotent).
        for r in cat_results:
            tech = r.get("technique", "")
            if tech.startswith("acm_soft_"):
                r["technique"] = "acm_soft"
            elif tech.startswith("acm_"):
                r["technique"] = "acm"

        # Group by technique
        by_tech: dict[str, list[float]] = defaultdict(list)
        for r in cat_results:
            if "error" not in r:
                by_tech[r["technique"]].append(r["final_quality"])

        techniques = [t for t in TECHNIQUE_ORDER if t in by_tech]
        if "baseline" in by_tech and "baseline" not in techniques:
            techniques.insert(0, "baseline")

        means, ci_los, ci_his, colors, labels = [], [], [], [], []
        for tech in techniques:
            scores = by_tech[tech]
            m = float(np.mean(scores))
            lo, hi = _bootstrap_ci(scores)
            means.append(m)
            ci_los.append(m - lo)
            ci_his.append(hi - m)
            colors.append(TECHNIQUE_COLORS.get(tech, "#999"))
            labels.append(TECHNIQUE_LABELS.get(tech, tech))

        x = np.arange(len(techniques))
        ax.bar(x, means, yerr=[ci_los, ci_his], capsize=3,
               color=colors, edgecolor="0.3", linewidth=0.5,
               error_kw=dict(lw=1.2, capthick=1))

        # Significance stars vs baseline (paired Wilcoxon, within this category)
        baseline_scores_all: dict[str, float] = {}
        for r in cat_results:
            if "error" not in r and r["technique"] == "baseline":
                baseline_scores_all[r["task_id"]] = r["final_quality"]

        if baseline_scores_all:
            for i, tech in enumerate(techniques):
                if tech == "baseline":
                    continue
                paired_t, paired_b = [], []
                for r in cat_results:
                    if "error" not in r and r["technique"] == tech:
                        bl = baseline_scores_all.get(r["task_id"])
                        if bl is not None:
                            paired_t.append(r["final_quality"])
                            paired_b.append(bl)
                p = _wilcoxon_p(paired_t, paired_b)
                star = _significance_star(p)
                if star and star != "ns":
                    ax.text(i, means[i] + ci_his[i] + 0.015, star,
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#228833")

        n_tasks = len({r["task_id"] for r in cat_results
                       if r.get("technique") == "baseline"})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean quality (95% CI)")
        title = CATEGORY_LABELS.get(cat, cat)
        if n_tasks:
            title = f"{title} (n={n_tasks})"
        ax.set_title(title)
        ax.set_ylim(0, 1.05)

        if "baseline" in by_tech:
            bl_mean = float(np.mean(by_tech["baseline"]))
            ax.axhline(bl_mean, color="#BBBBBB", ls="--", lw=1, zorder=0)

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "curated_category_bars.png")
    fig.savefig(output_dir / "curated_category_bars.pdf")
    plt.close(fig)
    logger.info("Saved curated_category_bars plot")


# ===================================================================
# Plot 13: Hard-Benchmark Pareto Scatter (appendix — one panel per dataset)
# ===================================================================

def plot_hard_benchmark_pareto(
    results: list[dict],
    output_dir: str | Path = "plots",
    model_info: str = "",
):
    """
    Appendix figure: quality vs. normalised cost overhead per hard-benchmark
    dataset (one Pareto scatter panel per dataset).

    Splits ``results`` by ``task_id`` prefix as in
    ``plot_hard_benchmark_bars`` and skips silently if no ``*_hard_*`` task
    ids are present.
    """
    by_dataset = _split_results_by_hard_dataset(results)
    if not by_dataset:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(by_dataset.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.5), squeeze=False)
    axes = axes[0]

    for ax, ds_name in zip(axes, datasets, strict=False):
        ds_results = by_dataset[ds_name]

        # Aggregate ACM sub-techniques (idempotent).
        for r in ds_results:
            tech = r.get("technique", "")
            if tech.startswith("acm_soft_"):
                r["technique"] = "acm_soft"
            elif tech.startswith("acm_"):
                r["technique"] = "acm"

        _ensure_normalized(ds_results)

        by_tech: dict[str, list[dict]] = defaultdict(list)
        for r in ds_results:
            if "error" not in r:
                by_tech[r["technique"]].append(r)

        mean_xs: list[float] = []
        all_points: list[tuple[float, float, str]] = []

        for tech in TECHNIQUE_ORDER:
            if tech not in by_tech:
                continue
            runs = by_tech[tech]
            overheads = [r.get("cost_overhead", 1.0) for r in runs]
            quals = [r["final_quality"] for r in runs]
            mx = float(np.mean(overheads))
            my = float(np.mean(quals))
            ci_lo, ci_hi = _bootstrap_ci(quals)
            mean_xs.append(mx)
            all_points.append((mx, my, tech))

            c = TECHNIQUE_COLORS.get(tech, "#333")
            m = TECHNIQUE_MARKERS.get(tech, "o")
            lab = TECHNIQUE_LABELS.get(tech, tech)

            # Individual points (faded)
            ax.scatter(overheads, quals, c=c, marker=m, alpha=0.12, s=10, linewidths=0)
            # Error bar
            ax.errorbar(mx, my, yerr=[[my - ci_lo], [ci_hi - my]],
                        fmt="none", ecolor=c, capsize=4, capthick=1.2,
                        elinewidth=1.6, zorder=4)
            # Mean marker
            ax.scatter([mx], [my], c=c, marker=m, s=60,
                       edgecolors="black", linewidths=0.5, label=lab, zorder=5)

        # Pareto frontier
        sorted_pts = sorted(all_points, key=lambda p: p[0])
        fx, fy = [], []
        best = -1.0
        for cost, qual, _ in sorted_pts:
            if qual > best:
                fx.append(cost)
                fy.append(qual)
                best = qual
        if fx:
            ax.plot(fx, fy, "k--", alpha=0.35, lw=1.2, zorder=3)

        ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
        ax.set_xlabel(r"Cost overhead $\rho$ ($\times$baseline)")
        ax.set_ylabel("Quality")
        ax.set_title(DATASET_LABELS.get(ds_name, ds_name))
        ax.set_ylim(0, 1.05)
        if mean_xs:
            ax.set_xlim(0, max(mean_xs) * 1.35)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 5), fontsize=7,
                   framealpha=0.9, edgecolor="0.7",
                   bbox_to_anchor=(0.5, -0.02))

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2.5, rect=[0, 0.06, 1, 0.97])
    fig.savefig(output_dir / "hard_benchmark_pareto.png")
    fig.savefig(output_dir / "hard_benchmark_pareto.pdf")
    plt.close(fig)
    logger.info("Saved hard_benchmark_pareto plot")


# ===================================================================
# Plot 13b: Curated-Task Category Pareto Scatter (one panel per category
# over the curated 69-task set; same layout as Plot 13).
# ===================================================================

def plot_curated_category_pareto(
    results: list[dict],
    output_dir: str | Path = "plots",
    model_info: str = "",
):
    """
    Quality vs. normalised cost overhead per task category for the curated
    set (qa / reasoning / creative / code), one Pareto scatter panel per
    category. Mirrors ``plot_hard_benchmark_pareto`` and skips silently when
    no curated results are present.
    """
    by_cat = _split_curated_results_by_category(results)
    if not by_cat:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cats = [c for c in CURATED_CATEGORIES if c in by_cat]
    len(cats)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Flatten the 2D axes array to a 1D array for easy iteration
    axes_flat = axes.flatten()

    axes = axes_flat

    # for i, ax in enumerate(axes_flat):

    for ax, cat in zip(axes_flat, cats, strict=False):
        cat_results = by_cat[cat]

        # Aggregate ACM sub-techniques (idempotent).
        for r in cat_results:
            tech = r.get("technique", "")
            if tech.startswith("acm_soft_"):
                r["technique"] = "acm_soft"
            elif tech.startswith("acm_"):
                r["technique"] = "acm"

        _ensure_normalized(cat_results)

        by_tech: dict[str, list[dict]] = defaultdict(list)
        for r in cat_results:
            if "error" not in r:
                by_tech[r["technique"]].append(r)

        mean_xs: list[float] = []
        all_points: list[tuple[float, float, str]] = []

        for tech in TECHNIQUE_ORDER:
            if tech not in by_tech:
                continue
            runs = by_tech[tech]
            overheads = [r.get("cost_overhead", 1.0) for r in runs]
            quals = [r["final_quality"] for r in runs]
            mx = float(np.mean(overheads))
            my = float(np.mean(quals))
            ci_lo, ci_hi = _bootstrap_ci(quals)
            mean_xs.append(mx)
            all_points.append((mx, my, tech))

            c = TECHNIQUE_COLORS.get(tech, "#333")
            m = TECHNIQUE_MARKERS.get(tech, "o")
            lab = TECHNIQUE_LABELS.get(tech, tech)

            ax.scatter(overheads, quals, c=c, marker=m, alpha=0.12, s=10, linewidths=0)
            ax.errorbar(mx, my, yerr=[[my - ci_lo], [ci_hi - my]],
                        fmt="none", ecolor=c, capsize=4, capthick=1.2,
                        elinewidth=1.6, zorder=4)
            ax.scatter([mx], [my], c=c, marker=m, s=60,
                       edgecolors="black", linewidths=0.5, label=lab, zorder=5)

        # Pareto frontier
        sorted_pts = sorted(all_points, key=lambda p: p[0])
        fx, fy = [], []
        best = -1.0
        for cost, qual, _ in sorted_pts:
            if qual > best:
                fx.append(cost)
                fy.append(qual)
                best = qual
        if fx:
            ax.plot(fx, fy, "k--", alpha=0.35, lw=1.2, zorder=3)

        n_tasks = len({r["task_id"] for r in cat_results
                       if r.get("technique") == "baseline"})
        ax.axvline(1.0, color="0.5", ls="--", lw=0.8, zorder=0)
        ax.set_xlabel(r"Cost overhead $\rho$ ($\times$baseline)")
        ax.set_ylabel("Quality")
        title = CATEGORY_LABELS.get(cat, cat)
        if n_tasks:
            title = f"{title} (n={n_tasks})"
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        if mean_xs:
            ax.set_xlim(0, max(mean_xs) * 1.35)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 5), fontsize=7,
                   framealpha=0.9, edgecolor="0.7",
                   bbox_to_anchor=(0.5, -0.02))

    _add_model_info(fig, model_info)
    fig.tight_layout(w_pad=2.5, rect=[0, 0.06, 1, 0.97])
    fig.savefig(output_dir / "curated_category_pareto.png")
    fig.savefig(output_dir / "curated_category_pareto.pdf")
    plt.close(fig)
    logger.info("Saved curated_category_pareto plot")
