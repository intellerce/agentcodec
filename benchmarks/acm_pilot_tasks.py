"""
Pilot task sets for cross-distribution training of the ACM learned router.

Why this file exists
--------------------
The current `acm_learned` router is fit on the same 69 curated tasks it is
later evaluated on. K-fold CV at fit time gives an honest in-distribution
estimate (`cv_mean_q`), but the *deployed* weights have seen all 69 tasks
during training, so any `run_benchmark.py` invocation against the curated
set produces an in-sample cache. To break that population coupling we
define two cross-distribution protocols, each backed by its own training
pilot:

- Direction A (downloaded -> curated):
    Train pilot = a random sample of MMLU + GSM8K + HumanEval.
    Eval set   = the 69 curated tasks (`benchmarks.tasks.get_all_tasks`).

- Direction B (curated -> downloaded):
    Train pilot = the 69 curated tasks.
    Eval set   = a held-out sample of MMLU + GSM8K + HumanEval, *disjoint*
                 from Direction A's train pilot when seeded identically.

Because train and eval populations are disjoint in both directions, the
acm_learned cache produced by `run_benchmark.py` under either protocol is
genuine out-of-distribution evaluation, not an in-sample fit.

Caveats
-------
1. Category coverage. The downloaded sources only cover QA (MMLU), reasoning
   (GSM8K), and code (HumanEval). They have no creative tasks, so the
   downloaded pilot provides no training signal for the creative one-hot.
   Direction A's deployed router will fall back to bias + difficulty terms
   on creative inputs at eval time.
2. Sample size. The multinomial logit has ~6 features and 8-11 candidate
   classes; ~300+ training examples is the rule of thumb for a stable fit.
   The defaults below (100 per dataset -> 300 total in Direction A; 69 in
   Direction B) bracket that floor.
3. Cache requirement. Training the router needs per-technique quality data
   on the pilot tasks. This file only defines task *lists*. Generating the
   corresponding caches requires a benchmark run on these task lists.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from benchmarks.download_datasets import load_dataset_tasks
from benchmarks.tasks import get_all_tasks

if TYPE_CHECKING:
    from agentcodec.models import TaskItem

logger = logging.getLogger(__name__)


DOWNLOADED_DATASETS = ("mmlu", "gsm8k", "humaneval")
DEFAULT_PILOT_SIZE_PER_DATASET = 100
DEFAULT_HOLDOUT_SIZE_PER_DATASET = 100
DEFAULT_SEED = 20260423


def get_downloaded_split(
    n_train_per_dataset: int = DEFAULT_PILOT_SIZE_PER_DATASET,
    n_eval_per_dataset: int = DEFAULT_HOLDOUT_SIZE_PER_DATASET,
    data_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list["TaskItem"], list["TaskItem"]]:
    """Return a deterministic, disjoint (train_pilot, eval_holdout) split of
    MMLU + GSM8K + HumanEval.

    The same seed always yields the same partition, so Direction A's training
    cache and Direction B's eval cache are guaranteed disjoint as long as
    callers use this single helper to source their tasks.
    """
    rng = random.Random(seed)
    train_pilot: list["TaskItem"] = []
    eval_holdout: list["TaskItem"] = []
    needed = n_train_per_dataset + n_eval_per_dataset

    for ds in DOWNLOADED_DATASETS:
        kwargs = {"data_dir": data_dir} if data_dir else {}
        try:
            all_tasks = load_dataset_tasks(ds, n=None, **kwargs)
        except FileNotFoundError as e:
            logger.warning(
                "Skipping %s: %s. Run: python benchmarks/download_datasets.py "
                "--datasets %s", ds, e, ds,
            )
            continue

        if len(all_tasks) < needed:
            logger.warning(
                "%s has only %d tasks; requested %d (train=%d + eval=%d). "
                "Splitting proportionally.",
                ds, len(all_tasks), needed,
                n_train_per_dataset, n_eval_per_dataset,
            )

        indices = list(range(len(all_tasks)))
        rng.shuffle(indices)
        n_tr = min(n_train_per_dataset, len(indices))
        n_ev = min(n_eval_per_dataset, len(indices) - n_tr)
        train_pilot.extend(all_tasks[i] for i in indices[:n_tr])
        eval_holdout.extend(all_tasks[i] for i in indices[n_tr:n_tr + n_ev])

    return train_pilot, eval_holdout


def get_downloaded_pilot(
    n_per_dataset: int = DEFAULT_PILOT_SIZE_PER_DATASET,
    data_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> list["TaskItem"]:
    """Direction A training pilot: sample from the downloaded datasets.

    The router fit on this pilot is evaluated on the 69 curated tasks via
    `run_benchmark.py`. Disjoint from `get_downloaded_holdout()` under the
    same seed.
    """
    pilot, _ = get_downloaded_split(
        n_train_per_dataset=n_per_dataset,
        n_eval_per_dataset=DEFAULT_HOLDOUT_SIZE_PER_DATASET,
        data_dir=data_dir,
        seed=seed,
    )
    return pilot


def get_downloaded_holdout(
    n_per_dataset: int = DEFAULT_HOLDOUT_SIZE_PER_DATASET,
    data_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> list["TaskItem"]:
    """Direction B eval set: held-out sample of the downloaded datasets.

    The router fit on the 69 curated tasks (`get_curated_pilot()`) is
    evaluated on this set. Disjoint from `get_downloaded_pilot()` under
    the same seed.
    """
    _, holdout = get_downloaded_split(
        n_train_per_dataset=DEFAULT_PILOT_SIZE_PER_DATASET,
        n_eval_per_dataset=n_per_dataset,
        data_dir=data_dir,
        seed=seed,
    )
    return holdout


def get_curated_pilot(min_difficulty: str | None = "hard") -> list["TaskItem"]:
    """Direction B training pilot: the 69 curated tasks.

    Already cached as `results/cache_deepseek14_phi314_gemma4_31/` under the
    main paper's model setup, so this direction's training step is just
    `python -m scripts.train_acm_router <that cache>` — no extra run needed
    on the train side.
    """
    return get_all_tasks(
        include_standard=False,
        use_downloaded=False,
        include_hard=True,
        min_difficulty=min_difficulty,
    )


__all__ = [
    "get_downloaded_split",
    "get_downloaded_pilot",
    "get_downloaded_holdout",
    "get_curated_pilot",
    "DOWNLOADED_DATASETS",
    "DEFAULT_PILOT_SIZE_PER_DATASET",
    "DEFAULT_HOLDOUT_SIZE_PER_DATASET",
    "DEFAULT_SEED",
]
