#!/usr/bin/env python3
"""
Download standard benchmark datasets for AgentCodec evaluation.

Downloads and converts to TaskItem-compatible JSON:
  - MMLU (Massive Multitask Language Understanding)
  - GSM8K (Grade School Math 8K)
  - HumanEval (OpenAI code generation)

Usage:
    # Download all datasets
    python benchmarks/download_datasets.py

    # Download specific datasets
    python benchmarks/download_datasets.py --datasets mmlu gsm8k humaneval

    # Control sample sizes (for faster experiments)
    python benchmarks/download_datasets.py --mmlu-n 100 --gsm8k-n 200 --humaneval-n 50

    # Download to custom directory
    python benchmarks/download_datasets.py --output-dir data/benchmarks

After downloading, use in your config or code:
    from benchmarks.loader import load_dataset_tasks
    tasks = load_dataset_tasks("mmlu", n=100)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_mmlu(output_dir: Path, n: int | None = None) -> Path:
    """
    Download MMLU from HuggingFace datasets.

    Source: https://huggingface.co/datasets/cais/mmlu
    Fallback: https://huggingface.co/datasets/tasksource/mmlu

    Each item becomes a multiple-choice QA task.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _install_datasets()
        from datasets import load_dataset

    log.info("Downloading MMLU from HuggingFace...")
    # Use the 'all' config which merges all subjects
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    choices = ["A", "B", "C", "D"]
    tasks = []

    for i, row in enumerate(ds):
        if n and i >= n:
            break

        question = row["question"]
        options = row["choices"]
        answer_idx = row["answer"]
        subject = row.get("subject", "unknown")

        # Format as multiple-choice prompt
        prompt_lines = [question]
        for j, opt in enumerate(options):
            prompt_lines.append(f"({choices[j]}) {opt}")
        prompt_lines.append("Answer with the letter only.")

        tasks.append({
            "id": f"mmlu_{i+1:04d}",
            "category": "qa",
            "prompt": "\n".join(prompt_lines),
            "reference": choices[answer_idx],
            "score_mode": "exact_letter",
            "metadata": {
                "source": "mmlu",
                "subject": subject,
                "answer_index": answer_idx,
            },
        })

    out_path = output_dir / "mmlu.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    log.info(f"MMLU: saved {len(tasks)} tasks to {out_path}")
    return out_path


def download_gsm8k(output_dir: Path, n: int | None = None) -> Path:
    """
    Download GSM8K from HuggingFace datasets.

    Source: https://huggingface.co/datasets/openai/gsm8k
    Each item is a math word problem with a step-by-step solution.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _install_datasets()
        from datasets import load_dataset

    log.info("Downloading GSM8K from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    tasks = []

    for i, row in enumerate(ds):
        if n and i >= n:
            break

        question = row["question"]
        answer_text = row["answer"]

        # GSM8K answers end with "#### <number>"
        numerical_answer = None
        if "####" in answer_text:
            numerical_answer = answer_text.split("####")[-1].strip()
            numerical_answer = numerical_answer.replace(",", "")
            try:
                numerical_answer = float(numerical_answer)
                if numerical_answer == int(numerical_answer):
                    numerical_answer = int(numerical_answer)
            except ValueError:
                pass

        tasks.append({
            "id": f"gsm8k_{i+1:04d}",
            "category": "reasoning",
            "prompt": question,
            "reference": answer_text,
            "score_mode": "numeric",
            "metadata": {
                "source": "gsm8k",
                "answer": numerical_answer,
            },
        })

    out_path = output_dir / "gsm8k.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    log.info(f"GSM8K: saved {len(tasks)} tasks to {out_path}")
    return out_path


def download_humaneval(output_dir: Path, n: int | None = None) -> Path:
    """
    Download HumanEval from HuggingFace datasets.

    Source: https://huggingface.co/datasets/openai/openai_humaneval
    Each item is a Python function completion with test cases.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _install_datasets()
        from datasets import load_dataset

    log.info("Downloading HumanEval from HuggingFace...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    tasks = []

    for i, row in enumerate(ds):
        if n and i >= n:
            break

        task_id = row["task_id"]           # e.g. "HumanEval/0"
        prompt_code = row["prompt"]         # function signature + docstring
        canonical = row["canonical_solution"]
        test_code = row["test"]
        entry_point = row["entry_point"]

        # Build a natural-language prompt from the code signature
        prompt = (
            f"Write a Python function to complete the following:\n\n"
            f"```python\n{prompt_code.rstrip()}\n```\n\n"
            f"Return only the complete function implementation."
        )

        # Extract individual assert statements from test code for metadata
        test_cases = []
        for line in test_code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("assert"):
                test_cases.append(stripped)

        tasks.append({
            "id": f"humaneval_{i+1:04d}",
            "category": "code",
            "prompt": prompt,
            "reference": prompt_code + canonical,
            "metadata": {
                "source": "humaneval",
                "task_id": task_id,
                "entry_point": entry_point,
                "test_code": test_code,
                "test_cases": test_cases[:5],  # keep first 5 for quick validation
                "prompt_code": prompt_code,
            },
        })

    out_path = output_dir / "humaneval.json"
    out_path.write_text(json.dumps(tasks, indent=2))
    log.info(f"HumanEval: saved {len(tasks)} tasks to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Loader — use downloaded datasets in the benchmark
# ---------------------------------------------------------------------------

def load_dataset_tasks(
    dataset: str,
    n: int | None = None,
    data_dir: str | Path | None = None,
) -> list:
    """
    Load downloaded dataset as TaskItem objects.

    Args:
        dataset: one of "mmlu", "gsm8k", "humaneval"
        n: max number of tasks to load (None = all)
        data_dir: directory containing the JSON files (default: benchmarks/data/)

    Returns:
        list of TaskItem instances
    """
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agentcodec.models import TaskCategory, TaskItem

    category_map = {
        "qa": TaskCategory.QA,
        "reasoning": TaskCategory.REASONING,
        "creative": TaskCategory.CREATIVE,
        "code": TaskCategory.CODE,
    }

    data_dir = Path(data_dir) if data_dir else DEFAULT_OUTPUT_DIR
    path = data_dir / f"{dataset}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python benchmarks/download_datasets.py --datasets {dataset}"
        )

    with open(path) as f:
        raw = json.load(f)

    tasks = []
    for item in raw[:n]:
        tasks.append(TaskItem(
            id=item["id"],
            category=category_map[item["category"]],
            prompt=item["prompt"],
            reference=item.get("reference"),
            score_mode=item.get("score_mode"),
            metadata=item.get("metadata", {}),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_datasets():
    """Install the HuggingFace datasets library if missing."""
    import subprocess
    log.info("Installing 'datasets' library (required for downloading)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "datasets"],
        stdout=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for AgentCodec evaluation"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["mmlu", "gsm8k", "humaneval", "all"],
        default=["all"],
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--mmlu-n", type=int, default=None,
        help="Max MMLU questions to keep (default: all ~14k)",
    )
    parser.add_argument(
        "--gsm8k-n", type=int, default=None,
        help="Max GSM8K problems to keep (default: all ~1.3k)",
    )
    parser.add_argument(
        "--humaneval-n", type=int, default=None,
        help="Max HumanEval tasks to keep (default: all 164)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["mmlu", "gsm8k", "humaneval"]

    log.info(f"Output directory: {output_dir}")
    log.info(f"Datasets: {datasets}")

    results = {}
    if "mmlu" in datasets:
        results["mmlu"] = download_mmlu(output_dir, args.mmlu_n)
    if "gsm8k" in datasets:
        results["gsm8k"] = download_gsm8k(output_dir, args.gsm8k_n)
    if "humaneval" in datasets:
        results["humaneval"] = download_humaneval(output_dir, args.humaneval_n)

    log.info("")
    log.info("=" * 50)
    log.info("Download complete!")
    log.info("=" * 50)
    for name, path in results.items():
        size = path.stat().st_size / 1024
        with open(path) as f:
            count = len(json.load(f))
        log.info(f"  {name:>12}: {count:>6} tasks  ({size:.0f} KB)")
    log.info("")
    log.info("Usage in Python:")
    log.info("  from benchmarks.download_datasets import load_dataset_tasks")
    log.info('  tasks = load_dataset_tasks("mmlu", n=100)')
    log.info("")
    log.info("Or run benchmark with downloaded data:")
    log.info("  python run_benchmark.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
