"""Per-category prompt augmentation in the benchmark runner.

Covers the `category_prompts` block on ``ExperimentConfig`` and the
non-mutating ``BenchmarkRunner._augment_task_for_category`` hook:

  * eager validation rejects malformed specs at construction time;
  * ``system_prompt`` / ``user_prompt_template`` rewrite generation only;
  * the original ``TaskItem`` (shared across techniques / repeats / parallel
    workers) is never mutated, and ``prompt`` / ``reference`` — what the
    judge and the deterministic scorers grade against — are preserved.

The shipped ``configs/ollama_nemotron_devstral_glm5.1_datasets_combined.yaml``
is also smoke-loaded so its category_prompts block stays valid.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentcodec.models import TaskCategory, TaskItem
from agentcodec.runner import BenchmarkRunner, ExperimentConfig


def _runner_with(category_prompts: dict) -> BenchmarkRunner:
    """A BenchmarkRunner stub carrying only what the augmentation needs.

    We bypass ``__init__`` (which builds channels / validates pricing) and
    attach the config that ``_augment_task_for_category`` reads from
    (``self.config.category_prompts``) — keeping the test free of network
    and model-cost concerns while exercising the real attribute path.
    """
    cfg = ExperimentConfig(
        models=[{"model": "gpt-4o-mini", "temperature": 0.7}],
        judge_model="gpt-4o-mini",
        category_prompts=category_prompts,
    )
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.config = cfg
    return runner


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_none_and_empty_normalize_to_empty() -> None:
    assert ExperimentConfig(category_prompts=None).category_prompts == {}
    assert ExperimentConfig(category_prompts={}).category_prompts == {}


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        ExperimentConfig(category_prompts={"banana": {"mode": "as_is"}})


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        ExperimentConfig(category_prompts={"qa": {"mode": "rewrite_everything"}})


def test_system_prompt_requires_text() -> None:
    with pytest.raises(ValueError, match="non-empty 'system_prompt'"):
        ExperimentConfig(category_prompts={"qa": {"mode": "system_prompt"}})
    with pytest.raises(ValueError, match="non-empty 'system_prompt'"):
        ExperimentConfig(
            category_prompts={"qa": {"mode": "system_prompt", "system_prompt": "  "}}
        )


def test_user_template_requires_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{prompt\}"):
        ExperimentConfig(
            category_prompts={
                "reasoning": {
                    "mode": "user_prompt_template",
                    "user_prompt_template": "no placeholder here",
                }
            }
        )


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentConfig(
            category_prompts={"qa": {"mode": "as_is", "systemprompt": "typo"}}
        )


# ---------------------------------------------------------------------------
# Augmentation behavior
# ---------------------------------------------------------------------------


def test_system_prompt_mode_is_non_mutating_and_scoring_safe() -> None:
    runner = _runner_with(
        {"qa": {"mode": "system_prompt", "system_prompt": "Answer: <LETTER>"}}
    )
    task = TaskItem(
        id="q1", category=TaskCategory.QA,
        prompt="2+2? A) 3 B) 4", reference="B",
    )
    aug = runner._augment_task_for_category(task)

    assert aug is not task                       # new object
    assert task.request.system is None           # original untouched
    assert aug.request.system == "Answer: <LETTER>"
    # Scoring inputs preserved verbatim.
    assert aug.prompt == task.prompt
    assert aug.reference == task.reference
    assert aug.score_mode == task.score_mode


def test_user_prompt_template_rewrites_user_turn_only() -> None:
    runner = _runner_with(
        {
            "reasoning": {
                "mode": "user_prompt_template",
                "user_prompt_template": "Solve step by step:\n{prompt}\nFinal number only.",
            }
        }
    )
    task = TaskItem(
        id="r1", category=TaskCategory.REASONING,
        prompt="If A=2 and B=3, what is A+B?", reference="5",
    )
    aug = runner._augment_task_for_category(task)

    user_text = aug.request.last_user_text
    assert "Solve step by step:" in user_text
    assert "If A=2 and B=3, what is A+B?" in user_text   # {prompt} substituted
    # The prompt the scorer/judge see is still the raw question.
    assert aug.prompt == task.prompt
    assert task.request.last_user_text == task.prompt    # original untouched


def test_as_is_and_unconfigured_are_noops() -> None:
    runner = _runner_with({"creative": {"mode": "as_is"}})
    creative = TaskItem(
        id="c1", category=TaskCategory.CREATIVE, prompt="Write a haiku.",
    )
    code = TaskItem(id="k1", category=TaskCategory.CODE, prompt="def f(): ...")
    # as_is → same object; unconfigured category → same object.
    assert runner._augment_task_for_category(creative) is creative
    assert runner._augment_task_for_category(code) is code


def test_no_rules_returns_same_task() -> None:
    runner = _runner_with({})
    task = TaskItem(id="q", category=TaskCategory.QA, prompt="hi", reference="A")
    assert runner._augment_task_for_category(task) is task


# ---------------------------------------------------------------------------
# Shipped config stays valid
# ---------------------------------------------------------------------------


def test_shipped_combined_config_category_prompts_valid() -> None:
    path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "ollama_nemotron_devstral_glm5.1_datasets_combined.yaml"
    )
    if not path.exists():
        pytest.skip("combined benchmark config not present")
    cfg = yaml.safe_load(path.read_text())
    # Mirror run_benchmark.py's pre-construction pops.
    for k in (
        "include_standard", "use_downloaded", "downloaded_n", "data_dir",
        "include_curated", "include_hard", "plot_dir",
    ):
        cfg.pop(k, None)
    ec = ExperimentConfig(**cfg)
    # All four categories present and each names a valid mode.
    assert set(ec.category_prompts) == {"qa", "reasoning", "creative", "code"}
    for spec in ec.category_prompts.values():
        assert spec["mode"] in {"as_is", "system_prompt", "user_prompt_template"}
