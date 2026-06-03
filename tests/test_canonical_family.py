"""Tests for the shared model-family canonicalizer and size fingerprint."""

from __future__ import annotations

import pytest

from agentcodec.routing import canonical_family, parse_params_b, parse_quant
from agentcodec.routing.remote import _derive_user_config


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("nvidia/Llama-3.1-Nemotron-70B-Instruct", "nemotron"),
        ("ollama:nemotron:70b", "nemotron"),
        ("mistralai/Devstral-Small-2505", "devstral"),
        ("devstral-22b", "devstral"),
        ("zhipuai/glm-5.1-32b", "glm-5.1"),
        ("ollama:glm:5.1", "glm-5.1"),
        ("Qwen/Qwen2.5-72B-Instruct", "qwen-2.5"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "llama-3"),
        ("claude-3-5-sonnet-20241022", "claude-sonnet"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o"),
    ],
)
def test_known_families(raw: str, expected: str) -> None:
    assert canonical_family(raw) == expected


def test_unknown_model_passes_through_lowercased() -> None:
    # An internal fine-tune that doesn't match any family pattern should
    # come through lowercased (the registry can still accept it; it just
    # won't Jaccard-match any trained profile).
    assert canonical_family("acme-corp/Internal-FineTune-v3") \
        == "acme-corp/internal-finetune-v3"


def test_empty_string() -> None:
    assert canonical_family("") == ""


@pytest.mark.parametrize(
    "name, expected",
    [
        ("nemotron-3-nano:30b-cloud", 30),
        ("devstral-small-2:24b-cloud", 24),
        ("qwen3:8b", 8),
        ("qwen3:0.6b", 0.6),            # decimal sizes preserved
        ("llama3.1:70b", 70),
        ("mixtral:8x7b", 56),          # MoE = experts × per-expert size
        ("deepseek-r1:14b", 14),
        ("my-ft-70b-q4_K_M", 70),
        ("glm-5.1:cloud", None),       # judge name carries no size
        ("gpt-4o", None),              # closed model
        ("claude-sonnet-4", None),
        ("", None),
    ],
)
def test_parse_params_b(name, expected) -> None:
    assert parse_params_b(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("my-ft-70b-q4_K_M", "q4_k_m"),
        ("llama-3.1-8b-instruct-fp16", "fp16"),
        ("model-awq", "awq"),
        ("nemotron-3-nano:30b-cloud", None),  # no quant marker -> unknown
        ("qwen3:8b", None),
    ],
)
def test_parse_quant(name, expected) -> None:
    assert parse_quant(name) == expected


def test_channel_specs_distinguishes_size_same_family() -> None:
    """Two channels of the SAME family but DIFFERENT sizes must remain
    distinguishable in the fingerprint — the whole point of the fix."""
    from agentcodec.config import LibraryConfig

    cfg = LibraryConfig.model_validate({
        "models": [
            {"model": "nemotron:30b", "base_url": "x", "api_key": "x"},
            {"model": "nemotron:70b", "base_url": "x", "api_key": "x"},
        ],
        "judge": {"model": "glm-5.1:cloud", "base_url": "x", "api_key": "x"},
        "strategy": {"type": "fixed", "technique": "baseline"},
    })
    fp = _derive_user_config(cfg)

    # Family list collapses (both 'nemotron') — back-compat, unchanged.
    assert fp["model_families"] == ["nemotron"]
    # ...but channel_specs keeps per-channel size, so 30b != 70b.
    assert fp["channel_specs"] == [
        {"family": "nemotron", "params_b": 30, "quant": None},
        {"family": "nemotron", "params_b": 70, "quant": None},
    ]


def test_channel_specs_excludes_judge() -> None:
    """The judge is not a channel and must not appear in the fingerprint."""
    from agentcodec.config import LibraryConfig

    cfg = LibraryConfig.model_validate({
        "models": [
            {"model": "nemotron-3-nano:30b-cloud", "base_url": "x", "api_key": "x"},
            {"model": "devstral-small-2:24b-cloud", "base_url": "x", "api_key": "x"},
        ],
        "judge": {"model": "glm-5.1:cloud", "base_url": "x", "api_key": "x"},
        "strategy": {"type": "fixed", "technique": "baseline"},
    })
    fp = _derive_user_config(cfg)
    assert fp["model_families"] == ["nemotron", "devstral"]
    assert [c["params_b"] for c in fp["channel_specs"]] == [30, 24]
    assert "glm-5.1" not in fp["model_families"]
