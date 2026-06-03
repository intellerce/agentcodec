"""Validation tests for RouterConfig (the new SemKNN-remote schema)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcodec.config import RouterConfig


def _minimal_semknn(**overrides) -> dict:
    base = {
        "type": "semknn",
        "server_url": "https://semknn.example.com",
        "lambda": 5.0,
    }
    base.update(overrides)
    return base


def test_semknn_minimal_valid() -> None:
    rc = RouterConfig.model_validate(_minimal_semknn())
    assert rc.type == "semknn"
    assert rc.server_url == "https://semknn.example.com"
    assert rc.lambda_ == 5.0
    assert rc.strict_match is None
    assert rc.fallback == "none"


def test_semknn_rejects_local_cache_field() -> None:
    with pytest.raises(ValidationError, match="not supported for type=semknn"):
        RouterConfig.model_validate(_minimal_semknn(cache="weights/semknn.json"))


def test_semknn_omitted_server_url_defaults_to_public_endpoint() -> None:
    """`server_url` is optional — when missing, the schema fills in the
    public hosted backend so the user doesn't have to know it. Override
    via YAML or AGENTCODEC_SEMKNN_SERVER_URL when running a self-host."""
    from agentcodec._endpoints import AGENTCODEC_SERVER_URL
    payload = _minimal_semknn()
    del payload["server_url"]
    rc = RouterConfig.model_validate(payload)
    assert rc.server_url == AGENTCODEC_SERVER_URL


def test_semknn_requires_lambda() -> None:
    payload = _minimal_semknn()
    del payload["lambda"]
    with pytest.raises(ValidationError, match="lambda is required"):
        RouterConfig.model_validate(payload)


def test_semknn_strict_match_optional() -> None:
    rc = RouterConfig.model_validate(_minimal_semknn(strict_match=True))
    assert rc.strict_match is True
    rc = RouterConfig.model_validate(_minimal_semknn(strict_match=False))
    assert rc.strict_match is False


def test_semknn_linear_fallback_needs_cache() -> None:
    with pytest.raises(ValidationError, match="fallback_cache"):
        RouterConfig.model_validate(_minimal_semknn(fallback="linear"))
    rc = RouterConfig.model_validate(
        _minimal_semknn(fallback="linear", fallback_cache="weights/linear.json"),
    )
    assert rc.fallback == "linear"


def test_acm_linear_requires_cache() -> None:
    with pytest.raises(ValidationError, match="cache is required"):
        RouterConfig.model_validate({"type": "acm_linear"})


def test_acm_table_requires_table_entries() -> None:
    with pytest.raises(ValidationError, match="table or router.category_tables"):
        RouterConfig.model_validate({"type": "acm_table"})


def test_acm_linear_rejects_semknn_fields() -> None:
    with pytest.raises(ValidationError, match="only valid for type=semknn"):
        RouterConfig.model_validate({
            "type": "acm_linear",
            "cache": "weights/linear.json",
            "server_url": "https://x",
        })


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(_minimal_semknn(unknown_field="hi"))
