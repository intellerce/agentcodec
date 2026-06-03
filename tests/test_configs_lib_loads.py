"""Every YAML under ``configs/lib/`` must validate against the strict
``LibraryConfig`` schema.

The CI workflow runs the same check, but having it as a pytest module
means broken configs fail locally too — and contributors editing a
config without running CI catch the regression at `pytest tests/`.

If you add a new YAML, you don't need to register it here — the
parametrize collects them from disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcodec.config import LibraryConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "lib"
YAML_FILES = sorted(CONFIGS_DIR.glob("*.yaml")) if CONFIGS_DIR.exists() else []


@pytest.mark.skipif(
    not YAML_FILES,
    reason="No configs/lib/*.yaml files; nothing to validate.",
)
@pytest.mark.parametrize(
    "path", YAML_FILES, ids=[p.name for p in YAML_FILES],
)
def test_config_yaml_loads(path: Path) -> None:
    cfg = LibraryConfig.from_yaml(path)
    assert cfg.models, f"{path.name}: must declare at least one model"
    assert cfg.judge, f"{path.name}: must declare a judge"
