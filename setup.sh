#!/usr/bin/env bash
# Set up a local development environment for the public AgentCodec package.
#
#   ./setup.sh           # create .venv, install package + dev + provider SDKs
#   ./setup.sh --heavy   # also install benchmark + remote-semknn (torch, etc.)
#
# The default install includes the openai/anthropic/ollama provider SDKs and
# the eval extra so the full test suite and examples run without "module not
# found" surprises. The heavy extras (torch, transformers, matplotlib, pandas,
# scikit-learn, datasets) are opt-in via --heavy since most contributors don't
# need them and they make a clean install much larger.
#
# Re-runnable; uses uv (https://docs.astral.sh/uv/) for fast installs.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Default: dev tooling + every provider SDK + eval stats. Lets all tests run
# without pulling in torch/matplotlib/datasets.
EXTRAS="dev,openai,anthropic,ollama,eval"
case "${1:-}" in
    --heavy)
        # `all` bundles the provider SDKs plus remote-semknn/benchmark/eval.
        EXTRAS="dev,all"
        ;;
    "")
        ;;
    *)
        echo "Unknown option: $1 (expected --heavy)" >&2
        exit 1
        ;;
esac

# Create the venv (Python >= 3.10 per pyproject.toml).
if [[ ! -d .venv ]]; then
    uv venv --python 3.11 .venv
fi

# Editable install with the requested extras. Resolves from pyproject.toml.
uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"

echo
echo "Installed extras: ${EXTRAS}"
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run tests with:       ./.venv/bin/pytest tests/"
