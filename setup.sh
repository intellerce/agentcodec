#!/usr/bin/env bash
# Set up a local development environment for the public AgentCodec package.
#
#   ./setup.sh           # create .venv, install package + dev deps
#   ./setup.sh --remote  # also install the remote-semknn extra (BGE encoder)
#
# Re-runnable; uses uv (https://docs.astral.sh/uv/) for fast installs.

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

EXTRAS="dev"
if [[ "${1:-}" == "--remote" ]]; then
    EXTRAS="dev,remote-semknn"
fi

# Create the venv (Python >= 3.10 per pyproject.toml).
if [[ ! -d .venv ]]; then
    uv venv --python 3.11 .venv
fi

# Editable install with the requested extras. Resolves from pyproject.toml.
uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run tests with:       ./.venv/bin/pytest tests/"
