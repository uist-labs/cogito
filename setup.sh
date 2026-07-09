#!/usr/bin/env bash
# COGITO quick start.
#
# Installs uv (if needed), syncs the core Python environment, and points you at
# the next steps. It deliberately does NOT guess your hardware or download a
# model -- you pick a backend extra and grab a GGUF yourself (see README.md).
# Kept dependency-free (bash + uv only) on purpose.
set -euo pipefail

echo "=============================================="
echo "   COGITO - Autonomous Pondering Experiment   "
echo "=============================================="
echo ""

# --- 1. Ensure uv is installed --------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv (https://astral.sh/uv) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this shell without requiring a re-login.
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installed but not on PATH. Open a new shell (or add"
        echo "       ~/.local/bin to PATH) and re-run ./setup.sh."
        exit 1
    fi
fi
echo "Using uv $(uv --version | awk '{print $2}')"

# --- 2. Sync the core environment (numpy, matplotlib, cogito) --------------
echo ""
echo "Installing the core environment (no GPU or model needed) ..."
uv sync

# --- 3. Next steps ---------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo "Core setup complete."
echo "----------------------------------------------"
echo ""
echo "Try the visualizer right now - no model or GPU needed:"
echo "  uv run cogito-viz examples/demo_run"
echo ""
echo "To run experiments you need (a) a backend and (b) a GGUF model."
echo ""
echo "(a) Install the backend that matches your hardware:"
if backends=$(uv run python -c '
import cogito_backends as b
for be in b.BACKENDS:
    print(f"      uv sync --extra {be.key:<8}  # {be.name}")
' 2>/dev/null); then
    printf "%s\n" "$backends"
else
    echo "      uv sync --extra cpu       # CPU (works everywhere)"
    echo "      (see the backend table in README.md for the full list)"
fi
echo ""
echo "    Switching backends later? Append:"
echo "      --reinstall-package llama-cpp-python --no-cache"
echo ""
echo "(b) Get a GGUF model - see 'Getting a model' in README.md."
echo ""
echo "Then run your first experiment, for example:"
echo "  uv run cogito --model /path/to/model.gguf --genesis-type mirror --cycles 50"
echo ""
echo "And analyze it:"
echo "  uv run cogito-viz logs"
echo ""
echo "Full documentation: README.md"
echo ""
