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

# --- 3. Hand off to the guided backend installer --------------------------
echo ""
echo "----------------------------------------------"
echo "Core setup complete. Launching the backend installer..."
echo "----------------------------------------------"
echo ""
# cogito-install detects your hardware, recommends and installs the matching
# llama-cpp-python backend, then points you at getting a model and a first run.
# Re-run it any time with 'uv run cogito-install' (e.g. to switch backends).
# exec hands the terminal to the wizard so its interactive prompts get the tty.
exec uv run cogito-install
