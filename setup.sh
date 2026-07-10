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

# --- 3. Guided backend install, then model picker -------------------------
echo ""
echo "----------------------------------------------"
echo "Core setup complete. Launching the guided installer..."
echo "----------------------------------------------"
echo ""
# cogito-install detects your hardware and installs the matching llama-cpp-python
# backend. The '|| echo' keeps 'set -e' from aborting the chain if the backend
# verify only warns -- you can still download a model and fix the runtime after.
# Re-run either wizard any time: 'uv run cogito-install' / 'uv run cogito-model'.
uv run cogito-install || echo "(backend install reported an issue -- re-run 'uv run cogito-install' any time.)"

echo ""
echo "----------------------------------------------"
echo "Now let's get a model..."
echo "----------------------------------------------"
echo ""
# cogito-model recommends a GGUF that fits your hardware and downloads it. The
# '|| echo' keeps 'set -e' from aborting the chain if the model step is skipped
# (e.g. a non-interactive shell) -- you can grab one later and re-run cogito-run.
uv run cogito-model || echo "(model step skipped -- grab one later with 'uv run cogito-model'.)"

echo ""
echo "----------------------------------------------"
echo "Ready to ponder..."
echo "----------------------------------------------"
echo ""
# cogito-run finds the model, guides the run parameters (press Enter to start
# with the recommended defaults, or 'tune' to adjust them; Ctrl-C to defer),
# launches the ponder loop, and opens the visualizer when it finishes.
# Re-run any time: 'uv run cogito-run'.
exec uv run cogito-run
