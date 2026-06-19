#!/usr/bin/env bash
# COGITO quick-start: set up a Python environment and (optionally) fetch a model.
set -e

echo "=============================================="
echo "   COGITO - Autonomous Pondering Experiment   "
echo "=============================================="
echo ""

# --- Python ----------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 is required but not found."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# --- GPU detection ---------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_CUDA=1
else
    echo "No NVIDIA GPU detected (will use CPU)"
    HAS_CUDA=0
fi

# --- Virtual environment ---------------------------------------------------
echo ""
echo "----------------------------------------------"
echo "Step 1: Python environment"
echo "----------------------------------------------"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using already-active virtualenv: $VIRTUAL_ENV"
    PIP="pip"
    ACTIVATE_HINT=""
else
    if [ ! -d ".venv" ]; then
        echo "Creating virtualenv in ./.venv ..."
        python3 -m venv .venv
    fi
    echo "Using ./.venv"
    PIP=".venv/bin/pip"
    ACTIVATE_HINT="source .venv/bin/activate"
fi
"$PIP" install --upgrade pip >/dev/null

# --- Dependencies ----------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo "Step 2: Installing dependencies"
echo "----------------------------------------------"
if [ "$HAS_CUDA" -eq 1 ]; then
    echo "Building llama-cpp-python with CUDA support (this can take several minutes)..."
    echo "  Tip: to skip the compile, use a prebuilt CUDA wheel instead - see"
    echo "       the 'Running on RunPod' section of README.md."
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 "$PIP" install llama-cpp-python --force-reinstall --no-cache-dir
else
    echo "Installing llama-cpp-python (CPU build)..."
    "$PIP" install llama-cpp-python
fi
echo "Installing remaining dependencies (numpy, matplotlib)..."
"$PIP" install -r requirements.txt

# --- Model setup -----------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo "Step 3: Model setup"
echo "----------------------------------------------"
MODEL_PATH=""
MODELS=$(find . -name "*.gguf" 2>/dev/null || true)
if [ -n "$MODELS" ]; then
    echo "Found existing GGUF model(s):"
    echo "$MODELS"
    echo ""
    read -r -p "Use one of these? (enter a path, or press Enter to download a new one): " MODEL_PATH
fi

if [ -z "$MODEL_PATH" ]; then
    echo ""
    echo "Recommended starter models:"
    echo "  1) TinyLlama 1.1B Q4 (~700MB)   - fastest, good for a first smoke test"
    echo "  2) Phi-2 2.7B Q4 (~1.6GB)       - balanced"
    echo "  3) Mistral 7B Q4 (~4.4GB)       - higher quality, needs more VRAM"
    echo ""
    read -r -p "Download which? (1/2/3, or Enter to skip): " CHOICE
    case "$CHOICE" in
        1)
            echo "Downloading TinyLlama..."
            wget -q --show-progress https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
            MODEL_PATH="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            ;;
        2)
            echo "Downloading Phi-2..."
            wget -q --show-progress https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
            MODEL_PATH="./phi-2.Q4_K_M.gguf"
            ;;
        3)
            echo "Downloading Mistral 7B..."
            wget -q --show-progress https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf
            MODEL_PATH="./mistral-7b-v0.1.Q4_K_M.gguf"
            ;;
        *)
            echo "Skipping download. Provide a model path with --model when you run."
            ;;
    esac
fi

# --- Done ------------------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo "Setup complete!"
echo "----------------------------------------------"
echo ""
if [ -n "$ACTIVATE_HINT" ]; then
    echo "Activate the environment first:"
    echo "  $ACTIVATE_HINT"
    echo ""
fi
echo "See the visualizer right now - no model or GPU needed:"
echo "  python visualize.py examples/demo_run"
echo ""
if [ -n "$MODEL_PATH" ]; then
    echo "Run your first experiment:"
    echo "  python cogito.py --model $MODEL_PATH --genesis-type mirror --cycles 50"
    echo ""
    echo "Then analyze it:"
    echo "  python visualize.py logs"
else
    echo "Once you have a GGUF model, run:"
    echo "  python cogito.py --model /path/to/model.gguf --genesis-type mirror --cycles 50"
fi
echo ""
echo "Read README.md for full documentation."
echo ""
