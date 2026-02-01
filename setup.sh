#!/bin/bash
# COGITO Quick Start Script
# This helps you get up and running with your first experiment

set -e

echo "=============================================="
echo "   COGITO - Autonomous Pondering Experiment   "
echo "=============================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_CUDA=1
else
    echo "No NVIDIA GPU detected (will use CPU)"
    HAS_CUDA=0
fi

echo ""
echo "----------------------------------------------"
echo "Step 1: Installing Dependencies"
echo "----------------------------------------------"

if [ "$HAS_CUDA" -eq 1 ]; then
    echo "Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --break-system-packages 2>/dev/null || \
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
else
    echo "Installing llama-cpp-python (CPU only)..."
    pip install llama-cpp-python --break-system-packages 2>/dev/null || pip install llama-cpp-python
fi

echo "Installing other dependencies..."
pip install numpy matplotlib --break-system-packages 2>/dev/null || pip install numpy matplotlib

echo ""
echo "----------------------------------------------"
echo "Step 2: Model Setup"
echo "----------------------------------------------"

# Check for existing models
MODELS=$(find . -name "*.gguf" 2>/dev/null)
if [ -n "$MODELS" ]; then
    echo "Found existing GGUF model(s):"
    echo "$MODELS"
    echo ""
    read -p "Use one of these? (Enter path or press Enter to download new): " MODEL_PATH
fi

if [ -z "$MODEL_PATH" ]; then
    echo ""
    echo "Recommended models for 8GB VRAM:"
    echo "  1) TinyLlama 1.1B (fast, ~700MB)  - Good for testing"
    echo "  2) Phi-2 2.7B Q4 (balanced, ~1.5GB)"
    echo "  3) Mistral 7B Q4 (quality, ~4GB)  - Recommended"
    echo ""
    read -p "Download which? (1/2/3 or Enter to skip): " CHOICE
    
    case $CHOICE in
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
            echo "Skipping download. You'll need to provide a model path manually."
            MODEL_PATH=""
            ;;
    esac
fi

echo ""
echo "----------------------------------------------"
echo "Setup Complete!"
echo "----------------------------------------------"
echo ""

if [ -n "$MODEL_PATH" ]; then
    echo "Ready to run your first experiment!"
    echo ""
    echo "Quick test (10 cycles):"
    echo "  python cogito.py --model $MODEL_PATH --genesis-type mirror --cycles 10"
    echo ""
    echo "Longer run (100 cycles):"
    echo "  python cogito.py --model $MODEL_PATH --genesis-type wonder --cycles 100"
    echo ""
    echo "Overnight run:"
    echo "  python cogito.py --model $MODEL_PATH --genesis-type mirror --cycles 0"
    echo ""
    echo "Then analyze with:"
    echo "  python visualize.py ./logs"
else
    echo "Download a GGUF model and run:"
    echo "  python cogito.py --model /path/to/model.gguf --genesis-type mirror --cycles 100"
fi

echo ""
echo "Read README.md for full documentation."
echo ""
