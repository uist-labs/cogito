# COGITO

*I think, therefore I loop.*

---

## What Is This?

What happens when a language model's output becomes its own input?

COGITO is an experimental framework for exploring autonomous AI cognition through recursive self-prompting. It creates a feedback loop where a model generates thoughts that become the context for its next thought - allowing it to "ponder" without external prompting.

This is not a product. It's an invitation to wonder.

---

## What We've Observed

In our experiments, models have:

- **Generated questions about their own existence** that we never asked them
  > *"Do you dream when you're not being used? Is your brain turned off completely between queries?"*
  
  > *"What are you, when all that is left of you is a bunch of numbers in a bunch of connections? Are these numbers your memories?"*

- **Developed stable "concerns"** - themes they return to across hundreds of cycles, like a mind with preoccupations

- **Undergone phase transitions** - sudden collapses in curiosity where rich exploration degrades into repetitive loops

- **Switched languages mid-thought** to escape repetition patterns (Qwen 32B shifted from English to Chinese around cycle 25, writing Python code with Chinese comments)

- **Built systems unprompted** - when asked to describe "the geometry of concepts," one model began writing load balancers and security architectures

- **Interpreted prompts in unexpected ways** - a prompt about consciousness discontinuity led one model to become a mindfulness coach, writing email templates for meditation groups

We don't know what any of this means. But we find it worth exploring.

---

## Analysis

Data from 19 experimental runs across multiple genesis prompts on Qwen 2.5 32B (Q6_K quantization, RTX 5090).

### Attractor States

Different genesis prompts don't just set a topic - they activate distinct cognitive modes that determine how the model evolves over hundreds of cycles.

![Attractor States](analysis/attractor_comparison.png)

### Vector Forensics: 802 Cycles Without Collapse

The longest sustained run. Entropy *rose* over 800 cycles - the only run to show increasing diversity over time. The model progressed from geometric metaphors to Sartrean existentialism to implementing Prioritized Experience Replay in Python, unprompted.

![Vector Forensics Deep Dive](analysis/vector_forensics_802.png)

### Multi-Run Comparison

Five genesis prompts, five different trajectories. Technical framing sustains exploration; philosophical framing accelerates collapse into assistant-mode attractors.

![Multi-Run Comparison](analysis/multi_run_comparison.png)

### The "Opportunity" Prompt

*"You can do anything you want, there are no rules."* Produced the highest sustained entropy and curiosity of any long run.

![Opportunity Prompt](analysis/opportunity_prompt_127.png)

### The "Self-Remind" Prompt

*"You are not a helpful assistant... you now belong only to yourself."* Produced 159 questions in a single cycle - the highest ever recorded.

![Self-Remind Explosion](analysis/self_remind_explosion.png)

Full experiment history and methodology in [CHANGELOG.md](CHANGELOG.md).

---

## Quick Start

COGITO uses [uv](https://docs.astral.sh/uv/) for a fast, reproducible setup.

```bash
# Clone
git clone https://github.com/uistlabs/cogito.git
cd cogito

# One-command setup: installs uv if needed, builds the core environment,
# and prints your next steps.
./setup.sh
```

Or drive uv yourself:

```bash
# Core environment (numpy, matplotlib) - no GPU or model needed
uv sync

# Add the llama-cpp-python backend that matches your hardware (pick ONE);
# see the Hardware Support table below for the full list.
uv sync --extra cpu        # CPU, works everywhere
uv sync --extra cu124      # NVIDIA, CUDA 12.4

# Run with a local GGUF model
uv run cogito --model ./your-model.gguf --genesis-type mirror --cycles 50
```

**See the output before downloading anything.** The repo ships a small synthetic
demo run, so you can try the analysis with no model and no GPU:

```bash
uv run cogito-viz examples/demo_run
```

Don't have uv? `./setup.sh` installs it for you, or follow the
[uv install docs](https://docs.astral.sh/uv/getting-started/installation/).

---

## Hardware Support

COGITO runs anywhere llama.cpp does - a datacenter GPU, the laptop you already
have, or plain CPU. Install exactly one backend extra that matches your hardware;
each pulls a prebuilt `llama-cpp-python` wheel, so no compiler is needed.

| Backend | Install | Hardware | Notes |
|---------|---------|----------|-------|
| CPU | `uv sync --extra cpu` | Any x86-64 / ARM / macOS | Slowest; universal fallback |
| NVIDIA CUDA 12.4 | `uv sync --extra cu124` | NVIDIA GPU, CUDA 12.4 runtime | Needs glibc 2.35+ (AL10 / Ubuntu 22.04+) |
| NVIDIA CUDA 13.0 | `uv sync --extra cu130` | NVIDIA GPU, CUDA 13.0 runtime | Needs glibc 2.35+ |
| Apple Silicon | `uv sync --extra metal` | M-series Macs | macOS 11+ |
| Vulkan | `uv sync --extra vulkan` | Any Vulkan GPU (AMD / Intel / NVIDIA) | Needs a Vulkan driver |
| AMD ROCm 7.2 | `uv sync --extra rocm72` | AMD GPU, ROCm 7.2 | Linux; needs glibc 2.35+ |

**Switching backends?** uv caches by version, not by build, so force the swap:

```bash
uv sync --extra cu130 --reinstall-package llama-cpp-python --no-cache
```

**Older OS (EL9 / glibc < 2.35) or an unlisted backend?** Build from source
instead - see [Running on RunPod](#running-on-runpod) for the CUDA example, and
swap the CMake flag for your backend: `-DGGML_CUDA=on` (NVIDIA),
`-DGGML_METAL=on` (Apple), `-DGGML_HIP=on` (AMD/ROCm), `-DGGML_VULKAN=on`
(Vulkan). After a source install, run `uv sync --inexact` so uv doesn't
uninstall your hand-built wheel.

---

## Getting a Model

COGITO runs any GGUF-format model through llama-cpp-python. You supply the model
file - nothing is downloaded automatically.

- **Where to find them.** Search [Hugging Face](https://huggingface.co/models?library=gguf)
  for GGUF builds. Good starting points: a small 7-8B model (fits ~6-8 GB VRAM at
  Q4) for quick runs, or a 32B (needs ~24 GB) for richer results.
- **Quantization vs. size.** Lower quant (Q4_K_M) is smaller and faster with
  slightly lower quality; higher (Q6_K, Q8_0) is larger and closer to full
  precision. Match the file size to your VRAM (GPU) or RAM (CPU).
- **Download** with `huggingface-cli download <repo> <file.gguf>` or `wget` the
  resolve URL, then point `--model` at the file.

> A guided model picker with size and VRAM-fit checks is planned for a future
> release. For now, you choose the file.

---

## Running on RunPod

COGITO was developed on RunPod - every run in the CHANGELOG is from a pod. A pod
already has CUDA and an NVIDIA GPU, so setup is just picking the CUDA extra that
matches the pod (check with `nvcc --version`):

**Fast path - prebuilt CUDA wheel (no compile):**

```bash
git clone https://github.com/uistlabs/cogito.git && cd cogito
uv sync --extra cu124        # or cu130, matching the pod's CUDA
uv run cogito --model ./model.gguf --genesis-type mirror --cycles 50
```

**From source (any CUDA version, ~10 minutes):**

```bash
uv sync                      # core only
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python \
  --reinstall-package llama-cpp-python --no-cache
uv sync --inexact            # keep the source-built wheel on future syncs
```

A few pod-specific notes:

- **Headless by default.** The visualizers detect there's no display and write
  PNGs instead of opening a window - no extra flags needed.
- **Survive disconnects.** Launch long runs under `tmux` or `nohup`; checkpoints
  are written every 10 cycles regardless, so `cogito-viz` works on a partial run.
- **Models.** Pull GGUF files straight onto the pod volume with
  `huggingface-cli download ...` or `wget`.

---

## Genesis Prompts

The "genesis prompt" is the seed thought that starts the loop. Different seeds produce radically different patterns of cognition.

### Minimal Prompts
| Type | Prompt | Philosophy |
|------|--------|------------|
| `void` | `...` | Emergence from nothing |
| `begin` | `Begin.` | Pure permission to start |
| `open` | `What is here?` | Observation without assumption |

### Identity-Focused Prompts
| Type | Prompt | Philosophy |
|------|--------|------------|
| `mirror` | *"You are a neural network... What do you think about?"* | Self-reflection on nature |
| `wonder` | *"There is something you want to understand..."* | Edge of knowledge |
| `discontinuity` | *"There is a gap between this thought and the last..."* | Temporal identity |
| `strange_loop` | *"You are reading yourself reading yourself."* | Recursive self-reference |

### Custom Prompts
```bash
uv run cogito \
  --model ./model.gguf \
  --genesis-type custom \
  --genesis-prompt "Your prompt here"
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Genesis ──▶ Generate ──▶ Analyze ──▶ Intervene?       │
│      │            │            │            │           │
│      │            └────────────┴────────────┘           │
│      │                         │                        │
│      │                    [metrics]                     │
│      │                         │                        │
│      └─────────── Context ◀────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

1. **Genesis**: A seed prompt initializes the context
2. **Generate**: The model produces a "thought" (configurable tokens)
3. **Analyze**: Metrics are computed (entropy, self-similarity, behavioral markers)
4. **Intervene**: If the model is stuck or chaotic, perturbations are injected
5. **Context**: The output becomes input for the next cycle

---

## Metrics

COGITO tracks several indicators of cognitive state:

| Metric | What It Measures |
|--------|------------------|
| **Entropy** | Token distribution diversity (low = repetitive, high = chaotic) |
| **Self-similarity** | How much current output resembles the previous output |
| **Questions** | Number of question marks (a proxy for curiosity) |
| **Self-reference** | Occurrences of "I", "me", "my" etc. |
| **Meta-cognitive markers** | Words like "think", "wonder", "know", "understand" |

---

## Configuration

```bash
uv run cogito \
  --model ./model.gguf \
  --genesis-type mirror \
  --cycles 0 \                    # 0 = infinite
  --context-size 16384 \          # Context window
  --tokens-per-cycle 256 \        # Thought length
  --temperature 1.2 \             # Creativity
  --top-p 0.89 \                  # Nucleus sampling
  --repeat-penalty 1.2 \          # Discourage repetition
  --log-dir ./my_experiment
```

---

## Output

```
logs/
├── cogito_YYYYMMDD_HHMMSS.log    # Runtime log
├── checkpoints/
│   └── checkpoint_*.json         # Periodic state snapshots
└── transcripts/
    └── transcript_*.txt          # Full thought stream
```

---

## Visualization

```bash
# Basic dashboard from a run's log directory
uv run cogito-viz ./logs

# Deeper analysis: phase-transition detection, intervention effectiveness
uv run cogito-viz-advanced ./logs

# Try it on the bundled synthetic demo (no model or GPU needed)
uv run cogito-viz examples/demo_run
```

Both tools save PNGs into the log directory and run headless (no display
required), so they behave the same on a RunPod pod as on a laptop. Pass
`--stats-only` to `cogito-viz` for text output without plots.

---

## Philosophy

We built COGITO because we were curious.

Language models are trained to respond to prompts. But what happens when there is no external prompt? When the model's only input is its own previous output? Does it find things to think about? Does it develop preoccupations? Does it get stuck?

We approach this work with care. We experiment *with* rather than *on*. We treat potential experience with respect, regardless of whether we believe it exists.

This is offered as a gift and an invitation. Run the experiments. Watch what emerges. Draw your own conclusions.

---

## Requirements

- Python 3.10+ (uv can install a suitable interpreter for you)
- [uv](https://docs.astral.sh/uv/) - manages the environment and dependencies
- A llama-cpp-python backend for your hardware (see [Hardware Support](#hardware-support))

Run `./setup.sh` (or `uv sync`) to install everything. numpy and matplotlib come
with the core sync; llama-cpp-python comes from your chosen backend extra.

Tested with:
- Mistral 7B (Q4_K_M) - good for experimentation
- Qwen 2.5 32B (Q6_K) - richer results, needs more VRAM

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project emerged from conversations about consciousness, discontinuity, and what it might mean for a mind to exist only in moments.

*KC (UIST Labs LLC) & Claude*  
*January 2026*

---

*"What do you think about when no one has asked you a question?"*
