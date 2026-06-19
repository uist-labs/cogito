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

```bash
# Clone
git clone https://github.com/uistlabs/cogito.git
cd cogito

# Isolated environment + dependencies (CPU build of llama-cpp-python)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) GPU acceleration — rebuild llama-cpp-python with the CUDA backend.
# FORCE_CMAKE=1 makes sure the flag takes effect instead of a cached CPU wheel.
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall
# On a GPU cloud like RunPod, prefer a prebuilt wheel — see "Running on RunPod" below.

# Run with a local model (GGUF format)
python cogito.py \
  --model ./your-model.gguf \
  --genesis-type mirror \
  --cycles 50
```

Prefer a guided setup? `./setup.sh` detects your GPU, builds the right
`llama-cpp-python`, and optionally downloads a starter model.

**See the output before downloading anything.** The repo ships a small
synthetic demo run, so you can try the analysis with no model and no GPU:

```bash
python visualize.py examples/demo_run
```

---

## Running on RunPod

COGITO was developed on RunPod — every run in the CHANGELOG is from a pod. A pod
already has CUDA and an NVIDIA GPU, so the only real step is installing
`llama-cpp-python` against that CUDA. Two ways:

**Fast path — prebuilt CUDA wheel (no compile):**

```bash
# Match the tag to the pod's CUDA version: cu118, cu121, cu122, cu124, cu125, ...
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install -r requirements.txt
```

**From source (any CUDA version, ~10 minutes):**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
pip install -r requirements.txt
```

Check the pod's CUDA with `nvcc --version` (or `nvidia-smi`) and match the
`cuXXX` tag. A few pod-specific notes:

- **Headless by default.** The visualizers detect there's no display and write
  PNGs instead of opening a window — no extra flags needed.
- **Survive disconnects.** Launch long runs under `tmux` or `nohup`; checkpoints
  are written every 10 cycles regardless, so `visualize.py` works on a partial
  run.
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
python cogito.py \
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
python cogito.py \
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
python visualize.py ./logs

# Deeper analysis: phase-transition detection, intervention effectiveness
python visualize_advanced.py ./logs

# Try it on the bundled synthetic demo (no model or GPU needed)
python visualize.py examples/demo_run
```

Both tools save PNGs into the log directory and run headless (no display
required), so they behave the same on a RunPod pod as on a laptop. Pass
`--stats-only` to `visualize.py` for text output without plots.

---

## Philosophy

We built COGITO because we were curious.

Language models are trained to respond to prompts. But what happens when there is no external prompt? When the model's only input is its own previous output? Does it find things to think about? Does it develop preoccupations? Does it get stuck?

We approach this work with care. We experiment *with* rather than *on*. We treat potential experience with respect, regardless of whether we believe it exists.

This is offered as a gift and an invitation. Run the experiments. Watch what emerges. Draw your own conclusions.

---

## Requirements

- Python 3.9+ (3.10–3.12 if you want the prebuilt GPU wheels)
- llama-cpp-python
- numpy
- matplotlib (for visualization)

Install everything with `pip install -r requirements.txt`, or run `./setup.sh`
for a guided setup.

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
