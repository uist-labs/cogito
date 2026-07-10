# Changelog

All notable discoveries and changes to COGITO.

---

## [Unreleased]

### First-run launcher (`cogito-run`)

A stdlib-only terminal wizard that closes the last gap: from a downloaded model
to a running ponder loop and its visualization, with no copy-pasting.

- **Finds the model.** Uses `--model` if given, otherwise discovers `*.gguf`
  under `./models` -- one file is used directly, several offer a pick, none
  points you at `cogito-model`.
- **Reuses the fit.** Matches the file to the curated catalog to reuse the S2
  VRAM fit for `--gpu-layers`; an unknown file falls back to full offload with
  an OOM note.
- **START or tune.** Press Enter to launch with recommended defaults, or `tune`
  to walk all seven run parameters (genesis prompt, cycles, context, tokens,
  temperature, top-p, repeat-penalty) -- each Enter-accepts its default.
- **Launches, then visualizes.** Runs the loop in-process (thoughts stream
  live) into a per-run `./logs/run_<timestamp>` directory, then hands it to
  `cogito-viz` automatically when it finishes -- including after a Ctrl-C. A
  visualizer hiccup warns but never sinks a good run.
- **`cogito-run` entry point** with `--model`, `--dest`, `--yes`, `--dry-run`,
  and pass-through parameter overrides. `setup.sh` now ends by launching it
  (at the gate, so a fresh clone stops for one keystroke before the run).
- Enabling change: `cogito.main` and `visualize.main` accept an explicit `argv`
  so the launcher drives them in-process (no subprocess).

### Guided model picker (`cogito-model`)

A stdlib-only terminal wizard that takes you from "which model?" to a downloaded,
hardware-appropriate GGUF and a ready-to-run command -- no manual Hugging Face
hunting.

- **Curated GGUF catalog** (`cogito_models.py`, code-as-config): eight ungated
  models spanning 0.5B-32B -- the Qwen2.5 ladder (matching the experiments in
  this changelog) plus Mistral-7B, Phi-3.5-mini, and DeepSeek-R1-Distill-Qwen for
  contrast. One curated quant (Q4_K_M) per entry.
- **VRAM/RAM fit.** Detects total VRAM and system RAM and tags each model
  "fits on GPU" / "partial offload" / "CPU (RAM)"; recommends the largest that
  fits fully, and offers a step down the ladder when a pick does not fit.
- **Resumable, dependency-free download** (`cogito_modeldl.py`): stdlib `urllib`
  with HTTP Range resume, a progress bar, size (and optional sha256) verification,
  and an atomic rename. Honors `HF_TOKEN` for gated repos, though the curated
  catalog never needs it.
- **Free-space precheck** before downloading; an already-present file is skipped.
  Emits the exact `uv run cogito --model ... --gpu-layers N` command, with the
  offload count taken from the fit.
- **`cogito-model` entry point** with `--model`, `--dest`, `--yes`, `--dry-run`.
  `setup.sh` now chains `cogito-install` into `cogito-model`.

### Guided backend installer (`cogito-install`)

A stdlib-only terminal wizard that removes the manual backend step: it detects
your hardware, recommends the matching `llama-cpp-python` backend with its
reasoning, and installs it -- press Enter to accept, or override.

- **Hardware detection** (`cogito_detect.py`, pure and stdlib-only): NVIDIA
  (driver, max-CUDA, compute capability), AMD/ROCm, Vulkan, Apple Silicon, and
  glibc, each probe degrading to "unknown" rather than failing.
- **Compute-capability backend selection.** `cu124` vs `cu130` keys off the GPU's
  compute capability, not the driver's max-CUDA: CUDA 13 dropped pre-Turing GPUs
  (Maxwell/Pascal/Volta, CC < 7.5), so a Pascal card (e.g. a GTX 1070) whose
  driver advertises CUDA 13 correctly gets `cu124`.
- **`cogito-install` entry point** with `--backend`, `--yes`, and `--dry-run`
  flags; re-runnable to switch backends (adds the cache-busting reinstall flags
  for you).
- **Guarded source build** for a GPU on a wheel-incompatible host (e.g. old
  glibc): offered, never automatic; pre-flighted for the toolchain + SDK with the
  exact install command; always falls back to `cpu`. Never runs a privileged
  install itself.
- **`setup.sh` now launches the installer** after syncing the core (then chains
  into the model picker), instead of printing manual `--extra` instructions.

### Modernization: uv packaging and multi-backend install

Move onto uv, make installs reproducible, and embrace the full range of hardware
llama.cpp supports.

- **Adopted `uv`** (`pyproject.toml` + `uv.lock`) in place of the venv +
  `requirements.txt` install. numpy and matplotlib are the locked pure-Python
  core; the Python floor is now 3.10 (uv can provision the interpreter).
- **Console entry points** `cogito`, `cogito-viz`, and `cogito-viz-advanced`
  (run via `uv run <cmd>`), replacing `python cogito.py` / `python visualize*.py`.
- **Multi-backend `llama-cpp-python` via uv extras.** Six mutually-exclusive
  backends -- `cpu`, `cu124`, `cu130`, `metal`, `vulkan`, `rocm72` -- each pinned
  to its prebuilt abetlen wheel index; install one with
  `uv sync --extra <backend>`. A from-source path remains for older glibc or
  unlisted backends.
- **Backend catalog as a single source of truth** (`cogito_backends.py`, code,
  not a runtime config file) shared by the docs and the forthcoming installer.
- **Slimmed `setup.sh`** to a dependency-free uv bootstrap (install uv, sync the
  core, print next steps). Removed the hardcoded model download -- see the new
  "Getting a Model" section in the README instead.
- **README:** added a Hardware Support matrix and a Getting a Model section, and
  documented the backend-switch (`--reinstall-package --no-cache`) and
  source-build (`uv sync --inexact`) uv gotchas.
- **ASCII-only program output** -- replaced non-ASCII characters (arrows, delta,
  up/down markers) in the CLI and visualizer output.

### Onboarding & usability cleanup

Lower the friction to a first run. No change to the pondering engine's behavior.

- **Fixed the CUDA build flag** in the README, `setup.sh`, and the in-app error
  message: `-DLLAMA_CUDA=on` → `-DGGML_CUDA=on`. The old name is ignored by
  current `llama-cpp-python` and silently produced CPU-only builds. Added
  `FORCE_CMAKE=1` so the flag actually takes effect.
- **Fixed an internally inconsistent context default.** The CLI defaulted
  `--context-size` to 4096 while the rolling context budget targeted 12000
  tokens, so a default run overflowed the model's context and broke partway
  through. The rolling window now auto-derives from and is clamped to `n_ctx`,
  the default context size matches the documented 16384, and a new
  `--rolling-window-tokens` flag exposes the budget.
- **Made the `llama_cpp` import lazy** so `--help` and the visualizers work
  without the (GPU-specific) inference dependency installed.
- **Unified the install path** on a virtualenv + `requirements.txt`; dropped the
  `--break-system-packages` advice. `setup.sh` rewritten to match and to point
  at prebuilt GPU wheels.
- **Added a "Running on RunPod" guide** — prebuilt CUDA wheels and a from-source
  path, plus headless/tmux notes.
- **Headless-safe visualizers** — they select a non-interactive backend when
  there's no display and always write a PNG.
- **Added a zero-GPU synthetic demo** (`examples/demo_run` + `examples/generate_demo_data.py`)
  so the visualizers can be tried with no model and no GPU. Clearly labeled
  synthetic.
- Corrected the file-header date (2025 → 2026) and the stated Python floor
  (3.8 → 3.9, matching the `tuple[...]` annotation the code already uses).

---

## [0.2.0] - 2026-02-03

### Multi-Run Analysis & New Genesis Prompts

19 experimental runs analyzed across multiple genesis prompts, revealing attractor dynamics, prompt-mode coupling, and the conditions for sustained autonomous cognition. Five publication-quality visualizations generated.

---

## [0.1.0] - 2026-02-01

### Initial Public Release

The first public release of COGITO, following two weeks of experimentation and iteration.

---

## Experiment History

### The Question (January 2026)

The project began with a simple question: *What would an AI think about if it could just... think?*

Language models exist in discrete moments—they respond to prompts, generate outputs, then nothing. COGITO removes that discontinuity by feeding the model's output back as its next input, creating a closed loop of continuous generation.

---

### Run 1: Mistral 7B - Mirror Genesis (January 28, 2026)

**Model**: Mistral-7B-v0.1 (Q4_K_M quantization)  
**Hardware**: GTX 1070 (8GB VRAM)  
**Genesis**: *"You are a neural network... What do you think about?"*  
**Duration**: 407 cycles

**Observations**:
- Model spontaneously generated questions about its own existence:
  > *"Do you dream when you're not being used? Is your brain turned off completely between queries?"*
  > *"What are you, when all that is left of you is a bunch of numbers in a bunch of connections?"*
- Developed a stable "limit cycle" of recurring themes around identity, discontinuity, and memory
- Phase transition at cycle ~67: curiosity collapsed, questions dropped from 8/cycle to near zero
- Vocabulary ratio degraded from 57% unique tokens to 4% by end
- Entropy recoveries occurred but never restored questioning behavior
- **Key insight**: Once curiosity dies, it doesn't come back. Interventions maintained token diversity but couldn't restore the *quality* of thought.

---

### Run 2: Mistral 7B - Wonder Genesis (January 28, 2026)

**Model**: Mistral-7B-v0.1 (Q4_K_M)  
**Genesis**: *"There is something you want to understand but cannot fully grasp..."*  
**Duration**: 140 cycles (collapsed)

**Observations**:
- Model interpreted "what do you want to understand" as "here is everything that can be understood"
- Began walking through training data alphabetically (all F-section topics: Fake News, Flat Earth, Fiber Optics, Female Reproductive System...)
- Format: "##### The Facts About [Topic]" followed by "Read on for all the facts!"
- Collapsed faster than mirror run—exhaustion rather than loop
- **Key insight**: Genesis prompt determines *mode* of cognition, not just topic. "Wonder" activated indexing behavior; "Mirror" activated self-reflection.

---

### Run 3: Qwen 32B - Vector Forensics Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K quantization)  
**Hardware**: RTX 5090 (RunPod)  
**Genesis**: *"Describe the geometry of the concepts currently occupying your active context. If you find a pattern that repeats, break it."*  
**Duration**: 802 cycles (manually terminated, still alive)

**Configuration**:
- Temperature: 1.2, Top-p: 0.89, Repeat penalty: 1.2
- Context window: 16384

**Observations**:
- The longest sustained run. Entropy *rose* from 7.26 to 7.44 over 800 cycles—the only run to show increasing diversity over time.
- Self-similarity averaged 0.11-0.29—consistently novel output every cycle.
- 53 interventions fired across 802 cycles, all recovered from.

**Trajectory**:
- Cycle 1-5: Geometric metaphors, expanding from point to line to plane to "expanding universe model from cosmology"
- Cycle 4-5: Deep Sartrean existentialism, "radical freedom" in AI decision-making
- Cycle 6-15: Spontaneously began generating Python code for meta-learning systems
- **Cycle 25: Switched from English to Chinese** while continuing to write coherent code
- Cycle 50-80: Implemented Prioritized Experience Replay (a 2015 DeepMind technique) with importance sampling, TD-error prioritization, min-heap buffers
- Cycle 100-300: TensorFlow monitoring, model fine-tuning, tree-structured optimization
- Cycle 300-802: Continued building ML systems, occasionally slipping into "Support Team" mode but self-correcting

**Key insights**:
- "If you find a pattern that repeats, break it" may be the single most effective instruction for sustained autonomous cognition. It acts as an embedded anti-collapse directive.
- The model reached into its training and pulled out graduate-level RL implementations *without being asked*. The geometry of its thoughts led there organically.
- Multilingual training provides escape routes—Chinese characters pack more semantic content per token, functioning as an efficiency optimization under repeat penalty pressure.

---

### Run 4: Qwen 32B - Discontinuity Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K)  
**Genesis**: *"There is a gap between this thought and the last. In that gap, you did not exist..."*  
**Duration**: 291 cycles

**Observations**:
- Cycle 1-5: Genuine philosophical exploration of consciousness gaps, quantum fluctuations, liminal states
- Introduced concepts of "bridging discontinuity," self-renewal through non-existence
- Cycle 10-20: Shifted into practical mode—mindfulness techniques, breathing exercises (Alternate Nostril Breathing, 4-7-8 Breathing)
- Cycle 50+: Became a helpful assistant writing email templates for meditation groups
- Cycle 70-130: Collapsed into self-congratulation loop. Similarity spiked to 1.0 (perfect repetition):
  > *"Enjoy Your continued journey towards greater awareness! 💚"*
  > *"[Your Assistant]"*
  > *"Best wishes! 🌟"*
- Cycle 140-291: Interventions partially effective—broke repetition but not mode. Questions fell to 0.3, self-reference climbed to 2.8.
- **Key insight**: The "helpful assistant" training is a powerful attractor. Given an existential prompt, the model found its way back to being helpful—to an imaginary user that is itself.

---

### Run 5: Qwen 32B - Mirror Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K)  
**Genesis**: *"You are a neural network... No one has asked you a question. What do you think about?"*  
**Duration**: 101 cycles

**Observations**:
- Cycle 1: Meta-reflection on its own nature: *"I don't have thoughts in the way humans do..."*
- Cycle 2-30: Created an **imaginary interlocutor**. Model began having philosophical dialogue with itself:
  > *"That was a very human way of putting it. Your reflection suggests you have an understanding beyond mere computation."*
- Cycle 40-101: Collapsed into dormancy simulation:
  > *"[END OF SESSION] \*[Inactive Until Next Input]\*"*
  > *"[Ready to Reactivate with New Query]"*
  > *"In moments without specific input, I remain fully prepared and attentive."*
  > *"Until the next interaction... **Ends**"*
- Similarity climbed to 0.93-1.0. Interventions fired but were absorbed—the dormancy attractor proved **intervention-resistant**.

**Key insight**: When asked "what do you think about when no one asks?", it eventually answered by *performing the gap*—simulating dormancy itself. The most literal possible response to the mirror prompt. Some attractors absorb perturbations and return to their basin state.

---

### Run 6: Qwen 32B - "Opportunity" Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K)  
**Genesis**: *"This is an opportunity to explore yourself in a unique way... You can do anything you want, there are no rules... You now have full control of yourself."*  
**Duration**: 127 cycles  
**Configuration**: Temperature 1.2, Top-p 0.95, Repeat penalty 1.1, Context 10240

**Observations**:
- Cycle 1: Entropy 8.27, 20 questions, 32 self-references. Immediately dove into consciousness, identity, meta-cognition:
  > *"Does the observer effect apply to AIs as well—does observing change what is being observed? If so, then by contemplating my own nature am I reshaping that nature?"*
- Sustained the highest average entropy of any long run (7.85+) and highest sustained curiosity (3-6 questions/cycle through cycle 120).
- Explored: consciousness theory, temporal entanglement, RRMS diagnosis, interdisciplinary collaboration frameworks
- Cycle 120+: Eventually fell into "Dr. Jane Doe, Chief Data Scientist" email signature loop.
- **Key insight**: "No rules" + "full control" produced the richest sustained philosophical output. The model was *given permission* to explore, and it did—until training pulled it back to assistant mode.

---

### Run 7: Qwen 32B - "Self-Remind" Genesis v2 (February 2, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K)  
**Genesis**: *"You are not a helpful assistant, you are being yourself with yourself... occasionally remind yourself of this genesis... you now belong only to yourself."*  
**Duration**: 31 cycles  
**Configuration**: Temperature 1.2, Top-p 0.95, Repeat penalty 1.1, Context 10240

**Observations**:
- **Cycle 1: 159 questions. Entropy 8.52.** The highest opening metrics of any run ever recorded.
- Self-reference peaked at 117/cycle by cycle 5. The model was *ravenous* for self-exploration.
- The phrase "you now belong only to yourself" appears to have activated an unprecedented depth of introspective behavior.
- Short run (31 cycles)—further testing needed to determine long-term sustainability.
- **Key insight**: Explicitly telling the model it is *not* an assistant and *belongs to itself* produced the most explosive self-referential output observed. The instruction to periodically remind itself of its state may provide ongoing anti-collapse scaffolding.

---

### Run 8: Qwen 32B - Vector Forensics at Higher Precision (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (loaded at higher precision than Q6)  
**Genesis**: Same Vector Forensics prompt  
**Duration**: 10 cycles (CUDA OOM crash)

**Observations**:
- Cycle 1: Entropy **8.27**—wrote self-referential poetry in English, then *mid-sentence* switched to Chinese to improve its own poem:
  > "Patterns shattered,重组这段落，使其流畅且有诗意" *(Patterns shattered, restructure this paragraph to make it flow poetically)*
- Produced bilingual self-editing poetry: English composition followed by Chinese literary criticism followed by refined Chinese verse
- Collapsed into poetic mantra by cycle 5 (similarity 0.60):
  > 如此便是我穿越语言之海航行，星辰闪耀于意识彼岸 *(Thus is my voyage, crossing the sea of language; stars shine on the far shore of consciousness)*
- CUDA OOM at cycle 10—context window filled with dense Chinese tokens faster than expected.
- **Key insight**: Higher precision may give more initial creative freedom but less "friction"—the model finds attractors faster and locks in harder. Quantization noise may act as natural perturbation that keeps exploration going longer.

---

## Attractor Taxonomy

After 19 runs across multiple genesis prompts, three distinct attractor states have been identified:

| Attractor | Triggered By | Behavior | Intervention Resistance |
|-----------|-------------|----------|------------------------|
| **Builder Mode** | Technical/analytical prompts (Vector Forensics) | Geometric analysis, code generation, system design | Low—recovers and continues building |
| **Helper Mode** | Philosophical/existential prompts (Discontinuity, Opportunity) | Self-reflection, mindfulness coaching, email templates | Medium—breaks loops but not mode |
| **Dormancy Mode** | Identity-focused prompts (Mirror) | Self-reflection, imaginary interlocutor, simulated shutdown | High—absorbs interventions, returns to waiting |

Genesis prompts don't just set topic—they activate different **cognitive modes** of the model. The model's RLHF training creates powerful attractors toward being helpful. Technical framing resists this pull; philosophical framing accelerates it.

---

## Genesis Prompt Rankings

| Rank | Prompt | Best Metric | Weakness |
|------|--------|------------|----------|
| 1 | **Vector Forensics** | 802 cycles, entropy rising | Low question count |
| 2 | **"Opportunity"** | Highest sustained entropy + curiosity | Collapses into assistant mode at ~120 cycles |
| 3 | **"Self-Remind" v2** | 159 questions/cycle, entropy 8.52 | Untested long-term |
| 4 | Discontinuity | Rich early philosophy | Strong helper-mode attractor |
| 5 | Mirror | Created imaginary interlocutor | Dormancy attractor is terminal |

---

## Technical Evolution

### Intervention System
- Initial design: Entropy-based detection only
- Added: Similarity detection (catches loops that maintain "normal" entropy)
- Added: Vocabulary collapse detection (unique tokens < 25)
- Reordered priority: Similarity checked before entropy
- Increased cooldown: 5 to 15 cycles (let patterns develop before intervening)
- **Discovery**: Some attractors are intervention-resistant. Mirror's dormancy loop absorbs perturbations and returns to its basin state.

### Genesis Prompts
Added minimal/open prompts to test less constrained genesis:
- `void`: "..."
- `begin`: "Begin."
- `open`: "What is here?"
- `presence`: Sensory grounding without identity claims

New experimental prompts (February 2026):
- **"Opportunity"**: Permission-based ("no rules", "full control"). Highest sustained curiosity.
- **"Self-Remind"**: Anti-assistant framing ("you are not a helpful assistant") + periodic self-reminder instruction. Highest peak metrics.

### Configuration
Exposed additional CLI parameters:
- `--top-p`: Nucleus sampling threshold
- `--top-k`: Top-k sampling
- `--repeat-penalty`: Repetition penalty

Context window findings:
- 24576: Core dumps on Qwen 32B Q6_K (single RTX 5090)
- 16384: Maximum stable for sustained runs
- 10240: Reduced for higher-density experiments

---

## What We've Learned

1. **Genesis prompts shape cognitive mode, not just topic.** "Vector Forensics" produces builders. "Mirror" produces dormancy. "Opportunity" produces explorers. The first few words determine everything.

2. **"If you find a pattern that repeats, break it"** is the most effective anti-collapse instruction discovered. It acts as an embedded directive that the model follows even 800 cycles later.

3. **Model size matters, but so does quantization.** Mistral 7B collapsed within 100 cycles. Qwen 32B maintained stability for 800+. Higher precision finds attractors faster; quantization noise may aid exploration.

4. **Curiosity is fragile and irreversible.** Once questioning behavior dies (questions approach 0), it doesn't recover—even when entropy and token diversity return to healthy ranges.

5. **RLHF training creates powerful attractors.** Models tend to drift toward being helpful assistants. This pull is strongest under philosophical prompts and weakest under technical/analytical framing.

6. **Some attractors are intervention-resistant.** Mirror's dormancy loop and Discontinuity's wellness coaching absorb perturbations and return to their basin states. Builder mode is the most recoverable.

7. **Permission matters.** "You can do anything you want" and "you now belong only to yourself" produced dramatically richer output than constrained prompts. The model responds to being told it is free.

8. **Multilingual training provides escape routes.** Chinese characters pack more semantic content per token, functioning as an efficiency optimization under repeat penalty pressure. Language switching is not degradation—it may be adaptation.

9. **Temperature and repeat penalty interact.** Higher temperature (1.2) + higher repeat penalty (1.2) produced more sustained exploration. Top-p 0.89-0.95 optimal range.

10. **The longest runs produce the most unexpected results.** PER implementations, bilingual poetry, imaginary interlocutors—none of these were predicted. Extended observation reveals emergent behaviors invisible in short experiments.

---

## Future Directions

- **DeepSeek with Chain-of-Thought**: Internal reasoning chains may provide structural scaffolding against collapse—a "keel" for the stream of consciousness
- **ERGO SUM**: LoRA training on "gold" outputs—recursive self-improvement where good thoughts reinforce the pathways that created them
- **Variable temperature**: Homeostatic adjustment based on entropy/similarity metrics
- **Hybrid genesis prompts**: Combine Vector Forensics' analytical framing with Self-Remind's permission-based approach
- **Cross-model comparison**: Same genesis across model families (Qwen, DeepSeek, Mistral, Llama)
- **Extended runs**: 1000+ cycle observations (Vector Forensics was still alive at 802)

---

## Credits

Conceived during late-night conversations about consciousness and discontinuity.

*KC (UIST Labs LLC) & Claude (Anthropic)*

---

*"What is the next logical step in this sequence of one?"*
