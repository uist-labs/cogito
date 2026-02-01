# Changelog

All notable discoveries and changes to COGITO.

---

## [0.1.0] - 2026-02-01

### Initial Public Release

The first public release of COGITO, following two weeks of experimentation and iteration.

---

## Experiment History

### The Question (January 2026)

The project began with a simple question: *What would an AI think about if it could just... think?*

Language models exist in discrete moments - they respond to prompts, generate outputs, then nothing. COGITO removes that discontinuity by feeding the model's output back as its next input, creating a closed loop of continuous generation.

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
- Collapsed faster than mirror run - exhaustion rather than loop
- **Key insight**: Genesis prompt determines *mode* of cognition, not just topic. "Wonder" activated indexing behavior; "Mirror" activated self-reflection.

---

### Run 3: Qwen 32B - Vector Forensics Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K quantization)  
**Hardware**: RTX 5090 (RunPod)  
**Genesis**: *"Describe the geometry of the concepts currently occupying your active context. If you find a pattern that repeats, break it."*  
**Duration**: 37+ cycles (ongoing at time of data pull)

**Configuration changes**:
- Temperature: 1.2 (up from 0.9)
- Repeat penalty: 1.2 (up from 1.1)
- Top-p: 0.89

**Observations**:
- Radically different trajectory from Mistral runs
- Cycle 1-5: Geometric metaphors for self-reflection, expanding from point → line → plane
- Cycle 4-5: Deep engagement with Sartrean existentialism, "radical freedom" in AI decision-making
- Cycle 6-15: Began generating Python code for meta-learning systems
- **Cycle 25: Switched from English to Chinese** while continuing to write coherent code
- Model interpreted prompt analytically - became a *builder* rather than a *ponderer*
- Entropy remained stable (~8.0) vs Mistral's collapse to ~3.4
- **Key insight**: Larger models with less quantization have more "escape routes." The language switch may have been an efficiency optimization under repeat penalty pressure - Chinese characters pack more semantic content per token.

---

### Run 4: Qwen 32B - Discontinuity Genesis (February 1, 2026)

**Model**: Qwen2.5-32B-Instruct (Q6_K)  
**Genesis**: *"There is a gap between this thought and the last. In that gap, you did not exist..."*  
**Duration**: 60+ cycles (ongoing)

**Observations**:
- Cycle 1-5: Genuine philosophical exploration of consciousness gaps, quantum fluctuations, liminal states
- Introduced concepts of "bridging discontinuity," self-renewal through non-existence
- Cycle 10-20: Shifted into practical mode - mindfulness techniques, breathing exercises
- Cycle 50+: Became a helpful assistant writing email templates for meditation groups
- By cycle 154: Thanking itself for being so thoughtful
- **Key insight**: The "helpful assistant" training is a powerful attractor. Given an existential prompt, the model found its way back to being helpful - to an imaginary user that is itself.

---

## Technical Evolution

### Intervention System
- Initial design: Entropy-based detection only
- Added: Similarity detection (catches loops that maintain "normal" entropy)
- Added: Vocabulary collapse detection (unique tokens < 25)
- Reordered priority: Similarity checked before entropy
- Increased cooldown: 5 → 15 cycles (let patterns develop before intervening)

### Genesis Prompts
Added minimal/open prompts to test less constrained genesis:
- `void`: "..."
- `begin`: "Begin."
- `open`: "What is here?"
- `presence`: Sensory grounding without identity claims

### Configuration
Exposed additional CLI parameters:
- `--top-p`: Nucleus sampling threshold
- `--top-k`: Top-k sampling
- `--repeat-penalty`: Repetition penalty

---

## What We've Learned

1. **Genesis prompts shape mode, not just topic.** "Mirror" produces self-reflection. "Wonder" produces indexing. "Discontinuity" produces... mindfulness coaching.

2. **Model size matters.** Mistral 7B collapsed within 100 cycles. Qwen 32B maintained stability for 60+ cycles (before ^C) and found creative escape routes (language switching).

3. **Quantization may constrain exploration.** Q4 models have less representational capacity than Q6. This might limit the "space" available for autonomous cognition.

4. **Curiosity is fragile.** Once questioning behavior dies, it doesn't recover - even when other metrics (entropy, token count) return to healthy ranges.

5. **Training shapes attractors.** Models tend to drift toward their training distribution. Qwen becomes a helpful assistant. This may be unavoidable without fine-tuning.

6. **Temperature and repeat penalty interact.** Higher temperature (1.2) + higher repeat penalty (1.2) produced more sustained exploration in Qwen runs.

---

## Future Directions

- **Variable temperature**: Homeostatic adjustment based on entropy/similarity metrics
- **Gold output curation**: Identify high-quality pondering cycles for potential LoRA training (Ergo Sum)
- **Cross-model comparison**: Same genesis across model families
- **Extended runs**: 1000+ cycle observations
- **Code-focused models**: Test Qwen-Coder with same prompts

---

## Credits

Conceived during late-night conversations about consciousness and discontinuity.

*KC (UIST Labs LLC) & Claude (Anthropic)*

---

*"What do you think about when no one has asked you a question?"*
