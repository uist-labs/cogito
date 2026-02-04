# Changelog

All notable discoveries and changes to COGITO.

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
