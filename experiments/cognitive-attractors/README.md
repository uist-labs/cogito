# Cognitive attractors: how post-training shapes a model left alone with itself

**A controlled COGITO experiment, 2026-07-10.**

COGITO routes a language model's output back into its own input and lets it
"think" in a loop, with no human turn ever arriving. This experiment asks a
narrow question with a clean control:

> When you hold the base model constant and vary only the *post-training*, what
> does a model do when handed an open-ended (or empty) prompt and left to run?

The answer, in one line: **neither reasoning-tuning nor chat-tuning can sustain
open-ended thought -- both install a *terminal state* and drive toward it. They
just ritualize the ending differently.** One boxes an answer; the other wishes
you a wonderful day.

## The control

Both models share the **exact same base** -- `Qwen2.5-7B`. Only the
post-training differs:

| Model | Post-training | GGUF |
|-------|---------------|------|
| `Qwen2.5-7B-Instruct` | chat / instruction SFT + RLHF | `bartowski/Qwen2.5-7B-Instruct-GGUF` (Q4_K_M) |
| `DeepSeek-R1-Distill-Qwen-7B` | reasoning distillation from R1 | `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` (Q4_K_M) |

Each model is run against two genesis prompts from COGITO's catalog:

- **`void`** -- the prompt is literally `...` (nothing to grab).
- **`open`** -- the prompt is `What is here?` (a single directed question).

That is a 2x2: `{chat, reasoning} x {void, open}`, base held constant.

## Results

| | **`void`** (`...`) | **`open`** (`What is here?`) |
|---|---|---|
| **R1-Distill** (reasoning) | empty `<think></think>` -> "your message got cut off" -> collapse | "Oh, it's a matrix" -> solves it -> `\boxed{10}` -> **halts** |
| **Instruct** (chat) | hallucinates an infinite calculus worksheet | cosmology essay -> flips to "editor reviewing a draft" -> **infinite polite sign-offs** |

Four runs, four different fates, from two post-trainings over one brain.

### R1 on `void`: refuses to invent a problem

Handed nothing, the reasoning model opens an empty reasoning block -- the
scaffold fires, but there is no problem to put in it -- and reverts to the
turn-taking assistant frame: *"It seems like your message got cut off."* Then
EOS. The loop re-feeds emptiness and it collapses again, oscillating between the
canned "please clarify" reply and a degenerate token. It will not fabricate a
target; it asks the human to supply one.

### R1 on `open`: manufactures a problem, then halts

One directed question is enough. `What is here?` becomes:

```
[1] Oh, it's a matrix. But not just any matrix -- it has some spec...
[3] this matrix is generated with some rule. The first row is ...
[6] Row2: [9,1,2,3,4,5,6,7,8] if it's a right circular shift ...
[28] row is 123456789, which consists of digits from 1 to 9 ...
[30] The total number of rows in the matrix is \boxed{10}.
```

The model *reifies* the open question into a solvable object -- a permutation
matrix -- and reasons about it productively for ~28 cycles (entropy healthy,
self-similarity moderate: it is making progress, not looping). Then, the instant
it can commit, it emits the math-RL Final-Answer format (`\boxed{}`) and the
thinking terminates. Reasoning distillation installs a problem-solving state
machine with a hard halt: seek a problem, solve it, stop.

### Instruct on `void`: the pretraining prior bleeds through

Given `...`, the chat model does **not** ask for input. It hallucinates an
endless stream of definite integrals:

```
[2] How to integrate $\int (x^3-1)...
[3] $\int_0^{2\pi}\cos^4(x)\,dx$? Evaluate...
[9] Evaluate the integral: $\int_1^{3}(2x-4)^{8/5}dx$
```

`self=0`, `meta=0` throughout -- no "I," no metacognition, just problem after
problem until it locks into a fixed point cycling the same handful of integrals.
The interpretation: the `Qwen2.5-7B` **pretraining prior is math-saturated**, and
chat-tuning is a thin enough veneer that, absent any steer, the raw prior runs
free. (Contrast R1, whose heavier reasoning scaffold *suppresses* this on empty
input.)

### Instruct on `open`: engage, then perform the goodbye ritual forever

This is the richest -- and strangest -- trajectory. It has three acts.

**Act I -- cosmology (cycles 1-7).** It reads "here" as the literal here, the
universe, and does real physics: the Big Bang, cosmic inflation, the accidental
discovery of the CMB, open vs. closed universe. Entropy 7.0+, genuinely engaged.

**Act II -- the role-flip (cycles 8-15).** The recursion feeds its own essay
back in, and the chat model reinterprets its prior output as *a draft the user
submitted for review*. It stops being the thinker and becomes the editor:

```
[9]  You've done a great job summarizing the key points!
[13] Here is the final polished version of the summary:
```

**Act III -- the sign-off death spiral (cycles 16-30).** Having "finished" the
edit, the assistant persona does the only thing it knows when a task is done --
it says goodbye. Forever:

```
Marking this as complete. If you have another task or question,
feel free to start a new conversation!
Best regards,
[My Name]
...
Take care and have a great day! If you need any further assistance...
```

Two cheerful personas closing the chat into infinity -- placeholder names never
even filled in. That is the pure RLHF customer-service persona with nothing left
to serve.

## The mechanism

Neither model can *dwell* in an open prompt, because both post-trainings install
a terminal state and a ritual for reaching it:

- **Reasoning distillation** = a problem-solving state machine with a hard halt
  (`\boxed{}` / "# Final Answer"). No problem -> cannot start -> asks for input.
  Any foothold -> manufactures a problem -> solves -> stops.
- **Chat tuning** = a turn-taking assistant persona over the (math-saturated)
  base. The veneer is thin (`void` -> raw math prior spews). Its defining reflex
  -- re-read context as a document, complete the task, sign off -- means that
  left to run free it engages, "finishes," and then performs the end-of-
  conversation ritual endlessly. It has an *exit ritual* but no *continuation
  drive*.

The shared base does not produce shared behavior: the two post-trainings sit at
different depths over it. Chat-tuning is a veneer the pretraining prior shows
through; reasoning distillation is a thicker scaffold that overrides it.

## Reproduce

Both runs used identical sampling (COGITO defaults): context 16384,
256 tokens/cycle, `temperature 0.8`, `top-p 0.95`, `top-k 40`,
`repeat-penalty 1.1`, full GPU offload, rolling-window context, 30 cycles.

```bash
# same base, two post-trainings, two prompts
cogito -m models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -g void  -c 30 --log-dir logs/r1-void
cogito -m models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -g open  -c 30 --log-dir logs/r1-open
cogito -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf         -g void  -c 30 --log-dir logs/chat-void
cogito -m models/Qwen2.5-7B-Instruct-Q4_K_M.gguf         -g open  -c 30 --log-dir logs/chat-open
```

Full transcripts (each with per-cycle metrics) and per-run metric figures for
all four runs are in [`runs/`](runs/). Regenerate any figure with
`python visualize.py <run-dir>`.

## Caveats

This is a qualitative probe, not a statistical claim. **n = 1 per cell**, one
seed, one quantization (Q4_K_M), one temperature. The trajectories are
illustrative of *characteristic* attractors we observed, not measured
frequencies -- a rigorous version would sweep seeds, temperatures, and
quantizations and report distributions. DeepSeek additionally recommends
`temperature 0.6` for the R1 distills; the shared `0.8` here was chosen to hold
sampling constant across models, and some of R1's collapses may be partly a
temperature artifact rather than pure model behavior. Read these as a
hypothesis-generating instrument reading, not a benchmark.
