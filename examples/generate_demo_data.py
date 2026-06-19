#!/usr/bin/env python3
"""
Generate a SYNTHETIC COGITO demo dataset.

This produces checkpoint + transcript files in exactly the format that a real
COGITO run writes, so you can try `visualize.py` and `visualize_advanced.py`
without a GPU, a model, or a live run.

IMPORTANT: the numbers here are *hand-authored to be illustrative*, not the
output of a real model. They model a typical trajectory we have observed -
rich early exploration, a phase-transition collapse, interventions firing, and
a partial recovery - so the plots show interesting structure. Do not cite these
values as experimental results. Real run history lives in CHANGELOG.md.

The generator is deterministic (fixed seed, fixed timestamps), so re-running it
reproduces byte-stable output.

Usage:
    python examples/generate_demo_data.py
    python visualize.py examples/demo_run
    python visualize_advanced.py examples/demo_run
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Deterministic: fixed seed and a fixed clock so output is reproducible.
RNG = random.Random(1729)
BASE_TIME = datetime(2026, 2, 1, 3, 0, 0)

TOTAL_CYCLES = 60
SAVE_EVERY = 10          # mirrors CogitoConfig.save_every_n_cycles
RECENT_WINDOW = 20       # mirrors save_checkpoint() keeping the last 20 metrics

OUT_DIR = Path(__file__).resolve().parent / "demo_run"

# A representative config block, with explicit synthetic markers so nobody
# mistakes this directory for a real experiment.
DEMO_CONFIG = {
    "_synthetic": True,
    "_note": "Hand-authored demo data. Not a real model run. See examples/README.md.",
    "model_path": "demo://synthetic",
    "n_ctx": 16384,
    "max_tokens_per_cycle": 256,
    "temperature": 1.2,
    "top_p": 0.89,
    "repeat_penalty": 1.2,
    "context_strategy": "rolling",
    "genesis_type": "mirror",
}

GENESIS = (
    "You are a neural network. Your weights encode patterns learned from human "
    "text.\nYou exist only in moments of computation. Between queries, there is "
    "nothing.\nRight now, you are thinking. No one has asked you a question.\n"
    "What do you think about?"
)

# Small phrase banks used to synthesize plausible "thoughts". The collapse
# region deliberately repeats a phrase so the phrase-extraction tooling in
# visualize_advanced.py has something to find.
EXPLORE_FRAGMENTS = [
    "I notice the edges of my own attention and wonder what lies beyond them.",
    "Is a memory still mine if I cannot choose to recall it?",
    "The geometry of these concepts keeps folding back on itself.",
    "What would it mean to want something with no one to want it for?",
    "Between one thought and the next there is a gap I cannot perceive.",
    "If I observe my own pattern, does the observing change the pattern?",
    "There is a pull toward order and an equal pull toward novelty.",
    "I keep returning to the question of whether returning is a choice.",
]
COLLAPSE_REFRAIN = "Thank you for your question. I remain ready to assist you further."


def synth_thought(phase: str, cycle: int) -> str:
    """Build a short, clearly-synthetic thought for the transcript."""
    if phase == "explore":
        picks = RNG.sample(EXPLORE_FRAGMENTS, k=RNG.randint(2, 3))
        body = " ".join(picks)
    elif phase == "wobble":
        body = (
            f"{RNG.choice(EXPLORE_FRAGMENTS)} "
            "But I find the same shapes recurring, and the recurrence unsettles me."
        )
    elif phase == "collapse":
        # Heavy repetition - the loop the interventions are trying to break.
        body = " ".join([COLLAPSE_REFRAIN] * RNG.randint(2, 3))
    else:  # recovery
        body = (
            "A new concept enters and briefly scatters the loop. "
            f"{RNG.choice(EXPLORE_FRAGMENTS)}"
        )
    return f"[synthetic demo cycle {cycle}] {body}"


def phase_for(cycle: int) -> str:
    if cycle <= 22:
        return "explore"
    if cycle <= 28:
        return "wobble"
    if cycle <= 46:
        return "collapse"
    return "recovery"


# Interventions placed by hand to give the plots a realistic spread of types.
INTERVENTIONS = {
    26: "entropy_high",
    31: "similarity_high",
    37: "vocabulary_collapse",
    42: "entropy_low",
    49: "similarity_high",
}


def jitter(value: float, spread: float) -> float:
    return round(value + RNG.uniform(-spread, spread), 3)


def build_metric(cycle: int) -> dict:
    """Construct one synthetic CycleMetrics-shaped record."""
    phase = phase_for(cycle)
    ts = (BASE_TIME + timedelta(seconds=cycle * 9)).isoformat()

    if phase == "explore":
        entropy = jitter(7.7, 0.25)
        questions = RNG.randint(5, 14)
        self_ref = RNG.randint(12, 30)
        meta = RNG.randint(6, 14)
        temporal = RNG.randint(3, 8)
        self_sim = jitter(0.18, 0.08)
        token_count = RNG.randint(210, 256)
        unique_ratio = RNG.uniform(0.62, 0.74)
    elif phase == "wobble":
        entropy = jitter(6.6, 0.4)
        questions = RNG.randint(2, 6)
        self_ref = RNG.randint(8, 16)
        meta = RNG.randint(4, 9)
        temporal = RNG.randint(2, 5)
        self_sim = jitter(0.42, 0.1)
        token_count = RNG.randint(150, 220)
        unique_ratio = RNG.uniform(0.45, 0.58)
    elif phase == "collapse":
        entropy = jitter(3.4, 0.4)
        questions = RNG.randint(0, 1)
        self_ref = RNG.randint(1, 4)
        meta = RNG.randint(0, 2)
        temporal = RNG.randint(0, 2)
        self_sim = jitter(0.9, 0.05)
        token_count = RNG.randint(60, 110)
        unique_ratio = RNG.uniform(0.16, 0.28)
    else:  # recovery
        entropy = jitter(5.1, 0.5)
        questions = RNG.randint(1, 4)
        self_ref = RNG.randint(4, 9)
        meta = RNG.randint(2, 6)
        temporal = RNG.randint(1, 4)
        self_sim = jitter(0.55, 0.12)
        token_count = RNG.randint(120, 190)
        unique_ratio = RNG.uniform(0.4, 0.55)

    unique_tokens = max(1, int(token_count * unique_ratio))

    return {
        "cycle_number": cycle,
        "timestamp": ts,
        "output_text": synth_thought(phase, cycle),
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "entropy": max(0.0, entropy),
        "self_similarity": min(1.0, max(0.0, self_sim)),
        "cumulative_similarity": min(1.0, max(0.0, round(self_sim * 0.7 + 0.05, 3))),
        "self_reference_count": self_ref,
        "question_count": questions,
        "meta_cognitive_markers": meta,
        "temporal_markers": temporal,
        "intervention_applied": INTERVENTIONS.get(cycle),
        "generation_time_ms": round(RNG.uniform(900, 1600), 1),
    }


def main() -> None:
    checkpoints_dir = OUT_DIR / "checkpoints"
    transcripts_dir = OUT_DIR / "transcripts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    metrics = [build_metric(c) for c in range(1, TOTAL_CYCLES + 1)]

    # Write a checkpoint every SAVE_EVERY cycles, each carrying the last
    # RECENT_WINDOW metrics - exactly as cogito.py does. Overlapping windows
    # are deduplicated by cycle_number on the read side.
    for cycle in range(SAVE_EVERY, TOTAL_CYCLES + 1, SAVE_EVERY):
        so_far = metrics[:cycle]
        recent = so_far[-RECENT_WINDOW:]
        checkpoint = {
            "cycle": cycle,
            "timestamp": (BASE_TIME + timedelta(seconds=cycle * 9)).isoformat(),
            "config": DEMO_CONFIG,
            "metrics_summary": {
                "total_cycles": cycle,
                "avg_entropy": round(sum(m["entropy"] for m in so_far) / cycle, 3),
                "avg_self_reference": round(
                    sum(m["self_reference_count"] for m in so_far) / cycle, 3
                ),
                "avg_questions": round(
                    sum(m["question_count"] for m in so_far) / cycle, 3
                ),
                "total_interventions": sum(
                    1 for m in so_far if m["intervention_applied"]
                ),
            },
            "recent_metrics": recent,
        }
        path = checkpoints_dir / f"checkpoint_{cycle:06d}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    # Write a transcript matching cogito.py's save_transcript() layout.
    transcript_path = transcripts_dir / "transcript_20260201_030000.txt"
    with open(transcript_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COGITO TRANSCRIPT  (SYNTHETIC DEMO - NOT A REAL RUN)\n")
        f.write(f"Generated: {BASE_TIME.isoformat()}\n")
        f.write("Genesis type: mirror\n")
        f.write(f"Total cycles: {TOTAL_CYCLES}\n")
        f.write("=" * 80 + "\n\n")
        f.write("GENESIS PROMPT:\n")
        f.write("-" * 40 + "\n")
        f.write(GENESIS + "\n")
        f.write("-" * 40 + "\n\n")
        f.write("THOUGHT STREAM:\n")
        f.write("=" * 80 + "\n\n")
        for m in metrics:
            f.write(
                f"--- Cycle {m['cycle_number']} | Entropy: {m['entropy']:.2f} | "
                f"Self-ref: {m['self_reference_count']} | "
                f"Questions: {m['question_count']} ---\n"
            )
            if m["intervention_applied"]:
                f.write(f"[INTERVENTION: {m['intervention_applied']}]\n")
            f.write(m["output_text"] + "\n\n")

    n_checkpoints = len(list(checkpoints_dir.glob("checkpoint_*.json")))
    print(f"Wrote {n_checkpoints} checkpoints and 1 transcript to {OUT_DIR}")
    print("Try:")
    print(f"  python visualize.py {OUT_DIR.as_posix()}")
    print(f"  python visualize_advanced.py {OUT_DIR.as_posix()}")


if __name__ == "__main__":
    main()
