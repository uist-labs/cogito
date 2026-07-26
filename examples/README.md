# Examples - zero-GPU demo data

This directory lets you try the COGITO visualizers **without a GPU, a model, or
a live run** - useful for kicking the tires before committing to a real
experiment.

## ⚠️ This data is synthetic

`demo_run/` contains hand-authored, illustrative data - **not** the output of a
real model. The numbers were written to model a *typical* trajectory we've
observed (rich early exploration → a phase-transition collapse → interventions
firing → a partial recovery) so the plots show interesting structure.

**Do not cite these values as experimental results.** Real run history and
findings live in [`../CHANGELOG.md`](../CHANGELOG.md), and the figures in
[`../analysis/`](../analysis) are from real runs.

## Try it

From the repo root (with `numpy` and `matplotlib` installed - see the main
[README](../README.md)):

```bash
# Basic dashboard (entropy, similarity, questions, interventions, ...)
python visualize.py examples/demo_run

# Deeper analysis (phase-transition detection, intervention effectiveness)
python visualize_advanced.py examples/demo_run
```

Both write PNGs into `examples/demo_run/` and run fine headless (no display
needed), so they work the same on a RunPod pod as on your laptop. The generated
PNGs are gitignored - regenerate them anytime.

## Regenerate the data

The dataset is produced deterministically (fixed seed, fixed timestamps), so
re-running the generator reproduces byte-stable output:

```bash
python examples/generate_demo_data.py
```

See [`generate_demo_data.py`](generate_demo_data.py) for exactly how the
synthetic trajectory is constructed.
