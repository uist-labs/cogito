#!/usr/bin/env python3
"""COGITO first-run launcher (`cogito-run`).

The third leg of the guided front door (setup.sh -> cogito-install ->
cogito-model -> cogito-run): take a downloaded GGUF and launch a ponder loop,
then hand the finished run to the visualizer. Guides the genesis prompt and the
run parameters behind a START/tune gate; every field Enter-accepts a default.

Design: docs/superpowers/specs/2026-07-10-first-run-launcher-design.md
Stdlib-only. Side effects (the loop, the visualizer, stdin, the listing, the
clock) are injected so the logic is testable offline.
"""

import sys
from pathlib import Path

import cogito_models as models


# --- Model resolution ------------------------------------------------------

def discover_ggufs(dest, lister=None):
    """Return the ``*.gguf`` files under ``dest``, most-recently-modified first.

    ``lister`` (injected in tests) maps a directory to a list of paths; the
    default globs the real directory. A missing directory yields an empty list.
    """
    if lister is not None:
        return list(lister(dest))
    d = Path(dest)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)


def _pick_model(files, input_fn, out):
    """Prompt for one of several discovered models. Enter accepts ``[1]``."""
    print("Several models found:", file=out)
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.name}", file=out)
    while True:
        raw = input_fn(f"Which model? [1-{len(files)}, Enter=1] ").strip()
        if not raw:
            return files[0]
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        print(f"  Please press Enter or enter a number 1-{len(files)}.", file=out)


def resolve_model(model_flag, dest, *, lister=None, input_fn=input, out=sys.stdout):
    """Resolve which model to run: ``--model`` wins, else discover under ``dest``.

    Returns a ``Path``, or ``None`` (with an actionable message printed) when the
    flagged file is missing or nothing is downloaded yet.
    """
    if model_flag:
        p = Path(model_flag)
        if p.exists():
            return p
        print(f"Model not found: {p}", file=out)
        return None

    files = discover_ggufs(dest, lister)
    if not files:
        print(f"No .gguf model found in {dest}. "
              "Run 'uv run cogito-model' first to download one.", file=out)
        return None
    if len(files) == 1:
        return files[0]
    return _pick_model(files, input_fn, out)


# --- GPU offload -----------------------------------------------------------

def resolve_offload(model_path, catalog, detection):
    """Return ``(gpu_layers, note)`` for ``model_path``.

    If the file matches a catalog entry (by GGUF filename), reuse S2's ``fit()``
    so the offload matches what cogito-model showed. An unknown file (hand
    downloaded or renamed) falls back to cogito's full-offload default with an
    honest note -- we never guess a specific wrong layer count.
    """
    name = Path(model_path).name
    for m in catalog:
        if m.hf_file == name:
            f = models.fit(m, detection)
            return f.gpu_layers, f.reason
    return -1, (f"{name} is not in the catalog; attempting full GPU offload -- "
                "lower --gpu-layers if you hit OOM")
