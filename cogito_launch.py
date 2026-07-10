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

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cogito
import cogito_detect as detect
import cogito_models as models
import visualize


# --- Run parameters --------------------------------------------------------

# The launcher's first-run defaults. Six match cogito.py's own CLI defaults;
# cycles is 50 (a gentler first run than cogito's bare-CLI 100, and the value
# cogito-model already prints in its suggested command -- the front door stays
# consistent with what the picker showed).
DEFAULTS = {
    "genesis_type": "mirror",
    "genesis_prompt": "",
    "cycles": 50,
    "context_size": 16384,
    "tokens_per_cycle": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}


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


def resolve_model(model_flag, dest, *, lister=None, input_fn=input, out=sys.stdout,
                  interactive=True):
    """Resolve which model to run: ``--model`` wins, else discover under ``dest``.

    Returns a ``Path``, or ``None`` (with an actionable message printed) when the
    flagged file is missing or nothing is downloaded yet. With several models and
    ``interactive`` false (``--yes``/``--dry-run``/no tty), the most-recent is
    used without prompting rather than blocking on stdin.
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
    if not interactive:
        print(f"Several models found; using the most recent: {files[0].name} "
              "(pass --model to choose another).", file=out)
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


# --- Parameter wizard ------------------------------------------------------

def _prompt_nonempty(label, input_fn, out):
    while True:
        raw = input_fn(f"{label}: ").strip()
        if raw:
            return raw
        print("  Please enter a non-empty prompt.", file=out)


def _prompt_genesis(default, input_fn, out):
    """Pick a genesis seed. Returns ``(genesis_type, genesis_prompt)``.

    The genesis field reads like a free-text box but wants one of a fixed set of
    named seeds -- so show the menu up front (sourced from cogito.GENESIS_PROMPTS,
    no drift) and, on unknown input, say what was typed and how to write your own
    ('custom'), rather than reprinting a bare list.
    """
    seeds = list(cogito.GENESIS_PROMPTS.keys())
    print("Genesis prompt -- the seed thought that starts the loop "
          "(see README for what each does).", file=out)
    print("  Named seeds: " + ", ".join(seeds), file=out)
    print("  ...or type 'custom' to write your own.", file=out)
    while True:
        raw = input_fn(f"Genesis [{default}]: ").strip()
        if not raw:
            return default, ""
        if raw == "custom":
            return "custom", _prompt_nonempty("Your genesis prompt", input_fn, out)
        if raw in seeds:
            return raw, ""
        print(f"  '{raw}' isn't a seed name -- pick one from the list above, "
              "or type 'custom' to write your own.", file=out)


def _prompt_number(label, default, cast, input_fn, out, *, minimum=None, maximum=None):
    kind = "whole number" if cast is int else "number"
    while True:
        raw = input_fn(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError:
            print(f"  Please enter a {kind}.", file=out)
            continue
        if minimum is not None and value < minimum:
            print(f"  Must be >= {minimum}.", file=out)
            continue
        if maximum is not None and value > maximum:
            print(f"  Must be <= {maximum}.", file=out)
            continue
        return value


def prompt_params(defaults, input_fn=input, out=sys.stdout):
    """START/tune gate, then the seven-field wizard. Every field Enter-defaults.

    Returns a params dict shaped like DEFAULTS. The gate's default is START, so
    Enter (or 'start') launches with recommended settings; 'tune' walks the
    full parameter set.
    """
    gate = input_fn("Start now, or tune parameters? [START/tune] ").strip().lower()
    if gate in ("", "start", "s"):
        return dict(defaults)

    p = dict(defaults)
    gtype, gtext = _prompt_genesis(defaults["genesis_type"], input_fn, out)
    p["genesis_type"] = gtype
    if gtype == "custom":
        p["genesis_prompt"] = gtext
    p["cycles"] = _prompt_number(
        "Cycles (0 = infinite)", defaults["cycles"], int, input_fn, out, minimum=0)
    p["context_size"] = _prompt_number(
        "Context size", defaults["context_size"], int, input_fn, out, minimum=1)
    p["tokens_per_cycle"] = _prompt_number(
        "Tokens per cycle", defaults["tokens_per_cycle"], int, input_fn, out, minimum=1)
    p["temperature"] = _prompt_number(
        "Temperature", defaults["temperature"], float, input_fn, out, minimum=0.0)
    p["top_p"] = _prompt_number(
        "Top-p", defaults["top_p"], float, input_fn, out, minimum=0.0, maximum=1.0)
    p["repeat_penalty"] = _prompt_number(
        "Repeat penalty", defaults["repeat_penalty"], float, input_fn, out, minimum=0.0)
    return p


# --- Argv construction -----------------------------------------------------

def build_argv(model_path, params, gpu_layers, log_dir):
    """The exact cogito CLI argv. Pure -- the contract the launch seam runs and
    --dry-run prints."""
    argv = [
        "--model", str(model_path),
        "--genesis-type", params["genesis_type"],
    ]
    if params["genesis_type"] == "custom":
        argv += ["--genesis-prompt", params["genesis_prompt"]]
    argv += [
        "--cycles", str(params["cycles"]),
        "--context-size", str(params["context_size"]),
        "--tokens-per-cycle", str(params["tokens_per_cycle"]),
        "--temperature", str(params["temperature"]),
        "--top-p", str(params["top_p"]),
        "--repeat-penalty", str(params["repeat_penalty"]),
        "--gpu-layers", str(gpu_layers),
        "--log-dir", str(log_dir),
    ]
    return argv


# --- Flow ------------------------------------------------------------------

def _backend_installed():
    """True if a llama-cpp-python backend is importable (mirrors cogito-model)."""
    return importlib.util.find_spec("llama_cpp") is not None


def _offload_desc(gpu_layers):
    if gpu_layers < 0:
        return "all layers on GPU"
    if gpu_layers == 0:
        return "CPU only"
    return f"{gpu_layers} layers on GPU"


_PASSTHROUGH = ("genesis_type", "genesis_prompt", "cycles", "context_size",
                "tokens_per_cycle", "temperature", "top_p", "repeat_penalty")


def main(argv=None, *, detect_fn=None, catalog=None, input_fn=input,
         lister=None, run_fn=None, viz_fn=None, out=None, now=None,
         backend_installed=None, isatty=None) -> int:
    out = out or sys.stdout
    detect_fn = detect_fn or detect.detect
    catalog = list(catalog) if catalog is not None else list(models.iter_models())
    run_fn = run_fn or cogito.main
    viz_fn = viz_fn or visualize.main
    backend_installed = backend_installed or _backend_installed
    now = now or (lambda: time.strftime("%Y%m%d_%H%M%S"))
    isatty = isatty or sys.stdin.isatty

    parser = argparse.ArgumentParser(
        prog="cogito-run",
        description="Launch a COGITO ponder loop from a downloaded model, then "
                    "visualize it.",
    )
    parser.add_argument("--model", metavar="PATH",
                        help="GGUF to run (default: discover under ./models)")
    parser.add_argument("--dest", metavar="PATH", default="models",
                        help="directory to look for models in (default: ./models)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the wizard and launch with recommended defaults")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan (model, run command, viz) and launch nothing")
    # Pass-through overrides: pre-seed the wizard defaults / the --yes launch.
    parser.add_argument("--genesis-type")
    parser.add_argument("--genesis-prompt")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--tokens-per-cycle", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repeat-penalty", type=float)
    args = parser.parse_args(argv)

    if not backend_installed():
        print("No llama-cpp-python backend installed. "
              "Run 'uv run cogito-install' first.", file=out)
        return 1

    # Interactive only when a human is at a tty and has not asked to skip prompts.
    interactive = isatty() and not args.yes and not args.dry_run

    # Everything up to launch may prompt; a Ctrl-C here means "not now", cleanly.
    try:
        model_path = resolve_model(args.model, args.dest, lister=lister,
                                   input_fn=input_fn, out=out, interactive=interactive)
        if model_path is None:
            return 1

        detection = detect_fn()
        gpu_layers, note = resolve_offload(model_path, catalog, detection)
        print(f"Model:   {model_path}", file=out)
        print(f"Offload: {_offload_desc(gpu_layers)} ({note})", file=out)

        defaults = dict(DEFAULTS)
        for key in _PASSTHROUGH:
            value = getattr(args, key)
            if value is not None:
                defaults[key] = value

        params = prompt_params(defaults, input_fn, out) if interactive else dict(defaults)
    except KeyboardInterrupt:
        print("\nCancelled before launch. Start it when you're ready with "
              "'uv run cogito-run'.", file=out)
        return 0

    log_dir = str(Path("logs") / f"run_{now()}")

    run_argv = build_argv(model_path, params, gpu_layers, log_dir)

    if args.dry_run:
        print("\n[dry run] would launch:", file=out)
        print("    uv run cogito " + " ".join(run_argv), file=out)
        print(f"    then: uv run cogito-viz {log_dir}", file=out)
        return 0

    print(f"\nStarting the ponder loop (Ctrl-C to stop early)...\n", file=out)
    try:
        run_fn(run_argv)
    except KeyboardInterrupt:
        print("\nRun interrupted -- visualizing what was captured so far.", file=out)

    try:
        # --no-show: the handoff is headless (the PNG is the artifact); avoids a
        # spurious "FigureCanvasAgg is non-interactive" warning from plt.show().
        viz_fn([log_dir, "--no-show"])
    except Exception as exc:  # a viz hiccup must never sink a good run
        print(f"\n(cogito-viz could not run: {exc}. "
              f"Visualize later with: uv run cogito-viz {log_dir})", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
