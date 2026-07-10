#!/usr/bin/env python3
"""COGITO guided backend installer -- the ``cogito-install`` console command.

Detects the host's hardware (cogito_detect), recommends a llama-cpp-python backend,
shows the reasoning, and installs it via ``uv sync --extra <backend>``. The newcomer
presses Enter to accept the recommendation; the full catalog stays overridable.

Interaction philosophy follows kiln: linear, one decision at a time, ASCII-only output,
the recommended choice pre-selected, consequential commands shown before they run. All
side effects (detection, subprocess, stdin) are injected so the flow is fully testable.
"""

import argparse
import functools
import importlib.util
import subprocess
import sys

import cogito_backends as backends
import cogito_detect as detect


# --- install runner (the single side-effect seam) --------------------------

def _default_runner(cmd, *, dry_run, out):
    """Print the command, then run it (streamed) unless --dry-run."""
    print("Running: " + " ".join(cmd), file=out)
    if dry_run:
        print("(dry run -- nothing executed)", file=out)
        return 0
    return subprocess.call(cmd)


def _llama_cpp_installed(_key):
    """True if a llama-cpp-python build is already present (switch => reinstall)."""
    return importlib.util.find_spec("llama_cpp") is not None


# --- menu / rendering ------------------------------------------------------

def _ordered_keys(recommended):
    """Catalog keys with the recommended one floated to the front."""
    rest = [k for k in backends.keys() if k != recommended]
    return [recommended] + rest


def _render(det, rec, ordered, out):
    print("COGITO backend installer", file=out)
    print("-------------------------", file=out)
    print("Detected hardware:", file=out)
    gpu = "none"
    if det.nvidia:
        cc = det.nvidia_compute_cap
        gpu = f"NVIDIA (driver {det.nvidia_driver}, compute_cap {cc})"
    elif det.amd_gpu:
        gpu = "AMD" + (" (ROCm)" if det.rocm_ok else "")
    print(f"  Platform : {det.system} {det.machine}", file=out)
    print(f"  GPU      : {gpu}", file=out)
    if det.glibc:
        print(f"  glibc    : {det.glibc[0]}.{det.glibc[1]}", file=out)
    print("", file=out)
    be = backends.by_key(rec.key)
    print(f"Recommended backend: {rec.key}  ({be.name})", file=out)
    print(f"  Why: {rec.rationale}", file=out)
    for caveat in rec.caveats:
        print(f"  Note: {caveat}", file=out)
    print("", file=out)
    for i, key in enumerate(ordered, start=1):
        mark = "  <- recommended" if key == rec.key else ""
        print(f"  [{i}] {key:<7} {backends.by_key(key).name}{mark}", file=out)
    print("", file=out)


def _choose(ordered, recommended, input_fn, out):
    """Return the chosen backend key. Enter accepts the recommendation."""
    prompt = f"Press Enter to accept {recommended}, or choose a number: "
    while True:
        raw = input_fn(prompt).strip()
        if raw == "":
            return recommended
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(ordered):
                return ordered[n - 1]
        print(f"  Please press Enter or enter a number 1-{len(ordered)}.", file=out)


def _next_steps(out):
    """Forward-pointing exit -- the single source for post-install guidance."""
    print("", file=out)
    print("Backend installed. Next steps:", file=out)
    print("  1. Get a GGUF model (see 'Getting a Model' in README.md).", file=out)
    print("  2. Run an experiment, e.g.:", file=out)
    print("       uv run cogito --model /path/to/model.gguf "
          "--genesis-type mirror --cycles 50", file=out)
    print("  Or watch the bundled demo now (no model/GPU):", file=out)
    print("       uv run cogito-viz examples/demo_run", file=out)


def _sync_cmd(key, *, switching):
    cmd = ["uv", "sync", "--extra", key]
    if switching:
        cmd += ["--reinstall-package", "llama-cpp-python", "--no-cache"]
    return cmd


def main(argv=None, *, detect_fn=None, runner=None, input_fn=input,
         is_installed=None, isatty=None, out=None) -> int:
    out = out or sys.stdout
    detect_fn = detect_fn or detect.detect
    is_installed = is_installed or _llama_cpp_installed
    isatty = isatty or sys.stdin.isatty

    parser = argparse.ArgumentParser(
        prog="cogito-install",
        description="Detect hardware and install the matching llama-cpp-python backend.",
    )
    parser.add_argument("--backend", metavar="KEY",
                        help="skip detection and install this backend (e.g. cu124)")
    parser.add_argument("--yes", action="store_true",
                        help="accept the recommendation without prompting")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the install command and exit without running it")
    args = parser.parse_args(argv)

    if runner is None:
        runner = functools.partial(_default_runner, dry_run=args.dry_run, out=out)

    # Explicit --backend bypasses detection entirely.
    if args.backend is not None:
        if backends.by_key(args.backend) is None:
            print(f"Unknown backend '{args.backend}'. Choose one of: "
                  f"{', '.join(backends.keys())}", file=out)
            return 2
        chosen = args.backend
    else:
        det = detect_fn()
        rec = detect.recommend(det)
        ordered = _ordered_keys(rec.key)
        _render(det, rec, ordered, out)
        # Prompt only when we can: an explicit --yes/--dry-run, or a non-interactive
        # stdin (pipe/CI/redirect), both accept the recommendation instead of hanging.
        if args.yes or args.dry_run or not isatty():
            chosen = rec.key
            if not args.yes and not args.dry_run:
                print("(non-interactive input; accepting the recommendation)", file=out)
        else:
            chosen = _choose(ordered, rec.key, input_fn, out)

    switching = bool(is_installed(chosen))
    rc = runner(_sync_cmd(chosen, switching=switching))
    if rc != 0:
        print(f"Install failed (exit {rc}).", file=out)
        return rc
    if not args.dry_run:
        _next_steps(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
