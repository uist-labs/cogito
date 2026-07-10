#!/usr/bin/env python3
"""COGITO guided model picker -- the ``cogito-model`` console command.

Detects the host's memory (cogito_detect), annotates the curated GGUF catalog
(cogito_models) by how each model fits, lets the operator pick (recommendation
pre-selected; Enter accepts), prechecks free space, downloads with resume
(cogito_modeldl), and prints a ready-to-paste first-run command. Re-runnable to
grab another model later.

Interaction follows kiln/S1: linear, one decision at a time, ASCII-only, the
recommendation pre-selected, consequential actions shown before they run. Every
side effect (detection, catalog, stdin, download, disk, backend probe) is
injected so the flow is fully testable offline.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import cogito_detect as detect
import cogito_models as models
import cogito_modeldl as modeldl

_TAGS = {
    "gpu": "fits on GPU",
    "partial": "partial offload",
    "cpu": "CPU (RAM)",
    "cpu_oversized": "too big for RAM",
}


def _backend_installed():
    """True if a llama-cpp-python backend is importable (S1 installed one)."""
    return importlib.util.find_spec("llama_cpp") is not None


def _gb(n):
    return f"{n / 1e9:.1f} GB"


def _render_catalog(catalog, det, recommended, out):
    print("COGITO model picker", file=out)
    print("-------------------", file=out)
    mem = []
    if det.vram_total_mb:
        mem.append(f"{det.vram_total_mb} MiB VRAM")
    if det.ram_total_mb:
        mem.append(f"{det.ram_total_mb} MiB RAM")
    print("Detected memory: " + (", ".join(mem) or "unknown"), file=out)
    print("", file=out)
    for i, model in enumerate(catalog, start=1):
        tag = _TAGS[models.fit(model, det).tier]
        mark = "  <- recommended" if model.key == recommended.key else ""
        print(f"  [{i}] {model.key:<20} {_gb(model.size_bytes):>8}  "
              f"{tag:<16} {model.name}{mark}", file=out)
    print("", file=out)


def _choose(catalog, recommended, input_fn, out):
    """Return the chosen Model. Enter accepts the recommendation."""
    prompt = f"Press Enter for {recommended.key}, or choose a number: "
    while True:
        raw = input_fn(prompt).strip()
        if raw == "":
            return recommended
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(catalog):
                return catalog[n - 1]
        print(f"  Please press Enter or enter a number 1-{len(catalog)}.", file=out)


def _maybe_step_down(catalog, det, chosen, input_fn, out):
    """Offer the next model down when the pick does not fully fit. Returns a Model."""
    f = models.fit(chosen, det)
    if f.tier not in ("partial", "cpu_oversized"):
        return chosen
    smaller = models.step_down(catalog, det, chosen)
    if smaller is None:
        return chosen
    ans = input_fn(
        f"{chosen.name}: {f.reason}. Download the smaller {smaller.name} that fits "
        f"better instead? [y/N]: "
    ).strip().lower()
    return smaller if ans in ("y", "yes") else chosen


def _run_command(out, chosen, target, f, backend_ok):
    print("", file=out)
    print(f"Model ready: {target}", file=out)
    if not backend_ok:
        print("  Note: no llama-cpp-python backend is installed yet -- run "
              "'uv run cogito-install' first.", file=out)
    print("  Run an experiment:", file=out)
    print(f"    uv run cogito --model {target} --gpu-layers {f.gpu_layers} "
          f"--genesis-type mirror --cycles 50", file=out)
    if f.tier == "partial":
        print(f"    (~{f.gpu_layers} of {chosen.n_layers} layers on GPU; raise "
              f"--gpu-layers if you have headroom, lower it if you hit OOM)", file=out)


def _existing_ancestor(path):
    """The nearest existing directory at or above ``path`` (for a space check)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def main(argv=None, *, detect_fn=None, catalog=None, input_fn=input,
         downloader=None, out=None, isatty=None, disk_usage=None,
         backend_installed=None) -> int:
    out = out or sys.stdout
    detect_fn = detect_fn or detect.detect
    catalog = list(catalog) if catalog is not None else list(models.iter_models())
    downloader = downloader or modeldl.download
    isatty = isatty or sys.stdin.isatty
    disk_usage = disk_usage or shutil.disk_usage
    backend_installed = backend_installed or _backend_installed

    parser = argparse.ArgumentParser(
        prog="cogito-model",
        description="Pick a curated GGUF model that fits your hardware and download it.",
    )
    parser.add_argument("--model", metavar="KEY",
                        help="skip the picker and download this model (e.g. qwen2.5-7b)")
    parser.add_argument("--dest", metavar="PATH", default="models",
                        help="directory to download into (default: ./models)")
    parser.add_argument("--yes", action="store_true",
                        help="download the recommended model without prompting")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan (model, URL, run command) and download nothing")
    args = parser.parse_args(argv)

    det = detect_fn()

    if args.model is not None:
        chosen = models.by_key(args.model)
        if chosen is None:
            print(f"Unknown model '{args.model}'. Choose one of: "
                  f"{', '.join(models.keys())}", file=out)
            return 2
    else:
        recommended = models.recommended_model(catalog, det)
        _render_catalog(catalog, det, recommended, out)
        if args.yes or args.dry_run:
            chosen = recommended
        elif not isatty():
            print("(non-interactive input; catalog listed above, nothing downloaded. "
                  "Pass --model KEY or --yes to fetch one.)", file=out)
            return 0
        else:
            chosen = _choose(catalog, recommended, input_fn, out)
            chosen = _maybe_step_down(catalog, det, chosen, input_fn, out)

    f = models.fit(chosen, det)
    target = Path(args.dest) / chosen.hf_file
    url = models.download_url(chosen)
    backend_ok = backend_installed()

    if target.exists() and target.stat().st_size == chosen.size_bytes:
        print(f"Already downloaded: {target}", file=out)
        _run_command(out, chosen, target, f, backend_ok)
        return 0

    if args.dry_run:
        print(f"[dry run] would download {chosen.name} ({_gb(chosen.size_bytes)})",
              file=out)
        print(f"          from {url}", file=out)
        print(f"          to   {target}", file=out)
        _run_command(out, chosen, target, f, backend_ok)
        return 0

    need = int(chosen.size_bytes * models.HEADROOM)
    free = disk_usage(str(_existing_ancestor(Path(args.dest)))).free
    if free < need:
        print(f"Not enough disk space at {args.dest}: need ~{_gb(need)}, "
              f"have {_gb(free)} free. Choose a smaller model or a different --dest.",
              file=out)
        return 1

    Path(args.dest).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {chosen.name} ({_gb(chosen.size_bytes)}) to {target} ...",
          file=out)
    try:
        downloader(url, target, size_bytes=chosen.size_bytes,
                   sha256=chosen.sha256, out=out)
    except modeldl.DownloadError as exc:
        print(str(exc), file=out)
        return 1
    _run_command(out, chosen, target, f, backend_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
