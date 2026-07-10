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
import shutil
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


# --- guarded source-build path ---------------------------------------------

# cmake_flag -> (SDK probe tool, human label, vendor installer link). The SDK is
# never auto-installed (privileged, driver-version-matched); we link the vendor.
_SDK = {
    "-DGGML_CUDA=on": ("nvcc", "the CUDA Toolkit",
                       "https://developer.nvidia.com/cuda-downloads"),
    "-DGGML_HIP=on": ("hipcc", "ROCm",
                      "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"),
    "-DGGML_VULKAN=on": ("glslc", "the Vulkan SDK",
                         "https://vulkan.lunarg.com/sdk/home"),
}

# Package-manager -> (install prefix, build-tool package names).
_PKGS = {
    "dnf": ("sudo dnf install",
            {"compiler": "gcc-c++", "cmake": "cmake", "ninja": "ninja-build"}),
    "apt": ("sudo apt install",
            {"compiler": "g++", "cmake": "cmake", "ninja": "ninja-build"}),
    "brew": ("brew install", {"compiler": None, "cmake": "cmake", "ninja": "ninja"}),
}


def _pkg_manager(which, system):
    if system == "Darwin" or which("brew"):
        return "brew"
    if which("dnf"):
        return "dnf"
    if which("apt-get") or which("apt"):
        return "apt"
    return None


def _pkg_hint(missing_build, which, system):
    mgr = _pkg_manager(which, system)
    if mgr is None:
        return None
    prefix, names = _PKGS[mgr]
    pkgs = [names[m] for m in missing_build if names.get(m)]
    if not pkgs:
        if "compiler" in missing_build and mgr == "brew":
            return "xcode-select --install"
        return None
    return prefix + " " + " ".join(pkgs)


def _preflight(cmake_flag, which):
    """Return (missing_build_tools, missing_sdk_tuple_or_None) for a source build."""
    missing_build = []
    if not (which("cc") or which("gcc")) or not (which("c++") or which("g++")):
        missing_build.append("compiler")
    if not which("cmake"):
        missing_build.append("cmake")
    if not (which("make") or which("ninja")):
        missing_build.append("ninja")
    sdk = _SDK.get(cmake_flag)
    missing_sdk = sdk if (sdk and not which(sdk[0])) else None
    return missing_build, missing_sdk


def _print_prereqs(missing_build, missing_sdk, which, system, out):
    print("Cannot build from source yet -- missing prerequisites:", file=out)
    if missing_build:
        print(f"  Build tools: {', '.join(missing_build)}", file=out)
        hint = _pkg_hint(missing_build, which, system)
        if hint:
            print(f"    Install:  {hint}", file=out)
    if missing_sdk:
        tool, label, link = missing_sdk
        print(f"  {label} ({tool}) not found.", file=out)
        print(f"    Install it from the vendor: {link}", file=out)
    print("  Then re-run: cogito-install", file=out)


def source_build(backend_key, *, runner, which, out, system=None) -> int:
    """Build llama-cpp-python from source for a backend; always fall back to cpu.

    Pre-flights the toolchain + SDK. Missing prereqs -> print exact guidance (never
    run a privileged install ourselves) and install cpu. Build failure -> install cpu.
    Success -> ``uv sync --inexact`` so the exact sync does not clobber the wheel.
    """
    be = backends.by_key(backend_key)
    flag = be.cmake_flag
    missing_build, missing_sdk = _preflight(flag, which)
    if missing_build or missing_sdk:
        _print_prereqs(missing_build, missing_sdk, which, system, out)
        print("Falling back to the cpu backend.", file=out)
        return runner(_sync_cmd("cpu", switching=False))

    print(f"Building llama-cpp-python from source for {backend_key} "
          f"(CMAKE_ARGS={flag}); this compiles locally and takes several minutes.",
          file=out)
    build = ["env", f"CMAKE_ARGS={flag}", "uv", "pip", "install", "llama-cpp-python",
             "--reinstall-package", "llama-cpp-python", "--no-cache"]
    if runner(build) != 0:
        print("Source build failed; falling back to the cpu backend.", file=out)
        return runner(_sync_cmd("cpu", switching=False))
    return runner(["uv", "sync", "--inexact"])


def main(argv=None, *, detect_fn=None, runner=None, input_fn=input,
         is_installed=None, isatty=None, which=None, out=None) -> int:
    out = out or sys.stdout
    detect_fn = detect_fn or detect.detect
    is_installed = is_installed or _llama_cpp_installed
    isatty = isatty or sys.stdin.isatty
    which = which or shutil.which

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
            # A GPU on a wheel-incompatible host: offer the source build first
            # (never auto: it is heavy and needs a toolchain). Decline -> the menu.
            if rec.source_build_key:
                ans = input_fn(
                    f"Build the GPU backend ({rec.source_build_key}) from source? "
                    f"Needs a compiler + SDK and several minutes. [y/N]: "
                ).strip().lower()
                if ans in ("y", "yes"):
                    rc = source_build(rec.source_build_key, runner=runner,
                                      which=which, out=out, system=det.system)
                    if rc == 0:
                        _next_steps(out)
                    return rc
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
