#!/usr/bin/env python3
"""COGITO hardware detection -- pure, stdlib-only, side-effect-free.

This module answers "what GPU is this, which CUDA/ROCm version, what glibc" so the
installer wizard (cogito_install.py) can recommend a llama-cpp-python backend. It is
split into two layers:

  * pure ``parse_* `` functions (text -> value; never raise), and
  * thin probe runners + ``detect()`` (added alongside) that shell out and feed the
    parsers.

It is deliberately free of any cogito-specific imports (no cogito_backends here) so it
stays lift-ready for reuse by other GPU-backend projects -- see the design's reuse note.
Nothing here imports llama_cpp or performs a network/install action.
"""

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """Immutable snapshot of the host's relevant hardware/runtime facts.

    Every field defaults to the "unknown"/absent value, so a partial probe never
    leaves the record in an invalid state.
    """

    system: str = ""                                   # platform.system()
    machine: str = ""                                  # platform.machine()
    nvidia: bool = False
    nvidia_driver: Optional[str] = None
    nvidia_max_cuda: Optional[Tuple[int, int]] = None  # driver's max CUDA runtime
    nvidia_compute_cap: Optional[float] = None         # GPU arch, e.g. 6.1 (Pascal)
    amd_gpu: bool = False
    rocm_ok: bool = False                              # a ROCm GPU agent is present
    vulkan: bool = False
    glibc: Optional[Tuple[int, int]] = None
    vram_total_mb: Optional[int] = None                # total GPU memory (MiB)
    ram_total_mb: Optional[int] = None                 # total system RAM (MiB)


@dataclass(frozen=True)
class Recommendation:
    """A backend recommendation with the reasoning the wizard prints verbatim.

    ``source_build_key`` names the backend the user could build from source when the
    prebuilt wheel is not viable (e.g. a GPU on glibc < 2.35); None when the
    recommended key is directly installable.
    """

    key: str
    rationale: str
    caveats: Tuple[str, ...] = ()
    source_build_key: Optional[str] = None


# glibc floor for the CUDA/ROCm prebuilt wheels (cpu/vulkan are manylinux2014).
GLIBC_WHEEL_FLOOR = (2, 35)
# CUDA 13 dropped everything below Turing (compute capability 7.5).
TURING_CC = 7.5


# --- pure parsers (text -> value; never raise) -----------------------------

def parse_nvidia_smi_cuda(text: str) -> Optional[Tuple[int, int]]:
    """Driver's max CUDA runtime from the nvidia-smi banner ("CUDA Version: 13.0")."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_driver_version(text: str) -> Optional[str]:
    """NVIDIA driver version from the nvidia-smi banner ("Driver Version: 580.173.02")."""
    m = re.search(r"Driver Version:\s*([\d.]+)", text)
    return m.group(1) if m else None


def parse_compute_cap(text: str) -> Optional[float]:
    """GPU compute capability from ``nvidia-smi --query-gpu=compute_cap``.

    Takes the first GPU's value (single-GPU is the norm). Returns None on N/A/garbage.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            return None
    return None


def parse_glibc(text: str) -> Optional[Tuple[int, int]]:
    """glibc (major, minor) from "2.39" or the confstr form "glibc 2.39"."""
    text = text.strip()
    if not text:
        return None
    token = text.split()[-1]  # drop an optional "glibc " prefix
    m = re.match(r"(\d+)\.(\d+)", token)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_lspci_amd(text: str) -> bool:
    """True if lspci shows an AMD/ATI display controller.

    Vendor match is word-boundaried so "VGA compatible controller" does not
    false-positive on the "ati" inside "compatible".
    """
    vendor = re.compile(r"\bAMD\b|\bATI\b|ADVANCED MICRO DEVICES")
    for line in text.splitlines():
        upper = line.upper()
        if ("VGA" in upper or "DISPLAY" in upper or "3D CONTROLLER" in upper) and (
            vendor.search(upper)
        ):
            return True
    return False


def parse_rocminfo_agents(text: str) -> bool:
    """True if rocminfo lists a GPU agent (a working ROCm stack, not just a GPU)."""
    return re.search(r"Device Type:\s*GPU", text) is not None


def parse_vram(text: str) -> Optional[int]:
    """Total VRAM in MiB from ``nvidia-smi --query-gpu=memory.total`` (nounits).

    Takes the first GPU's value (single-GPU is the norm). Returns None on
    N/A/garbage/empty.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return int(line)
        except ValueError:
            return None
    return None


def parse_meminfo(text: str) -> Optional[int]:
    """Total system RAM in MiB from /proc/meminfo's ``MemTotal: N kB`` line."""
    m = re.search(r"^MemTotal:\s*(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) // 1024 if m else None


# --- probe runners + orchestrator ------------------------------------------

def _run(cmd) -> str:
    """Run a probe command and return stdout, swallowing any failure into "".

    Read-only by construction (callers only pass query commands). A missing tool,
    nonzero exit, or timeout all degrade to "" so detection never raises.
    """
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
        return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_meminfo() -> str:
    """Read /proc/meminfo, returning "" if it is absent (non-Linux) or unreadable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _ram_mb(read: Callable[[], str] = _read_meminfo, sysconf=os.sysconf) -> Optional[int]:
    """Total system RAM in MiB: /proc/meminfo first, then a sysconf fallback.

    Both sources are injected so the fallback path is testable. Any failure
    degrades to None so detection never raises.
    """
    ram = parse_meminfo(read())
    if ram is not None:
        return ram
    try:
        return sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError, TypeError):
        return None


def detect(
    *,
    run: Callable[[list], str] = _run,
    which: Callable[[str], Optional[str]] = shutil.which,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    libc: Optional[Tuple[str, str]] = None,
    ram_mb_fn: Optional[Callable[[], Optional[int]]] = None,
) -> Detection:
    """Probe the host and return a Detection. Side effects are injected for testing.

    ``run``/``which`` default to real subprocess/PATH lookups; tests pass fakes.
    ``system``/``machine``/``libc`` default to the real platform values.
    """
    system = system or platform.system()
    machine = machine or platform.machine()

    nvidia = False
    nvidia_driver = nvidia_max_cuda = nvidia_compute_cap = vram_total_mb = None
    if which("nvidia-smi"):
        smi = run(["nvidia-smi"])
        nvidia_driver = parse_driver_version(smi)
        nvidia_max_cuda = parse_nvidia_smi_cuda(smi)
        # Only a genuinely working driver (banner parsed) counts as an NVIDIA host.
        nvidia = bool(nvidia_driver or nvidia_max_cuda)
        if nvidia:
            nvidia_compute_cap = parse_compute_cap(
                run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
            )
            vram_total_mb = parse_vram(run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"]
            ))

    rocm_ok = False
    if which("rocminfo"):
        rocm_ok = parse_rocminfo_agents(run(["rocminfo"]))
    amd_gpu = rocm_ok
    if which("lspci"):
        amd_gpu = amd_gpu or parse_lspci_amd(run(["lspci"]))
    # /opt/rocm is a weaker hint than a live agent; it does not upgrade rocm_ok.

    vulkan = which("vulkaninfo") is not None

    if libc is None:
        try:
            libc = platform.libc_ver()
        except OSError:
            libc = ("", "")
    glibc = parse_glibc(libc[1]) if libc and libc[1] else None

    ram_total_mb = (ram_mb_fn or _ram_mb)()

    return Detection(
        system=system,
        machine=machine,
        nvidia=nvidia,
        nvidia_driver=nvidia_driver,
        nvidia_max_cuda=nvidia_max_cuda,
        nvidia_compute_cap=nvidia_compute_cap,
        amd_gpu=amd_gpu,
        rocm_ok=rocm_ok,
        vulkan=vulkan,
        glibc=glibc,
        vram_total_mb=vram_total_mb,
        ram_total_mb=ram_total_mb,
    )


# --- recommendation (pure Detection -> Recommendation) ---------------------

def _glibc_meets_floor(det: Detection) -> bool:
    return det.glibc is not None and det.glibc >= GLIBC_WHEEL_FLOOR


def _nvidia_cuda_key(det: Detection) -> str:
    """cu124 vs cu130 by GPU compute capability -- NOT by driver max-CUDA.

    CUDA 13 removed pre-Turing (CC < 7.5) support, so those GPUs need the CUDA 12.x
    wheel even when the driver advertises CUDA 13. Unknown CC -> conservative cu124.
    """
    cc = det.nvidia_compute_cap
    if cc is None or cc < TURING_CC:
        return "cu124"
    if det.nvidia_max_cuda is not None and det.nvidia_max_cuda >= (13, 0):
        return "cu130"
    return "cu124"


def recommend(det: Detection) -> Recommendation:
    """Map a Detection to a backend, conservative on uncertainty. Pure; no catalog."""
    # 1. Apple Silicon.
    if det.system == "Darwin" and det.machine in ("arm64", "aarch64"):
        return Recommendation("metal", "Apple Silicon (arm64) detected -- Metal backend.")

    # 2/3. NVIDIA.
    if det.nvidia:
        cuda_key = _nvidia_cuda_key(det)
        cc = det.nvidia_compute_cap
        cc_str = (f"compute capability {cc}" if cc is not None
                  else "unknown compute capability")
        if _glibc_meets_floor(det):
            return Recommendation(
                cuda_key, f"NVIDIA GPU ({cc_str}); recommending {cuda_key}."
            )
        why = ("glibc below the 2.35 floor" if det.glibc is not None
               else "glibc could not be verified")
        return Recommendation(
            "cpu",
            f"NVIDIA GPU ({cc_str}) but {why} for the prebuilt CUDA wheel; using cpu. "
            f"A source build enables the GPU.",
            caveats=(
                f"The prebuilt {cuda_key} wheel needs glibc >= 2.35; build "
                f"llama-cpp-python from source to use the GPU.",
            ),
            source_build_key=cuda_key,
        )

    # 4/5. AMD.
    if det.amd_gpu:
        if det.rocm_ok and _glibc_meets_floor(det):
            return Recommendation(
                "rocm72", "AMD GPU with a working ROCm stack; recommending rocm72."
            )
        if det.vulkan:
            note = ("ROCm not detected" if not det.rocm_ok
                    else "glibc below the 2.35 floor for the ROCm wheel")
            return Recommendation(
                "vulkan", f"AMD GPU; {note} -- using the cross-vendor Vulkan backend."
            )
        return Recommendation(
            "cpu", "AMD GPU but no usable ROCm or Vulkan runtime; using cpu."
        )

    # 6. Universal fallback.
    return Recommendation(
        "cpu", "No supported GPU detected; using the universal cpu backend."
    )
