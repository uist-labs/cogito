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

import re
from dataclasses import dataclass
from typing import Optional, Tuple


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
