#!/usr/bin/env python3
"""
COGITO backend catalog -- the single source of truth for the llama-cpp-python
install backends COGITO supports.

This is code-as-config (per the UIST runtime-first principle: no config file is
read at runtime). Two consumers share this one table so neither hardcodes its
own copy:

  * documentation / the README hardware-support matrix, and
  * the forthcoming TUI installer's hardware detection and recommendation.

Each backend ``key`` is chosen to equal a ``uv sync --extra <key>`` extra, so a
detected backend maps straight to an install command.

llama-cpp-python ships pre-built, Python-version-agnostic (py3-none) wheels per
backend at the abetlen index (see ``index_url``). ``cmake_flag`` is the flag to
use only when falling back to a from-source build (older glibc, or an unlisted
backend). Verified against llama-cpp-python 0.3.33 and the abetlen wheel index
on 2026-07-09.
"""

from dataclasses import dataclass
from typing import Optional

ABETLEN_WHL = "https://abetlen.github.io/llama-cpp-python/whl"


@dataclass(frozen=True)
class Backend:
    key: str                  # uv extra name == detector key, e.g. "cu124"
    name: str                 # human-readable label
    index_url: str            # abetlen pre-built-wheel index for this backend
    cmake_flag: str           # CMAKE flag for a from-source fallback ("" = none)
    min_glibc: Optional[str]  # glibc floor for the wheel, or None (e.g. macOS)
    kind: str                 # "wheel" -- from-source is always the fallback
    notes: str


BACKENDS = [
    Backend(
        key="cpu",
        name="CPU (no GPU)",
        index_url=f"{ABETLEN_WHL}/cpu",
        # Plain CPU build needs no flag; for OpenBLAS use
        # "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS".
        cmake_flag="",
        min_glibc="2.17",
        kind="wheel",
        notes="Runs anywhere (manylinux2014, macOS arm64, Windows). Slowest; universal fallback.",
    ),
    Backend(
        key="cu124",
        name="NVIDIA CUDA 12.4",
        index_url=f"{ABETLEN_WHL}/cu124",
        cmake_flag="-DGGML_CUDA=on",
        min_glibc="2.35",
        kind="wheel",
        notes="NVIDIA GPUs on a CUDA 12.4 runtime. Needs glibc 2.35+ (AL10 / Ubuntu 22.04+); EL9 falls back to source.",
    ),
    Backend(
        key="cu130",
        name="NVIDIA CUDA 13.0",
        index_url=f"{ABETLEN_WHL}/cu130",
        cmake_flag="-DGGML_CUDA=on",
        min_glibc="2.35",
        kind="wheel",
        notes="NVIDIA GPUs on a CUDA 13.0 runtime. Needs glibc 2.35+.",
    ),
    Backend(
        key="metal",
        name="Apple Silicon (Metal)",
        index_url=f"{ABETLEN_WHL}/metal",
        cmake_flag="-DGGML_METAL=on",
        min_glibc=None,
        kind="wheel",
        notes="Apple Silicon Macs (arm64), macOS 11+.",
    ),
    Backend(
        key="vulkan",
        name="Vulkan (cross-vendor GPU)",
        index_url=f"{ABETLEN_WHL}/vulkan",
        cmake_flag="-DGGML_VULKAN=on",
        min_glibc="2.17",
        kind="wheel",
        notes="Cross-vendor GPU (AMD / Intel / NVIDIA) via Vulkan; manylinux2014. Needs a Vulkan driver.",
    ),
    Backend(
        key="rocm72",
        name="AMD ROCm 7.2",
        index_url=f"{ABETLEN_WHL}/rocm72",
        cmake_flag="-DGGML_HIP=on",
        min_glibc="2.35",
        kind="wheel",
        notes="AMD GPUs via ROCm 7.2 (Linux). Needs glibc 2.35+ and a ROCm install.",
    ),
]


def iter_backends():
    """Yield the curated backends in catalog order."""
    return iter(BACKENDS)


def by_key(key):
    """Return the Backend with this uv-extra key, or None if unknown."""
    for b in BACKENDS:
        if b.key == key:
            return b
    return None


def keys():
    """Return the uv-extra keys (== detector keys) in catalog order."""
    return [b.key for b in BACKENDS]
