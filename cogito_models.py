#!/usr/bin/env python3
"""
COGITO model catalog -- the single source of truth for the curated GGUF models
the guided picker (cogito_modelpick.py) offers.

This is code-as-config (per the UIST runtime-first principle: no config file is
read at runtime), mirroring cogito_backends.py. The catalog drives both the
picker display and the VRAM/RAM fit filtering.

Curation rules:
  * every repo is UNGATED, so the auth-free download path never needs a token
    (HF_TOKEN is honored as a convenience for advanced users, but never required);
  * one curated quant per entry (Q4_K_M -- the size/quality sweet spot); a model
    that does not fit steps the user down the ladder rather than offering a quant
    menu.

size_bytes is the on-disk / download size in bytes, pinned live against the
Hugging Face resolve URL at build time (Content-Length). sha256 is pinned
opportunistically for models we have actually downloaded and verified; None
until then (size is always verified, hash when present). n_layers is the static
transformer block count, used to estimate a partial-offload --gpu-layers value.
"""

from dataclasses import dataclass
from typing import Optional

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"


@dataclass(frozen=True)
class Model:
    key: str                  # short stable id, e.g. "qwen2.5-7b"
    name: str                 # human label
    params_b: float           # parameter count in billions
    hf_repo: str              # ungated HF repo, e.g. "bartowski/Qwen2.5-7B-Instruct-GGUF"
    hf_file: str              # GGUF filename within the repo
    quant: str                # e.g. "Q4_K_M"
    size_bytes: int           # download / on-disk size (Content-Length)
    n_layers: int             # transformer block count (partial-offload estimate)
    sha256: Optional[str] = None  # pinned when known; verified when present
    notes: str = ""


# Sizes are pinned from the HF Content-Length (see the plan's Task 1 verify step).
MODELS = [
    Model(
        key="qwen2.5-0.5b",
        name="Qwen2.5 0.5B Instruct",
        params_b=0.5,
        hf_repo="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        hf_file="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=397808192,
        n_layers=24,
        notes="Tiny; runs on CPU or a small GPU (e.g. the GTX 1070). Fast, limited.",
    ),
    Model(
        key="qwen2.5-1.5b",
        name="Qwen2.5 1.5B Instruct",
        params_b=1.5,
        hf_repo="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        hf_file="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=986048768,
        n_layers=28,
        notes="Small; comfortable on 8 GB cards and modern CPUs.",
    ),
    Model(
        key="qwen2.5-7b",
        name="Qwen2.5 7B Instruct",
        params_b=7.0,
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        hf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=4683074240,
        n_layers=28,
        notes="The small default: good results, fits ~6-8 GB VRAM at Q4.",
    ),
    Model(
        key="qwen2.5-14b",
        name="Qwen2.5 14B Instruct",
        params_b=14.0,
        hf_repo="bartowski/Qwen2.5-14B-Instruct-GGUF",
        hf_file="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=8988110976,
        n_layers=48,
        notes="Mid-size; 12-16 GB cards, or partial offload / CPU with enough RAM.",
    ),
    Model(
        key="qwen2.5-32b",
        name="Qwen2.5 32B Instruct",
        params_b=32.0,
        hf_repo="bartowski/Qwen2.5-32B-Instruct-GGUF",
        hf_file="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=19851336576,
        n_layers=64,
        notes="The model every CHANGELOG run used; needs ~24 GB VRAM (e.g. RTX 5090).",
    ),
    Model(
        key="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        params_b=7.0,
        hf_repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        hf_file="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=4372812000,
        n_layers=32,
        notes="Apache-2.0; a different lab/lineage from Qwen for contrast.",
    ),
    Model(
        key="phi-3.5-mini",
        name="Phi-3.5 mini Instruct",
        params_b=3.8,
        hf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        hf_file="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=2393232672,
        n_layers=32,
        notes="MIT; trained heavily on synthetic 'textbook' data -- a distinct voice.",
    ),
    Model(
        key="deepseek-r1-qwen-7b",
        name="DeepSeek-R1-Distill-Qwen 7B",
        params_b=7.0,
        hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        hf_file="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        quant="Q4_K_M",
        size_bytes=4683073504,
        n_layers=28,
        notes="MIT; an explicit reasoning model that thinks out loud -- a fascinating "
              "subject for a recursive self-prompting loop.",
    ),
]


def iter_models():
    """Yield the curated models in catalog order."""
    return iter(MODELS)


def by_key(key):
    """Return the Model with this key, or None if unknown."""
    for model in MODELS:
        if model.key == key:
            return model
    return None


def keys():
    """Return the model keys in catalog order."""
    return [model.key for model in MODELS]


def download_url(model):
    """The Hugging Face resolve URL for this model's GGUF file."""
    return HF_RESOLVE.format(repo=model.hf_repo, file=model.hf_file)
