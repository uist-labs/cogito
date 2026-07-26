# COGITO uv Migration - Design Spec

- **Date:** 2026-07-09
- **Status:** Proposed (awaiting review)
- **Branch:** `claude/uv-migration`
- **Author:** Kenneth Kienle & Claude (Opus 4.8); technical verification by Fable

---

## 1. Goal & Why

Modernize COGITO's packaging and install path so anyone who finds it on GitHub can
pick it up and run it cleanly, across a wide range of hardware. The motivating use
case is a real one: onboarding a friend new to AI/ML, and any GitHub passer-by, with
a front door that does not lie about what it needs.

This spec covers the **uv migration spine** only. It makes COGITO *modern and
reproducible* and leaves clean seams for the TUI installer that follows immediately
after (its own spec). It deliberately does **not** build hardware auto-detection, a
model catalog, or space/VRAM-fit checks - those are the TUI's heart.

The design leans on llama.cpp/GGUF's breadth (old NVIDIA, CPU, Apple Silicon, AMD)
as the adoption story: "runs on the laptop you already have."

---

## 2. Scope

**In scope (this spec):**

- Move from venv + `requirements.txt` to `uv` (`pyproject.toml` + `uv.lock`).
- Flat-layout packaging with console entry points (`cogito`, `cogito-viz`).
- Hybrid dependency architecture: locked pure-Python core + per-backend
  llama-cpp-python extras (wheel-served) + a documented source-build escape hatch.
- Backend catalog as a single-source-of-truth **Python data module** (seam for the TUI).
- Slim `setup.sh` down to a `uv` bootstrap stub (seam for the TUI).
- Remove the hardcoded model download; replace with README documentation.
- README rewrite: install section, hardware-support matrix, "getting a model,"
  kill the `yourusername` placeholder.
- CHANGELOG update.

**Out of scope (deferred to the TUI installer spec):**

- Hardware/backend auto-detection (NVIDIA/Metal/ROCm/Vulkan/CPU).
- Curated model catalog with sizes, storage-path prompt, free-space precheck,
  VRAM-fit filtering, interactive model download.
- Any TUI framework decision (stdlib vs rich/textual/questionary).

The engine (`cogito.py` loop behavior) is **untouched**, consistent with prior work.

---

## 3. Current State

`main` (`06b93bb`, the merged onboarding-cleanup floor) has:

- `cogito.py` - single module, ~750 lines, `def main()` + argparse (line 751),
  lazy `llama_cpp` import (so `--help`, visualizers, demo run without the GPU dep).
- `visualize.py`, `visualize_advanced.py` - matplotlib visualizers (Agg, headless-safe).
- `requirements.txt` - `llama-cpp-python`, `numpy`, `matplotlib`.
- `setup.sh` - bash: creates a `.venv`, `pip install`s deps (correct `-DGGML_CUDA=on`
  CUDA flag), then an interactive model download with **hardcoded TheBloke URLs and no
  space check**. Also NVIDIA-or-CPU only.
- `examples/demo_run` + `generate_demo_data.py` - zero-GPU synthetic demo.

**Weak points this spec fixes:** bash/venv install (vs uv), no lockfile, model download
that assumes a high-end card and hardcodes soon-stale URLs, NVIDIA-or-CPU-only framing.

---

## 4. Design

### 4.1 Packaging - flat layout + pyproject + uv.lock + entry points

Keep the flat layout (no `src/` restructure - the engine stays put). Add:

- `pyproject.toml` with `[project]` metadata, `requires-python = ">=3.10"` (3.9 is EOL;
  `uv` can provision the interpreter via `uv python install`, so this floor does not
  gate users who lack it), hatchling build backend with a `py-modules` declaration for
  the flat modules.
- `[project.scripts]`:
  - `cogito = "cogito:main"`
  - `cogito-viz = "visualize:main"`
  - `cogito-viz-advanced = "visualize_advanced:main"` (promoted to a first-class command
    for discoverability/ease-of-use; `visualize_advanced.py` already has a clean `main()`,
    so the cost is one line and no visualizer-logic change).
- `uv.lock` committed (reproducible core env).
- Remove `requirements.txt` (superseded by pyproject/uv.lock).

Rationale: gives a clean `cogito` command (serves the visibility goal) with near-zero
churn to working code. A `src/` package is only warranted if COGITO is ever published
to PyPI, and even then the flat→src migration is cheap - so we do not pay for it now.

### 4.2 Dependency architecture - hybrid (verified 2026-07-09)

**Tier 1 - locked pure-Python core.** `numpy`, `matplotlib` declared as project
dependencies and pinned in `uv.lock`. Universal, reproducible, PyPI-sourced (robust,
unaffected by any third-party index availability). `uv sync` installs this.

**Tier 2a - wheel-served backends as locked uv extras (the common path).**
llama-cpp-python is now published as **`py3-none` ABI-agnostic wheels** (ctypes-based;
one wheel per backend covers every Python 3.x) at the abetlen GitHub Pages index,
current within ~1 day of each source release (0.3.33 as of this writing). Backends with
wheels: `cpu`, `metal`, `cu118`–`cu132`, `rocm72`, `hip-radeon`, `vulkan`.

Each curated backend becomes an **optional extra** pinned to its index via
`[[tool.uv.index]]` (with `explicit = true`) + `[tool.uv.sources]`, with
`[tool.uv] conflicts` declared between the mutually-exclusive backend extras. Then:

```
uv sync --extra cu124      # core + CUDA 12.4 llama-cpp-python, fully locked
uv sync --extra metal      # core + Apple Silicon Metal
uv sync --extra cpu        # core + CPU
```

This is the canonical uv "PyTorch index" idiom, and the current community
best-practice for llama-cpp-python specifically. It is fully locked/reproducible and
**avoids the `uv sync` exact-sync footgun** (see 4.2 gotchas) because the backend is a
declared dependency, not an out-of-lock install.

**Curated extras (initial set):** `cpu`, `cu124`, `cu130`, `metal`, `vulkan`, `rocm72`.
Chosen to cover the common hardware without listing every published CUDA point release;
extensible. (`hip-radeon` and additional `cuXXX` are one-line additions.)

**Tier 2b - source-build escape hatch (documented, not the default path).** For hosts
the wheels do not serve - **EL9 / glibc < 2.35** (CUDA & ROCm wheels are
`manylinux_2_35`; CPU & Vulkan wheels are `manylinux2014` and run anywhere), SYCL,
custom GPU-arch builds - document the source build:

```
# CUDA example; swap the flag per backend (see backend catalog)
CMAKE_ARGS="-DGGML_CUDA=on" \
  uv pip install llama-cpp-python --no-cache --reinstall-package llama-cpp-python
# equivalent, cache-visible PEP 517 form:
uv pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"
```

No `--no-build-isolation` needed - uv builds in an isolated PEP 517 env and
scikit-build-core auto-provisions `cmake`/`ninja`. Host must supply a C/C++ toolchain
and the backend SDK (nvcc for CUDA, ROCm/HIP toolchain, Vulkan SDK). `FORCE_CMAKE=1`
is vestigial under scikit-build-core and can be dropped.

**Current correct CMake flags (verified against upstream README 2026-07-09):**

| Backend | Flag |
|---|---|
| CUDA | `-DGGML_CUDA=on` |
| Metal | `-DGGML_METAL=on` |
| ROCm/HIP | `-DGGML_HIP=on` (**not** the old `GGML_HIPBLAS`) |
| Vulkan | `-DGGML_VULKAN=on` |
| CPU/OpenBLAS | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` (plain CPU needs no flags) |

**Two gotchas that go in the spec and the docs regardless:**

1. **`uv sync` exact-sync removes out-of-lock packages.** A bare `uv sync` after a
   tier-2b `uv pip install llama-cpp-python` will **uninstall** it. The hybrid avoids
   this for wheel backends (they are in the lock). Where the escape hatch is used, docs
   must say: use `uv sync --inexact`, or re-run the tier-2b install after any sync.
2. **Backend switches poison uv's cache** (keyed on version, not build flags). Any
   backend change must pass `--reinstall-package llama-cpp-python --no-cache`.

**Caveat (documented):** the abetlen index is a single-maintainer GitHub Pages endpoint,
not an SLA. It only ever affects the optional `--extra <backend>`; the core lock is
PyPI-sourced and unaffected. Our own hosts (AL10 = glibc 2.39, RunPod) satisfy the
`manylinux_2_35` wheels.

### 4.3 Backend catalog - single-source-of-truth module (TUI seam)

A small Python data module (e.g., `cogito_backends.py`) holds the curated backend
table as code - one record per backend:

```
name         # human label, e.g. "NVIDIA CUDA 12.4"
key          # uv extra name, e.g. "cu124"  (extra name == detector key)
index_url    # abetlen index for the wheel, or None for source-only
cmake_flag   # source-build flag for the escape hatch
min_glibc    # wheel glibc floor (e.g. "2.35"), or None
kind         # "wheel" | "source"
notes        # short guidance
```

This is **code-as-config in git** (compliant with the runtime-first principle - no
config file read at runtime). It is the single source consumed by: (a) README matrix
generation / docs, and (b) the future TUI's detect-and-recommend logic - so the TUI
never re-hardcodes the list. Extra names are chosen to equal the detector keys the TUI
will produce (`cu124`, `metal`, `cpu`, …), so `uv sync --extra <key>` is a clean handoff.

### 4.4 setup.sh - slim uv bootstrap stub (TUI seam)

Reduce `setup.sh` to a dependency-free (bash-only) bootstrap:

1. Ensure `uv` is present; if missing, install it (the one thing a newcomer cannot
   `uv`-their-way into).
2. `uv sync` the core.
3. Print concise next steps: the per-backend `--extra` options, how to get a model
   (README pointer), and "watch the demo right now, no GPU: `uv run cogito-viz examples/demo_run`".

Deliberately **dumb on hardware** - no backend auto-detection. Structured so the TUI can
later become the thing `setup.sh` launches. Not polished, by intent.

### 4.5 README rewrite

- Fix the `git clone https://github.com/yourusername/cogito.git` placeholder →
  `uistlabs/cogito`.
- Replace the `pip install` quick start with the uv flow (`uv sync` / `uv sync --extra <backend>`).
- Add a **hardware-support matrix** (backend → install command → wheel/source → glibc note),
  generated from / consistent with the backend catalog. Sell the breadth.
- Add a **"Getting a model"** section (generic guidance - where GGUFs live, quant vs size
  tradeoff) replacing the removed interactive download. Kept minimal so the TUI catalog
  supersedes it cleanly.
- Keep the "Running on RunPod" prebuilt-wheel guidance, updated to the current index.

### 4.6 CHANGELOG

Add `[Unreleased]` entries: uv migration, entry points, hybrid backend extras, backend
catalog module, slim setup.sh, removed hardcoded model download, README rewrite.

---

## 5. Testing & Verification Plan

Runnable without a GPU on the dev box (`/mnt/usb-single-1/dev/cogito`):

- `uv sync` succeeds; `uv run cogito --help` works.
- `uv run cogito-viz examples/demo_run` reproduces the demo PNGs (no model/GPU).
- `uv sync --extra cpu` installs a CPU llama-cpp-python wheel; `uv run cogito` loads a
  small GGUF and runs a short cycle count.
- Lockfile round-trips: fresh `uv sync` from a clean checkout reproduces the env.
- `pyproject.toml` extras/conflicts parse; `uv sync --extra cu124` resolves against the
  abetlen index (resolution check; full CUDA run is a RunPod/GPU-host task).
- Backend catalog module imports and enumerates cleanly.

GPU-backed end-to-end (CUDA wheel + real model loop) is a RunPod / dev-box-with-driver
task, tracked separately.

---

## 6. Risks & Open Questions

- **abetlen index availability** - mitigated: optional-extra-only; core lock is PyPI.
- **glibc gate on non-AL10 users** - mitigated: documented, with the source-build hatch.
- **Curated extras list churn** as new CUDA releases land - low; single-line additions,
  and the catalog module centralizes them.
- **Resolved (review):** initial curated extras set confirmed - `cpu`, `cu124`, `cu130`,
  `metal`, `vulkan`, `rocm72` (adjust later if needed).
- **Resolved (review):** `visualize_advanced.py` **promoted** to a `cogito-viz-advanced`
  entry point now (ease-of-use → traction; trivial cost).
