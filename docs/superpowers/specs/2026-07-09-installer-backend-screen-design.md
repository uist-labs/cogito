# COGITO Installer — Shell + Backend Screen — Design Spec

- **Date:** 2026-07-09
- **Status:** Proposed (awaiting review)
- **Branch:** `claude/installer-backend` (off `claude/uv-migration`)
- **Author:** Kenneth Kienle & Claude (Opus 4.8)

---

## 1. Goal & Why

Give COGITO a **guided terminal installer** so a newcomer to AI/ML — the motivating
user is a friend Ken is onboarding, and any GitHub passer-by — goes from a fresh clone
to the correct `llama-cpp-python` backend installed, without reading the hardware-support
matrix or knowing what `cu124` means. Today that step is manual: the user must read the
README, decide which `uv sync --extra <backend>` matches their hardware, and remember the
cache-busting reinstall flags when switching. This spec removes that burden for the
backend step.

This is **session 1 of a three-session roadmap** toward a full "zero-to-running-loop"
front door. It builds the interactive **wizard frame** (the walking skeleton) plus one
fully-working screen — hardware detection, backend recommendation, and install — so the
whole architecture is proven end-to-end before later screens are added.

**The roadmap (each its own spec -> plan -> build cycle):**

| # | Sub-spec | Newcomer outcome |
|---|----------|------------------|
| **1 (this spec)** | Installer shell + backend screen | Right backend detected, recommended, installed |
| 2 | Guided model picker | Curated GGUF suggestions, VRAM-fit checks, download |
| 3 | First-run launcher + publish | Pick genesis + cycles, launch the loop, hand to `cogito-viz`; GitHub topics, README, merge, tag |

The full front door is the destination; this spec walks the first step and leaves clean
seams for steps 2-3.

---

## 2. Scope

**In scope (this spec):**

- A Python **stdlib-only** guided-wizard frame, registered as the `cogito-install`
  console entry point, launched by `setup.sh` after the core sync.
- `cogito_detect.py` — pure, side-effect-free hardware probing returning a structured
  `Detection`. No cogito coupling (designed to be lift-ready for reuse — see 4.6).
- `cogito_install.py` — the wizard: recommendation mapping, confirm/override prompt,
  install execution, re-run/switch-backend handling.
- The **detect -> recommend-with-rationale -> confirm/override** interaction: auto-detect
  the best match, show what was found and why, preselect it as the Enter-default, keep the
  full catalog overridable.
- A **guarded source-build path** for GPU-on-old-glibc (EL9) and unlisted backends:
  offered (not default), pre-flighted for toolchain + SDK, explicitly confirmed, streamed,
  and always falling back to `cpu` on failure.
- Runtime-first flags: `--backend <key>`, `--yes`, `--dry-run`.
- pytest coverage of the recommendation mapping, probe parsers, and wizard choice logic.

**Out of scope (deferred):**

- The **model picker** (curated GGUF catalog, VRAM-fit, download) — session 2.
- The **first-run launcher** (genesis type + cycles, launching the loop) — session 3.
- **Discoverability/publish** (GitHub topics like beautifulyze, README polish, merge to
  main, tag release) — session 3.
- **Auto-installing the build toolchain / GPU SDK** — deliberately never done; it crosses
  the sudo/security boundary and is not portably automatable (see 4.4).
- Any full-screen / curses TUI. "TUI" here means a linear guided wizard, by design.

The engine (`cogito.py` loop) is untouched.

---

## 3. Current State

On `claude/uv-migration` (`1e90c8d`), which is **merged to `origin/main`** (fast-forward;
`origin/main` tip == `1e90c8d`). This local clone's `main` is stale at `06b93bb` (10 behind,
never pulled) — a `git pull` before publish is all that is needed:

- `setup.sh` — thin bash bootstrap: ensure `uv`, `uv sync` the core (numpy/matplotlib),
  print manual next steps (per-backend `--extra` options + model pointer). Deliberately
  dumb on hardware.
- `cogito_backends.py` — the single-source-of-truth catalog: six `Backend` records
  (`cpu`, `cu124`, `cu130`, `metal`, `vulkan`, `rocm72`), each with `key` (== `uv` extra ==
  detector key), `index_url`, `cmake_flag` (source fallback), `min_glibc`, `notes`. Its
  docstring explicitly names "the forthcoming TUI installer's hardware detection and
  recommendation" as its consumer.
- `pyproject.toml` — mutually-exclusive backend extras, each pinned to its abetlen wheel
  index; console scripts `cogito`, `cogito-viz`, `cogito-viz-advanced`.
- README — documents the manual `uv sync --extra <backend>` step, a "switching backends?"
  reinstall-flag dance, and flags a guided model picker as future work.

**What this spec adds:** the detection + recommendation + install logic the catalog was
built to feed, and the `setup.sh -> cogito-install` handoff the slim bootstrap anticipates.

---

## 4. Design

### 4.1 Entry flow & module layout

`setup.sh` stays the thin bash bootstrap; Python does the thinking:

```
./setup.sh
  1. ensure uv present          (unchanged bash bootstrap)
  2. uv sync                    (core env: numpy, matplotlib -- no GPU/model)
  3. uv run cogito-install      (hand off to the Python wizard)   <- new
```

`cogito-install` is also a first-class re-runnable entry point (`uv run cogito-install`
any time) for switching backends later.

Three flat modules (matching the repo's single-module convention):

| File | Responsibility | Depends on |
|------|----------------|------------|
| `cogito_detect.py` | Pure hardware probing -> a structured `Detection` (platform, GPU vendor, CUDA/ROCm availability + version, glibc). **No install side-effects, no user I/O.** | stdlib only |
| `cogito_install.py` | The wizard: `Detection` -> recommended `Backend`, render confirm/override prompt, run the install, handle re-run/switch. The `cogito-install` console script. | `cogito_detect`, `cogito_backends`, stdlib |
| `cogito_backends.py` | *(existing)* the catalog — unchanged, consumed as intended | — |

The seam that makes this testable: **detection is pure**, so it is unit-tested by feeding
synthetic probe outputs and asserting the recommendation; the **wizard owns all the
interactive and side-effecting parts**, injected so tests never touch the real environment.

### 4.2 Detection & recommendation

`cogito_detect.py` runs a fixed sequence of cheap, read-only probes and returns one
`Detection`. Each probe degrades to "unknown" rather than raising, so a missing tool never
crashes the wizard. Each probe is split into **run-command -> text** (thin, mockable) and
**parse-text -> value** (pure, fully tested).

| Signal | Probe (stdlib only) | Feeds |
|--------|---------------------|-------|
| Platform / arch | `platform.system()`, `platform.machine()` | Apple Silicon -> `metal` |
| NVIDIA present + driver max-CUDA | `shutil.which("nvidia-smi")`, parse driver + `CUDA Version` | NVIDIA path; CUDA-13-runtime gate |
| NVIDIA **compute capability** | `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` | **`cu124` vs `cu130` (the real discriminator)** |
| AMD GPU | `shutil.which("rocminfo")` / `/opt/rocm` / `lspci` VGA scan for AMD/ATI | `rocm72` vs `vulkan` |
| ROCm stack installed | `rocminfo` returns agents (not just GPU present) | gates `rocm72` |
| Vulkan runtime | `shutil.which("vulkaninfo")` | `vulkan` |
| glibc version | `platform.libc_ver()`, fallback `os.confstr("CS_GNU_LIBC_VERSION")` | wheel-viability gate |

**`recommend(detection) -> (backend_key, rationale, caveats)`** is a pure function.
First match wins, conservative on uncertainty:

1. Darwin + arm64 -> `metal`.
2. NVIDIA present + glibc >= 2.35 -> the `cu124`-vs-`cu130` choice keys off the GPU's
   **compute capability**, not the driver's max-CUDA version (the driver ceiling is a
   runtime ceiling, not a guarantee the wheel's build targets this GPU's architecture):
   - **CC < 7.5** (Maxwell/Pascal/Volta) -> **`cu124`**, always. CUDA 13 removed
     offline-compile + library support for these arches, so a `cu130` wheel contains no
     code for them and cannot JIT down (PTX is forward-compatible only). *(This is the
     GTX 1070 / Pascal CC 6.1 case — the wks1 dev box.)*
   - **CC >= 7.5** (Turing+) -> `cu130` if the driver supports the CUDA 13 runtime, else
     `cu124`.
   - **compute_cap unreadable** (ancient driver lacking the field) -> conservative `cu124`.
3. NVIDIA present but glibc < 2.35 -> **`cpu`**, with a visible caveat: the CUDA wheel
   needs glibc >= 2.35; the GPU works via a source build; here is the path (see 4.4).
4. AMD present + ROCm stack detected + glibc >= 2.35 -> `rocm72`.
5. AMD/Intel GPU present, no usable ROCm -> `vulkan` (if a Vulkan driver is present),
   else `cpu`.
6. Anything else / all-unknown -> `cpu`.

Every recommendation carries a short human rationale string the wizard prints verbatim
("Found an NVIDIA GPU; driver supports CUDA 12.4; glibc 2.35 meets the 2.35 floor —
recommending cu124").

### 4.3 Wizard UX flow

Linear, ASCII-only (per the no-Unicode-in-shipped-code rule), one decision at a time:

```
COGITO backend installer
-------------------------
Detecting hardware...
  Platform : Linux x86_64
  GPU      : NVIDIA (driver 550.x, supports CUDA 12.4)
  glibc    : 2.35

Recommended backend: cu124  (NVIDIA CUDA 12.4)
  Why: found an NVIDIA GPU; driver supports CUDA 12.4; glibc 2.35 meets the
       2.35 floor for the prebuilt wheel.

  [1] cu124   NVIDIA CUDA 12.4        <- recommended
  [2] cu130   NVIDIA CUDA 13.0
  [3] vulkan  Cross-vendor GPU
  [4] cpu     Runs anywhere (slowest)
  [5] more... (metal, rocm72)

Press Enter to accept cu124, or choose a number:
```

Flow rules:

- **Enter = accept the recommendation.** A newcomer types nothing and gets the right thing.
- The full catalog is always reachable (recommended floated to top; `more...` reveals the
  rest), so an informed user can override — including deliberately choosing a backend that
  triggers the source-build path.
- After a choice, a one-line **confirmation of the actual command** before it runs:
  `Running: uv sync --extra cu124` (or the reinstall variant when switching). Consequential
  action, shown before it happens.
- **Progress is streamed, not hidden** — `uv sync` output flows through so a long wheel
  download visibly heartbeats.
- On success, the screen ends by **pointing forward** — the same "now get a model, then
  run" next-steps `setup.sh` prints today. Sessions 2-3 replace those printed next-steps
  with real screens.

**Re-run / switch-backend:** if a backend extra is already installed, `cogito-install`
says so and offers to switch, adding `--reinstall-package llama-cpp-python --no-cache`
automatically (the README's manual "switching backends?" dance disappears).

### 4.4 Guarded source-build path

When the recommendation lands in a source-build situation (a real GPU but glibc < 2.35,
e.g. an EL9 box) or the user overrides to a backend needing a source build, the wizard
offers **two** options: the safe `cpu` default, and "build the GPU backend from source."
If the build is chosen:

1. **Pre-flight the toolchain** — check for a C/C++ compiler, `cmake`, `make`/`ninja`, and
   the backend SDK keyed off the chosen backend (`nvcc` for CUDA, `hipcc` for ROCm, Vulkan
   headers), using the catalog's `cmake_flag`.
2. **Missing prereqs -> stop clean.** Do not start a doomed build. Print exactly what is
   absent **and the precise, copy-pasteable install command for the detected platform**
   (e.g. `sudo dnf install gcc-c++ cmake ninja-build` on EL, `brew install cmake ninja` on
   macOS); for the CUDA/ROCm SDK specifically, link the vendor's official installer rather
   than guessing. Then fall back to offering `cpu`. **The privileged install stays a
   deliberate human action** — the wizard never runs `sudo` or adds vendor repos itself
   (crosses the security boundary; not portably automatable; out of scope).
3. **Prereqs present -> confirm explicitly** ("This compiles llama-cpp-python locally,
   takes several minutes, needs network + the CUDA toolkit — proceed?"), then run
   `CMAKE_ARGS="<cmake_flag>" uv pip install llama-cpp-python --reinstall-package
   llama-cpp-python --no-cache` with **live streamed output**.
4. **Success -> `uv sync --inexact`** so uv does not clobber the hand-built wheel (the
   README's documented gotcha, done for them).
5. **Failure -> diagnose and fall back.** Print the tail of the build log with a
   plain-language cause, then drop to the `cpu` recommendation so the user is never
   dead-ended.

Net: the fast path is automated when safe, the build is available when wanted, and every
failure mode lands somewhere that works.

### 4.5 Runtime-first flags

Per the no-on-disk-runtime-config rule, behavior is controlled by flags (which also make
the wizard scriptable and testable):

- `cogito-install --backend <key>` — skip detection, install a named backend.
- `--yes` — accept the recommendation non-interactively (scriptable installs; CI).
- `--dry-run` — print the exact command(s) and exit without touching the environment.

### 4.6 Reuse note (future, not built now)

`cogito_detect.py` — "what GPU is this, which CUDA/ROCm version, what glibc" — is generic
infrastructure, not cogito-specific. It is a **candidate shared module** for other UIST
GPU-backend projects (the poker trainer's RunPod image, MythCast, future llama-cpp
consumers). It is deliberately built **lift-ready** (pure, stdlib-only, no cogito imports)
but **not extracted** — a shared package before a second real consumer is premature
abstraction. **Trigger to revisit:** the second GPU-backend project that reaches for it.

---

## 5. Testing & Verification Plan

Three layers, matching the pure/injectable seams:

1. **Recommendation mapping (crown jewel).** Table-driven pytest over
   `recommend(detection)` feeding synthetic `Detection` records; assert backend key +
   caveat flags for every branch: metal, cu124, cu130, NVIDIA-on-old-glibc -> cpu+source
   note, AMD+ROCm -> rocm72, AMD-without-ROCm -> vulkan, all-unknown -> cpu, and the
   compute-capability cu124-vs-cu130 split. **Regression fixture (locked):** a
   Pascal/CC-6.1 detection with driver max-CUDA 13.0 (the real wks1 GTX 1070) must
   recommend **`cu124`**, not `cu130` — this is the case the original driver-version rule
   got wrong. Paired with a Turing+/CC-7.5+ fixture that correctly yields `cu130`. No
   hardware needed.
2. **Probe parsing.** Fixture-driven tests feed captured `nvidia-smi` / glibc / `platform`
   outputs to the pure parsers; presence/absence of tools simulated by mocking
   `shutil.which` + `subprocess`, so the full detector runs green on the dev box without
   the other vendors present.
3. **Wizard flow.** Choice parsing (Enter -> recommended, number -> that backend, invalid
   -> re-prompt) driven via stdin. The `uv sync` call is an **injected runner**; tests
   assert "would run: `uv sync --extra cu124`" without executing a real install. `--dry-run`
   exercises the same path end-to-end.

**Coverage (trust-but-verify):** the mapping and parsers are fully unit-tested via
fixtures. **Live NVIDIA validation is now possible** — the wks1 dev box's GTX 1070 driver
is installed (580.173.02, driver max-CUDA 13.0, compute_cap 6.1), so this session can
exercise the real `cu124` path end-to-end: detect -> recommend `cu124` (the Pascal case)
-> `uv sync --extra cu124` -> load a small GGUF with GPU offload and confirm the kernels
run on the 1070. That live run is the ground-truth check on the compute-capability rule.
Metal, ROCm, CUDA-13-capable (Turing+) NVIDIA, and old-glibc source-build paths still have
no local hardware and remain fixture + `--dry-run` only; the spec claims no live coverage
it does not have for those.

---

## 6. Risks & Open Questions

- **Detection false-positives** (e.g. GPU present but driver/runtime broken) — mitigated by
  the confirm/override prompt (the recommendation is never silently acted on) and the
  conservative-on-uncertainty mapping.
- **cu124-vs-cu130 misclassification** — the discriminator is the GPU's **compute
  capability**, not driver max-CUDA (CUDA 13 dropped pre-Turing arches; the driver ceiling
  is a runtime ceiling, not a build-target guarantee). CC < 7.5 -> `cu124`; unreadable
  compute_cap -> conservative `cu124`. Locked by the Pascal/CC-6.1 regression fixture.
  *Residual open question:* whether the abetlen `cu124` wheel ships Pascal (sm_61) SASS or
  only newer-arch PTX — resolved empirically by the live 1070 offload run (see §5).
- **Source build failing on a newcomer's box** — mitigated by pre-flight, explicit
  confirmation, streamed output, and always-fall-back-to-`cpu`.
- **Live validation now unblocked** — the dev box's 1070 driver is installed, so the
  `cu124`/Pascal path gets a real end-to-end check this session (see §5). Non-NVIDIA and
  Turing+ paths remain fixture-only.
- **Local clone's `main` is stale** (`06b93bb`, 10 behind). The uv spine **is** merged on
  `origin/main` (@ `1e90c8d`); this branch is based on `1e90c8d` (== `origin/main` tip), so it
  stacks correctly on merged content. A `git pull` on local `main` before publish is all that
  is needed — no reconciliation.
