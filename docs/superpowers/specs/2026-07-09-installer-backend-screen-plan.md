# COGITO Installer — Shell + Backend Screen — Implementation Plan

Companion to `2026-07-09-installer-backend-screen-design.md`. Execute on branch
`claude/installer-backend` in `/mnt/usb-single-1/dev/cogito` (the fapolicyd-sane dev zone;
home is noexec). Tasks are ordered; each is independently verifiable. The pure logic
(recommendation mapping, probe parsers) is written **test-first** — it is the crown jewel
and needs no hardware. Side effects (subprocess, `uv sync`, stdin) are injected/mocked so
the suite runs green on the dev box without touching the environment.

Verified facts (2026-07-09):
- CUDA 13.0 removed offline-compile + library support for Maxwell/Pascal/Volta (compute
  capability < 7.5). Source: NVIDIA CUDA 13.0 release notes; Tom's Hardware. => the
  `cu124`-vs-`cu130` split keys off **GPU compute capability**, not driver max-CUDA.
- wks1 dev box: GTX 1070 (Pascal, CC 6.1), driver 580.173.02, `nvidia-smi` reports
  `CUDA Version: 13.0`. This is the live Pascal test rig and the regression counterexample.
- `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` exposes CC directly (present
  on driver 580; absent on very old drivers -> fall back to conservative `cu124`).
- Existing `cogito_backends.py` catalog is unchanged and consumed as-is (six keys, each
  carrying `cmake_flag`, `min_glibc`).

---

## Task 1 — `cogito_detect.py`: `Detection` type + pure probe parsers (test-first)

**Goal:** the pure `parse-text -> value` layer and the immutable `Detection` record. No
subprocess, no user I/O.

**Files:** add `cogito_detect.py`; add `tests/test_detect_parsers.py`; add
`tests/fixtures/` captured outputs.

**Steps:**
- `@dataclass(frozen=True) Detection`: `system`, `machine`, `nvidia` (bool),
  `nvidia_driver` (str|None), `nvidia_max_cuda` (tuple|None), `nvidia_compute_cap`
  (float|None), `amd_gpu` (bool), `rocm_ok` (bool), `vulkan` (bool), `glibc`
  (tuple|None). Every field defaults to the "unknown"/absent value.
- Pure parsers, each taking captured text and returning a value (never raising):
  `parse_nvidia_smi_cuda(text) -> (13, 0)`, `parse_compute_cap(text) -> 6.1`,
  `parse_driver_version(text)`, `parse_glibc(ver_tuple)`, `parse_lspci_amd(text) -> bool`,
  `parse_rocminfo_agents(text) -> bool`.
- **Write the tests first**, driven by fixtures: capture real `nvidia-smi` output from the
  1070 into `tests/fixtures/nvidia_smi_gtx1070.txt` and
  `tests/fixtures/compute_cap_6_1.txt`; add synthetic fixtures for a Turing+ card
  (`compute_cap_8_6.txt`), an AMD `lspci` line, a `rocminfo`-with-agents blob, and a
  missing-field case.

**Verify:** `uv run pytest tests/test_detect_parsers.py -q` green; parsers return the
"unknown" sentinel (not an exception) on empty/garbage input.

**Done when:** parsers + `Detection` import with stdlib only and pass their fixture tests.

---

## Task 2 — `cogito_detect.py`: probe runners + `detect()` orchestrator

**Goal:** the thin `run-command -> text` layer and `detect() -> Detection`.

**Files:** `cogito_detect.py`; add `tests/test_detect_orchestrator.py`.

**Steps:**
- Probe runners using `shutil.which` + `subprocess.run` (short timeout, text mode,
  swallow `FileNotFoundError`/nonzero into ""). One per tool: `nvidia-smi`, `rocminfo`,
  `vulkaninfo`, `lspci`. `glibc` via `platform.libc_ver()` with
  `os.confstr("CS_GNU_LIBC_VERSION")` fallback.
- `detect()` assembles a `Detection` by running probes + feeding their text to the Task-1
  parsers. Pure-ish orchestration; the only impurity is the runners, which are injectable
  (default real, override in tests).
- Tests mock `shutil.which`/`subprocess` to simulate: NVIDIA-only (the 1070), AMD+ROCm,
  AMD-no-ROCm, Apple Silicon (`platform` monkeypatched), and bare CPU.

**Verify:** `uv run pytest tests/test_detect_orchestrator.py -q` green. Live smoke on the
dev box: `uv run python -c "import cogito_detect as d; print(d.detect())"` shows
`nvidia=True, nvidia_compute_cap=6.1, nvidia_max_cuda=(13,0)`.

**Done when:** `detect()` returns a correct `Detection` on the real box and under all mocks.

---

## Task 3 — `recommend()` mapping + Pascal regression fixture (test-first, crown jewel)

**Goal:** the pure `recommend(detection) -> (backend_key, rationale, caveats)` decision.

**Files:** `cogito_detect.py` (or a sibling `cogito_recommend.py` — keep in `cogito_detect`
for now, one import); add `tests/test_recommend.py`.

**Steps (tests first):** encode the §4.2 ordered rules. Table-driven cases asserting
`(key, caveat-flags)` for every branch:
- Darwin+arm64 -> `metal`.
- **NVIDIA CC 6.1 (Pascal) + driver max-CUDA 13.0 + glibc 2.39 -> `cu124`** — the locked
  regression case (the wks1 1070; the rule the driver-version approach got wrong).
- NVIDIA CC 8.6 (Turing+) + driver CUDA 13 + glibc 2.35 -> `cu130`.
- NVIDIA CC 8.6 + driver max-CUDA 12.x -> `cu124`.
- NVIDIA present, `compute_cap` unreadable -> `cu124` (conservative).
- NVIDIA + glibc < 2.35 -> `cpu` + source-build caveat.
- AMD + ROCm + glibc >= 2.35 -> `rocm72`; AMD no-ROCm + Vulkan -> `vulkan`; AMD no-ROCm
  no-Vulkan -> `cpu`.
- all-unknown -> `cpu`.
Rationale strings are asserted to mention the deciding signal (e.g. "compute capability
6.1").

**Verify:** `uv run pytest tests/test_recommend.py -q` green, Pascal case included.

**Done when:** every §4.2 branch is covered and the Pascal regression is locked.

---

## Task 4 — `cogito_install.py`: wizard core (render, choice, injected runner, flags)

**Goal:** the interactive screen + the `cogito-install` entry point, with all side effects
injected.

**Files:** add `cogito_install.py`; register `cogito-install = "cogito_install:main"` in
`pyproject.toml` `[project.scripts]`; add `tests/test_wizard.py`.

**Steps:**
- `main(argv)` with argparse flags: `--backend <key>`, `--yes`, `--dry-run` (runtime-first;
  no config file).
- Render: detected-hardware summary + recommended backend + verbatim rationale + the menu
  (recommended floated to top, `more...` reveals the rest), all ASCII-only.
- Choice parse: Enter -> recommended; number -> that backend; invalid -> re-prompt.
- **Injected runner** `run_sync(cmd, *, dry_run)` — the single side-effect seam. Prints
  `Running: <cmd>` before executing; streams output (no capture). `--dry-run` prints and
  returns without executing.
- Switch-backend: if a backend extra is already installed, note it and append
  `--reinstall-package llama-cpp-python --no-cache` to the sync command.
- Forward-pointing exit: reuse the "now get a model, then run" next-steps text (single
  source shared with `setup.sh` so they can't drift).
- Tests: drive stdin for Enter/number/invalid; assert the runner is called with the exact
  `uv sync --extra <key>` (or reinstall variant); assert `--dry-run` executes nothing;
  assert `--backend cpu --yes` is fully non-interactive.

**Verify:** `uv run pytest tests/test_wizard.py -q` green; `uv run cogito-install --dry-run`
on the dev box prints the detected 1070, recommends `cu124`, and shows
`Running: uv sync --extra cu124` without installing.

**Done when:** the wizard renders, parses every choice, and never touches the env under
`--dry-run`/tests.

---

## Task 5 — guarded source-build path

**Goal:** offer-first, pre-flighted, confirmed source build for GPU-on-old-glibc / unlisted
backends, always falling back to `cpu`.

**Files:** `cogito_install.py`; `tests/test_source_build.py`.

**Steps:**
- Triggered when the recommendation resolves to a source-build situation (real GPU +
  glibc < 2.35) or the user overrides to such a backend. Present two options: safe `cpu`
  (default) and "build from source".
- **Pre-flight** the toolchain + SDK keyed off the chosen backend's `cmake_flag`: `cc`/`c++`,
  `cmake`, `make`/`ninja`, and `nvcc` (CUDA) / `hipcc` (ROCm) / Vulkan headers.
- Missing prereqs -> **stop clean**: print exactly what is absent + the copy-paste install
  command for the detected platform (`sudo dnf install ...` on EL, `brew install ...` on
  macOS) and, for CUDA/ROCm SDKs, link the vendor installer; then fall back to `cpu`.
  **Never** run `sudo` or add repos.
- Prereqs present -> explicit confirm, then run
  `CMAKE_ARGS="<cmake_flag>" uv pip install llama-cpp-python --reinstall-package
  llama-cpp-python --no-cache` via the injected runner (streamed). Success ->
  `uv sync --inexact`. Failure -> print the log tail + plain-language cause -> fall back
  to `cpu`.
- Tests mock the pre-flight probes (all-present / nvcc-missing) and the runner: assert
  the missing-nvcc path prints the guidance + platform command and falls back to `cpu`;
  assert the success path issues the `CMAKE_ARGS=... uv pip install` then
  `uv sync --inexact`; assert build-failure falls back to `cpu`.

**Verify:** `uv run pytest tests/test_source_build.py -q` green. (No live old-glibc box —
this path is fixture + `--dry-run` only, stated honestly in the design §5.)

**Done when:** every source-build branch is covered and each dead-ends into a working
fallback.

---

## Task 6 — `setup.sh` handoff + CHANGELOG

**Goal:** wire the wizard into the front door; record the change.

**Files:** `setup.sh`; `CHANGELOG.md`.

**Steps:**
- `setup.sh` step 3: after `uv sync`, `exec uv run cogito-install` (replacing the printed
  manual `--extra` guidance, which the wizard now owns). Keep the "watch the demo now"
  zero-GPU line. Bash stays dependency-free.
- CHANGELOG `[Unreleased]`: guided backend installer (`cogito-install`), compute-capability
  backend detection, guarded source-build fallback, `setup.sh` launches the wizard.

**Verify:** `shellcheck setup.sh` clean; a dry run reaches `uv run cogito-install`.

**Done when:** `./setup.sh` on a fresh clone lands the user in the wizard.

---

## Task 7 — live hardware validation (the ground-truth check) + full suite

**Goal:** prove the compute-capability rule on real Pascal silicon — the thing the driver
install just unblocked.

**Files:** none (validation); note results in the CHANGELOG / a short log.

**Steps:**
- Full suite green: `uv run pytest -q` (parsers, orchestrator, recommend, wizard,
  source-build) + the existing smoke tests.
- **Live 1070 run:** `uv run cogito-install --yes` -> confirm it detects CC 6.1, recommends
  and installs `cu124` (`uv sync --extra cu124`), `uv run python -c "import llama_cpp"`
  loads. Then load a **small GGUF** (supply a ~1-3B Q4 that fits 8 GB) with
  `n_gpu_layers > 0` and confirm from llama.cpp's own log that layers are offloaded to the
  GPU and a short generation runs on the 1070.
- This resolves the design's residual open question: whether the abetlen `cu124` wheel
  actually ships Pascal (sm_61) code or only newer PTX. Record the answer.

**Verify:** GPU offload visible in the llama.cpp load log on the 1070; full pytest green.

**Done when:** the Pascal/`cu124` path is proven end-to-end on real hardware, or — if the
wheel turns out not to support Pascal — the finding is recorded and the recommendation
adjusted (e.g. steer Pascal to `cpu` or a source build) with a follow-up.

---

## Sequencing & dependencies

- Task 1 first (types + parsers). Task 2 depends on 1. Task 3 depends on 1's `Detection`.
- Task 4 depends on 2 + 3. Task 5 extends 4. Task 6 depends on 4. Task 7 depends on all.
- Riskiest: **Task 3** (the rule must be exactly right — Task 1's Pascal fixture is the
  guard) and **Task 7** (the live wheel/Pascal unknown; has a defined fallback if it fails).

## Execution options

- **Solo (Claude), test-first, per-task commits** to `claude/installer-backend` — the
  default. Pure cores (Tasks 1/3) get tests before code.
- Subagent-driven (superpowers:subagent-driven-development), one agent per task with review
  — reasonable given the clean task seams.
- Fable for none of these specifically; the logic is well-scoped for Opus.

Commit per task to the feature branch (standing OK). Push / PR / merge to `main`, and the
pre-publish `git pull` on the stale local `main`: ask.
