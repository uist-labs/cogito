# COGITO uv Migration - Implementation Plan

Companion to `2026-07-09-uv-migration-design.md`. Execute on branch
`claude/uv-migration` in `/mnt/usb-single-1/dev/cogito` (the fapolicyd-sane zone).
Tasks are ordered; each is independently verifiable. No existing test suite, so
verification is primarily runnable command checks plus one light smoke test.

Verified tooling facts (2026-07-09): llama-cpp-python 0.3.33, `py3-none` wheels at
`https://abetlen.github.io/llama-cpp-python/whl/<backend>`; scikit-build-core backend;
uv extra-index priority means the abetlen wheel wins over the PyPI sdist.

---

## Task 1 - pyproject.toml core + entry points + lockfile

**Goal:** replace venv/requirements with a uv project; add console commands.

**Files:** add `pyproject.toml`; remove `requirements.txt`; generate `uv.lock`;
update `.gitignore` (ignore `.venv/`, keep `uv.lock` tracked).

**Steps:**
- `[project]`: name `cogito`, version (mirror CHANGELOG), `requires-python = ">=3.10"`,
  description, license MIT, readme, authors.
- Core deps: `numpy>=1.24`, `matplotlib>=3.7`.
- Build backend: hatchling with an explicit flat `py-modules` / `force-include` list so
  `cogito`, `visualize`, `visualize_advanced`, `cogito_backends`, `generate_demo_data`
  are packaged. (Confirm hatchling flat-module config during execution.)
- `[project.scripts]`: `cogito = "cogito:main"`, `cogito-viz = "visualize:main"`,
  `cogito-viz-advanced = "visualize_advanced:main"`.
- `rm requirements.txt`.

**Verify:**
- `uv lock` then `uv sync` succeed; `uv.lock` created.
- `uv run cogito --help` exits 0 and prints the arg help.
- `uv run cogito-viz --help` (or usage) and `uv run cogito-viz-advanced` usage work.

**Done when:** clean `uv sync` from a fresh clone reproduces the core env and all three
commands resolve.

---

## Task 2 - backend catalog module (`cogito_backends.py`)

**Goal:** single source of truth for backends, as code (TUI seam; runtime-first compliant).

**Files:** add `cogito_backends.py`.

**Steps:** define a list of records (dataclass or plain dicts), one per curated backend,
fields: `name`, `key` (== uv extra name), `index_url` (or `None`), `cmake_flag`,
`min_glibc` (or `None`), `kind` ("wheel"|"source"), `notes`. Populate the curated set:
`cpu`, `cu124`, `cu130`, `metal`, `vulkan`, `rocm72` with verified index URLs + flags
(CUDA `-DGGML_CUDA=on`, Metal `-DGGML_METAL=on`, ROCm `-DGGML_HIP=on`,
Vulkan `-DGGML_VULKAN=on`, CPU OpenBLAS `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`).
Add a small helper (`by_key()`, `iter_backends()`) for consumers. No import of
`llama_cpp` (stays import-light).

**Verify:** `uv run python -c "import cogito_backends as b; print([x['key'] for x in b.BACKENDS])"`
lists all six; helper lookups work.

**Done when:** module imports with only stdlib and enumerates the curated set.

---

## Task 3 - per-backend uv extras (hybrid tier 2a)

**Goal:** locked, one-command backend installs via `uv sync --extra <key>`.

**Files:** `pyproject.toml`.

**Steps:**
- `[project.optional-dependencies]`: one extra per backend key, each requiring
  `llama-cpp-python` (version floor consistent with the wheels, e.g. `>=0.3.33`).
- `[[tool.uv.index]]` per backend: `name`, `url` (abetlen `whl/<key>`), `explicit = true`.
- `[tool.uv.sources]`: pin `llama-cpp-python` to the right index **per extra** (use uv's
  marker/extra-conditional source form; confirm exact current syntax against uv docs at
  execution time - this is the fiddliest part).
- `[tool.uv]` `conflicts`: declare the backend extras mutually exclusive.

Representative skeleton (finalize syntax during execution):
```
[project.optional-dependencies]
cpu    = ["llama-cpp-python>=0.3.33"]
cu124  = ["llama-cpp-python>=0.3.33"]
# ... metal, vulkan, rocm72, cu130

[[tool.uv.index]]
name = "llama-cpu"
url  = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
explicit = true
# ... one per backend

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu124" }, { extra = "cu130" },
              { extra = "metal" }, { extra = "vulkan" }, { extra = "rocm72" }]]
```

**Verify (on the dev box, AL10 glibc 2.39, no GPU):**
- `uv sync --extra cpu` installs a CPU llama-cpp-python wheel; `uv run python -c
  "import llama_cpp"` works.
- `uv sync --extra cu124` **resolves and downloads** the CUDA wheel (install-only check;
  no GPU run here).
- Conflicts fire: `uv sync --extra cpu --extra cu124` errors as mutually exclusive.

**Done when:** each curated extra resolves against its index; conflicts enforced.

---

## Task 4 - slim `setup.sh` to a uv-bootstrap stub

**Goal:** dependency-free front door; TUI seam.

**Files:** rewrite `setup.sh`.

**Steps:** (1) detect `uv`; if absent, install via the official installer; (2) `uv sync`
the core; (3) print next steps - the `--extra <backend>` options (enumerated from the
catalog's keys, or a static echo kept in sync), how to get a model (README pointer), and
the zero-GPU demo line (`uv run cogito-viz examples/demo_run`). No hardware detection.
No model download.

**Verify:** shellcheck clean; dry run on a box with uv present reaches `uv sync` + prints
steps; the uv-bootstrap branch is exercised (or guarded) without clobbering an existing uv.

**Done when:** `setup.sh` is bash-only, installs/uses uv, syncs core, prints guidance.

---

## Task 5 - README rewrite

**Files:** `README.md`.

**Steps:** fix `yourusername` → `uistlabs`; replace `pip install` quick start with the uv
flow; add a **hardware-support matrix** consistent with `cogito_backends.py` (backend →
`uv sync --extra <key>` → wheel/source → glibc note); add a minimal **"Getting a model"**
section (where GGUFs live, quant vs size); update the RunPod section to the current index;
document the two gotchas where relevant (source-hatch → `uv sync --inexact`; backend
switch → `--reinstall-package llama-cpp-python --no-cache`).

**Verify:** links resolve; matrix matches the catalog; no stale `pip`/placeholder text.

**Done when:** a newcomer can answer "will this run on my machine?" from the README.

---

## Task 6 - CHANGELOG

**Files:** `CHANGELOG.md`.

**Steps:** add `[Unreleased]` entries: uv migration (pyproject + uv.lock), console entry
points, hybrid backend extras, backend catalog module, slim setup.sh, removed hardcoded
model download, README rewrite, ASCII-clean visualizer output.

---

## Task 7 - verification pass + ASCII cleanup

**Steps:**
- ASCII-clean: replace the `Δ` in `visualize_advanced.py` output strings with `delta`
  (per the plain-ASCII-in-shipped-code standard). Grep the repo for any other non-ASCII
  in code/output while here.
- Add a light smoke test (`tests/test_smoke.py` or a script): import `cogito_backends`
  and assert the six keys; assert `cogito --help` exits 0. Keep minimal.
- Full manual check on the dev box: `uv sync` → `uv run cogito-viz examples/demo_run`
  reproduces demo PNGs (no GPU); `uv sync --extra cpu` → `uv run cogito --model
  <tiny.gguf> --cycles 3` completes a short loop.
- GPU-backed CUDA E2E (real model on RTX): tracked separately (RunPod / dev-box-with-driver).

**Done when:** demo + CPU loop verified locally; smoke test passes; no non-ASCII in code.

---

## Sequencing & dependencies

- Task 1 first (foundation). Task 2 independent (do early; Task 3 & 5 consume it).
- Task 3 depends on 1. Task 4 depends on 1. Task 5 depends on 1–4. Tasks 6–7 last.
- Fiddliest/riskiest: **Task 3** (uv per-extra source + conflicts syntax) - verify against
  current uv docs during execution; prove `--extra cpu` end-to-end before wiring all six.

## Execution options

- Solo (Claude) with per-task verification and commits to `claude/uv-migration`.
- Subagent-driven (superpowers:subagent-driven-development), one agent per task with review.
- Fable for Task 3 specifically (the syntax-sensitive uv config), Claude for the rest.

Commit per task to the feature branch (standing OK). Push/PR/merge to main: ask.
