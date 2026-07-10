# COGITO Guided Model Picker (S2) -- Implementation Plan

Companion to `2026-07-10-installer-model-picker-design.md`. Execute on branch
`claude/installer-model-picker` in `/mnt/usb-single-1/dev/cogito` (the
fapolicyd-sane dev zone; home is noexec). Tasks are ordered; each is
independently verifiable. Pure logic (catalog helpers, memory parsers, the
fit decision, the download engine's pure parts) is written **test-first** and
needs no hardware or network. Side effects (subprocess, urllib, filesystem,
stdin) are injected/mocked so the suite runs green on the dev box offline.

Mirrors S1's conventions: stdlib `unittest` (not pytest -- match the repo's
`tests/` convention, superseding the S1 plan's "pytest" wording), one injected
seam per side effect, per-task commit to the feature branch.

Verified/assumed facts:
- S1 shipped `cogito_detect.py` (`Detection` + parsers + `detect()` +
  `recommend()`), `cogito_backends.py`, `cogito_install.py`; merged @ de12897.
- wks1 dev box: GTX 1070 (8 GB VRAM, Pascal CC 6.1), driver 580.173.02, glibc
  2.39; `cu124` backend proven E2E in S1. This is the live download + partial/
  full-offload test rig.
- `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` reports
  total VRAM in MiB. `/proc/meminfo` `MemTotal` gives system RAM (kB).
- Hugging Face serves GGUFs at `{repo}/resolve/main/{file}`; a `HEAD` returns
  `Content-Length` (used to pin `size_bytes`) and supports `Range` (resume).

---

## Task 1 -- `cogito_models.py`: curated catalog (test-first helpers)

**Goal:** the code-as-config model catalog + its accessors and URL derivation,
mirroring `cogito_backends.py`.

**Files:** add `cogito_models.py`; add `tests/test_models.py`.

**Steps:**
- `@dataclass(frozen=True) Model`: `key`, `name`, `params_b` (float),
  `hf_repo`, `hf_file`, `quant`, `size_bytes` (int), `n_layers` (int, the
  static block count -- feeds the partial-offload estimate), `sha256`
  (Optional[str], None until pinned), `notes`.
- Accessors: `iter_models()`, `by_key(key)`, `keys()` (catalog order).
- `download_url(model) -> str` = `https://huggingface.co/{hf_repo}/resolve/main/{hf_file}`.
- Populate the 8 curated entries (design's table). **Author the repo/file from
  research** (bartowski or official-Qwen GGUF, all ungated); leave `size_bytes`
  and `sha256` as provisional -- pinned in this task's verify step.
- **Tests first:** `by_key`/`keys` round-trip; `download_url` derivation; every
  entry has a non-empty ungated repo/file and a positive `params_b`/`n_layers`;
  keys are unique.

**Verify:** `.venv/bin/python -m unittest tests.test_models -v` green. Then pin
metadata live: `HEAD` each `download_url` (a tiny stdlib script), record
`Content-Length` into `size_bytes`, confirm HTTP 200 (not a gated 401/403).
sha256 is captured opportunistically in Task 7 for models actually downloaded.

**Done when:** the catalog imports stdlib-only, helpers pass, and every entry's
URL resolves ungated with a real `size_bytes`.

---

## Task 2 -- `cogito_detect.py`: memory detection (test-first parsers)

**Goal:** add total VRAM and total RAM to `Detection` without disturbing S1.

**Files:** `cogito_detect.py`; add `tests/test_detect_memory.py`;
`tests/fixtures/` additions.

**Steps:**
- Extend `Detection` with `vram_total_mb: Optional[int]` and
  `ram_total_mb: Optional[int]` (default `None`; additive -- S1 fields and tests
  untouched).
- Pure parsers (never raise): `parse_vram(text) -> Optional[int]` (first line of
  the memory.total CSV, MiB); `parse_meminfo(text) -> Optional[int]` (`MemTotal:
  N kB` -> MB).
- Extend `detect()`: when `nvidia`, run
  `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` via the
  existing injected `run` seam -> `parse_vram`. RAM via reading `/proc/meminfo`
  (injected reader, default real) with an `os.sysconf('SC_PAGE_SIZE') *
  os.sysconf('SC_PHYS_PAGES')` fallback off-Linux. Stdlib only; no `psutil`.
  Module stays cogito-agnostic / lift-ready.
- **Tests first:** fixtures for the 1070's `memory.total` (8192) and a
  `/proc/meminfo` blob; assert parsers + that `detect()` under the S1 mocks now
  also populates the two fields (and leaves them `None` when the tool is absent).

**Verify:** `.venv/bin/python -m unittest tests.test_detect_memory -v` green;
S1's detect tests still green. Live: `... -c "import cogito_detect as d;
print(d.detect())"` shows `vram_total_mb=8192` (approx) and a real
`ram_total_mb` on the box.

**Done when:** memory fields populate on the real box and under mocks; S1
detection is unchanged.

---

## Task 3 -- fit decision + ladder step-down (test-first, crown jewel)

**Goal:** the pure `fit(model, detection) -> Fit` classifier and the
"next model down that fits" helper. No I/O.

**Files:** `cogito_models.py` (or sibling -- keep in `cogito_models` for one
import); add `tests/test_fit.py`.

**Steps (tests first):**
- `@dataclass(frozen=True) Fit`: `tier` (`"gpu"|"partial"|"cpu"|"cpu_oversized"`),
  `gpu_layers` (int for the run command: `-1` full, `0` cpu, else the partial
  estimate), `reason` (str the wizard prints).
- Rules, with a conservative `HEADROOM` multiplicative factor (KV-cache/context
  overhead; a module constant, tuned in this task and documented):
  - VRAM present and `size_bytes * HEADROOM <= vram_bytes` -> `gpu`,
    `gpu_layers = -1`.
  - VRAM present but smaller -> `partial`; estimate
    `gpu_layers = floor(n_layers * usable_vram / (size_bytes * HEADROOM))`,
    clamped to `[1, n_layers-1]` (a starting point; the suggestion says "adjust
    if you hit OOM").
  - No usable VRAM, `size_bytes * HEADROOM <= ram_bytes` -> `cpu`,
    `gpu_layers = 0`.
  - Exceeds RAM too -> `cpu_oversized` (still allowed; warned).
  - Memory unknown -> conservative `cpu`.
- `recommended_model(catalog, detection) -> Model`: the largest entry tagged
  `gpu` (or, with no GPU, the largest `cpu` that is not `cpu_oversized`).
- `step_down(catalog, detection, chosen) -> Optional[Model]`: the next smaller
  entry (by `params_b`) whose fit is at least `partial`.
- **Table-driven tests:** 8 GB card (7B -> gpu, 14B -> partial, 32B -> partial/
  oversized), 24 GB card (32B -> gpu), no-GPU/32 GB RAM (7B -> cpu, 32B ->
  cpu/oversized by RAM), unknown memory -> cpu. Assert `gpu_layers` and that
  step-down returns a strictly smaller fitting model (or None).

**Verify:** `.venv/bin/python -m unittest tests.test_fit -v` green.

**Done when:** every tier + the step-down are covered; `gpu_layers` values are
correct and clamped.

---

## Task 4 -- `cogito_modeldl.py`: resumable download engine (test-first)

**Goal:** the stdlib downloader in isolation -- resume, progress, token, verify.

**Files:** add `cogito_modeldl.py`; add `tests/test_modeldl.py`.

**Steps:**
- `download(url, dest, *, size_bytes, sha256=None, opener=..., env=os.environ,
  out=..., token=None) -> Path`. Single injected side-effect seam: an
  `opener(request) -> response` factory (default `urllib.request.urlopen`) and
  the destination path.
- Stream to `dest + ".part"`; if a `.part` exists, send `Range: bytes=<n>-` and
  append (resume). Progress: bytes/total + a simple ASCII bar to `out`,
  throttled.
- `HF_TOKEN`/`HF_HUB_TOKEN` (or explicit `token=`) -> `Authorization: Bearer`
  header; absent -> no header.
- On completion: verify final size == `size_bytes` (mismatch -> raise a clean
  `DownloadError` naming got-vs-expected, leave `.part` for inspection); if
  `sha256` given, verify it. Success -> atomic `os.replace(".part", dest)`.
- Retries: bounded attempts with backoff on transient `URLError`/timeout,
  resuming from the `.part` each time; exhausted -> `DownloadError` with the
  direct URL + `wget`/`curl -C -` fallback text.
- Skip: if `dest` already exists with the right size, return it immediately.
- **Tests first (no network):** a fake opener serving bytes from an in-memory
  buffer with `Range` support -> assert full download, resume from a partial
  `.part` (Range header asserted), size-mismatch rejection, sha256 pass/fail,
  atomic rename, token header present only when env/arg set, already-present
  skip, and retry-then-give-up.

**Verify:** `.venv/bin/python -m unittest tests.test_modeldl -v` green, offline.

**Done when:** every download branch is covered against the fake opener.

---

## Task 5 -- `cogito_modelpick.py`: the `cogito-model` wizard (test-first flow)

**Goal:** the interactive picker + entry point, all side effects injected.

**Files:** add `cogito_modelpick.py`; register
`cogito-model = "cogito_modelpick:main"` in `pyproject.toml` `[project.scripts]`;
add `tests/test_modelpick.py`.

**Steps:**
- `main(argv, *, detect_fn, catalog, input_fn, downloader, which, out, isatty)`
  with argparse flags: `--model KEY`, `--dest PATH`, `--yes`, `--dry-run`.
- Flow: `detect_fn()` -> annotate each catalog entry with `fit()` -> render the
  fit-tagged menu (recommended floated to top, ASCII-only) -> choose (Enter =
  recommended; number = that model) -> if the choice does not fit, offer
  `step_down` -> resolve storage path (default `./models`, prompt; offer to
  create a missing dir; error on unwritable) -> free-space precheck
  (`shutil.disk_usage(path).free` vs `size_bytes * HEADROOM`; fail clean with
  got-vs-need) -> `downloader(...)` -> print the ready-to-paste run command
  (`--gpu-layers` from the fit; partial notes "adjust if OOM").
- Semantics (design's error table): `--model KEY` downloads it (tty or not);
  `--yes` downloads the recommended model; implicit non-interactive (no tty, no
  `--model`, no `--yes`) prints the fit-annotated catalog and exits without
  downloading. `--dry-run` prints the resolved plan (chosen model, dest, URL,
  run command) and downloads nothing. Backend-not-installed -> warn, point at
  `cogito-install`, still allow.
- **Tests first:** inject a fake `detect_fn` (8 GB / 24 GB / no-GPU), a fake
  `downloader` recording its call, and scripted stdin -> assert recommended
  selection, number override, doesn't-fit -> step-down, missing-dir create
  path, precheck-fail aborts before calling the downloader, each flag, implicit
  non-interactive lists-and-exits, and the emitted run command / `--gpu-layers`.

**Verify:** `.venv/bin/python -m unittest tests.test_modelpick -v` green;
`uv run cogito-model --dry-run` on the box detects the 1070, recommends a
fitting model, prints its URL + a `cu124` run command, downloads nothing.

**Done when:** the wizard drives every branch offline and touches the disk/
network only through the injected downloader.

---

## Task 6 -- installer handoff + setup.sh + CHANGELOG

**Goal:** chain `cogito-install` into `cogito-model`; record the change.

**Files:** `cogito_install.py` (`_next_steps`); `setup.sh`; `CHANGELOG.md`.

**Steps:**
- `cogito_install.py` `_next_steps`: on a successful, verified backend install,
  replace the "Get a GGUF model (see README)" text with a launch of
  `cogito-model` (`uv run cogito-model`) -- and keep the re-runnable note and
  the zero-GPU demo line. Guard so a `--dry-run`/failed install does not chain.
- `setup.sh`: the comment/flow already `exec`s the wizard; update the trailing
  guidance to mention the model step now being automatic.
- README: replace the "guided model picker ... planned for a future release"
  note with a short "the installer now offers it" line (full README polish is
  S3).
- CHANGELOG `[Unreleased]`: guided model picker (`cogito-model`), curated GGUF
  catalog, VRAM/RAM fit, resumable stdlib download, installer chains into it.

**Verify:** `.venv/bin/python -m unittest discover tests -v` all green;
`shellcheck setup.sh` clean; an install `_next_steps` (mocked) points at
`cogito-model`.

**Done when:** a fresh `./setup.sh` walks zero -> backend -> model in one chain.

---

## Task 7 -- live validation on the 1070 (ground truth) + full suite

**Goal:** prove the picker end-to-end on real hardware and a real download.

**Files:** none (validation); record results in CHANGELOG / a short log; pin
`sha256` for the downloaded model(s) into `cogito_models.py`.

**Steps:**
- Full suite green: `.venv/bin/python -m unittest discover tests -v` (models,
  memory-detect, fit, modeldl, modelpick) + S1's suite + existing smoke tests.
- **Live pick + download:** `uv run cogito-model --model qwen2.5-1.5b --dest
  ./models` on the box -> downloads to the ZFS dev zone, progress renders, final
  size matches, atomic rename. Kill mid-download and re-run -> confirm `Range`
  resume (not a restart). Record the real `sha256` and pin it.
- **E2E to a run:** take the emitted run command and launch a short
  `uv run cogito --model ./models/<file>.gguf --gpu-layers -1 --cycles 2` ->
  confirm from llama.cpp's log that layers offload to the 1070 and it generates
  (reuses the S1-proven `cu124` path). Then a `--gpu-layers` partial value from
  a larger model's fit to sanity-check the partial estimate is loadable.

**Verify:** download resumes correctly; full suite green; a real cogito cycle
runs on the 1070 via the picker's suggested command.

**Done when:** picker -> download (with resume) -> suggested command -> real
offloaded run is proven on metal, and the downloaded model's `sha256` is pinned.

---

## Sequencing & dependencies

- Task 1 (catalog) and Task 2 (memory detect) are independent; do 1 then 2.
- Task 3 (fit) depends on 1's `Model` + 2's memory fields.
- Task 4 (downloader) is independent of 1-3 (pure engine); can be done anytime
  after 1 for the `Model` shape, but has no logic dependency.
- Task 5 (wizard) depends on 1, 3, 4. Task 6 depends on 5. Task 7 depends on all.
- Riskiest: **Task 3** (the fit math + partial estimate -- table tests are the
  guard) and **Task 7** (real download + the partial-offload sanity check;
  defined fallback if the estimate needs tuning).

## Execution options

- **Solo (Claude), test-first, per-task commits** to
  `claude/installer-model-picker` -- the default. Pure cores (1/3/4) get tests
  before code, matching S1.
- Subagent-driven (superpowers:subagent-driven-development), one agent per task
  with review -- reasonable given the clean seams; Tasks 1-4 are especially
  parallel-friendly (1, 2, and 4 have no inter-dependencies).
- Opus is right-sized for all of these; no Fable escalation needed.

Commit per task to the feature branch (standing OK). Push / PR / merge to `main`
(and the eventual origin push -- local `main` is still 10 ahead from S1): ask.
