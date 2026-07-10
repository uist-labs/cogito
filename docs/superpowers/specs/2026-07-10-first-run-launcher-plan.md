# COGITO First-Run Launcher (S3) -- Implementation Plan

Companion to `2026-07-10-first-run-launcher-design.md`. Execute on branch
`claude/first-run-launcher` in `/mnt/usb-single-1/dev/cogito` (the
fapolicyd-sane dev zone; home is noexec). Tasks are ordered; each is
independently verifiable. Pure logic (model resolution, offload lookup, the
parameter wizard, argv construction) is written **test-first** and needs no
hardware, network, or model load. Side effects (the ponder loop, the
visualizer, stdin, the filesystem listing, the clock) are injected/mocked so
the suite runs green on the dev box offline.

Mirrors S1/S2 conventions: stdlib `unittest` (not pytest -- match the repo's
`tests/` convention), one injected seam per side effect, per-task commit to the
feature branch.

Verified/assumed facts:
- S1 shipped `cogito_detect.py` (`Detection` + `detect()` + memory fields) and
  `cogito_install.py`; S2 shipped `cogito_models.py` (`Model` with `hf_file`,
  `n_layers`; `fit()` -> `Fit.gpu_layers`) and `cogito_modelpick.py` (with a
  `_backend_installed()` importlib check). Both merged to local `main` @ 45a1373.
- `cogito.main()` builds a `CogitoConfig` from argparse and calls
  `Cogito(config).run()`; it currently reads `sys.argv` (no `argv` param) and
  does not `sys.exit`. `--genesis-type` accepts the 12 built-ins in
  `GENESIS_PROMPTS` plus `custom` (which reads `--genesis-prompt`).
- `visualize.main()` takes a positional `log_dir` plus `--stats-only`/`--no-show`
  and likewise reads `sys.argv`.
- `cogito_modelpick.main` already uses the `main(argv=None, *, ...seams)`
  convention this module copies.
- wks1 dev box: GTX 1070 (8 GB VRAM, Pascal CC 6.1), `cu124` proven E2E in
  S1/S2 -- the live launch rig. GOTCHA (from S1/S2): a plain `uv run` re-syncs
  and uninstalls the out-of-lock CUDA-runtime pip pkgs; a GPU launch runs with
  `uv run --no-sync` and `LD_LIBRARY_PATH` to the `.venv` nvidia lib dirs.

---

## Task 1 -- `argv=None` seams on `cogito.main` and `visualize.main` (enabling touch-up)

**Goal:** make both entry points callable in-process with an explicit argv, so
the launcher's default run/viz seams need no `sys.argv` juggling or subprocess.

**Files:** `cogito.py`, `visualize.py`; add `tests/test_argv_seam.py`.

**Steps:**
- `cogito.py`: `def main(argv=None):` -> `args = parser.parse_args(argv)`.
  Nothing else changes; `argv=None` reads `sys.argv` exactly as before when run
  as a script.
- `visualize.py`: same one-line change.
- **Tests first:** assert `cogito.main(['--help'])` and `visualize.main(['--help'])`
  raise `SystemExit(0)` (argparse help path -- proves argv is threaded without
  loading a model or matplotlib figure); assert a bad flag raises `SystemExit(2)`.
  (Keep the test to argparse behaviour; do not invoke a real run here.)

**Verify:** `.venv/bin/python -m unittest tests.test_argv_seam -v` green; the S1/S2
suites still green (additive change).

**Done when:** both `main`s accept an argv list and behave identically when
called with `None`.

---

## Task 2 -- `cogito_launch.py`: model resolution + offload (test-first, pure)

**Goal:** the two pure decisions the launcher makes before any prompting.

**Files:** add `cogito_launch.py`; add `tests/test_launch_resolve.py`.

**Steps (tests first):**
- `_resolve_model(model_flag, dest, lister) -> Path | None`:
  - `model_flag` given -> that path (error clean if it does not exist).
  - else `lister(dest)` returns the `*.gguf` files: 0 -> `None` (caller prints the
    `cogito-model` guidance and exits non-zero); 1 -> that file; N -> caller's
    pick (return the list; the interactive pick lives in Task 3's prompt layer,
    or a small `_pick(files, input_fn, out)` here). Keep `lister` injected
    (default globs `dest`); most-recently-modified floats to `[1]`.
- `_resolve_offload(model_path, catalog, detection) -> (gpu_layers, note)`:
  basename matches a catalog entry's `hf_file` -> `fit(model, detection).gpu_layers`
  and its reason as the note; no match -> `(-1, "<file> is not in the catalog;
  attempting full GPU offload -- lower --gpu-layers if you hit OOM")`.
- **Table-driven tests:** explicit `--model` (exists / missing); 0/1/N discovered
  files (with a fake `lister`); N-file pick incl. bad-then-good input and
  Enter-default; catalog match -> the fit's `gpu_layers`; non-match -> `(-1,
  note)`.

**Verify:** `.venv/bin/python -m unittest tests.test_launch_resolve -v` green,
offline.

**Done when:** every model-source and offload branch is covered with no real
filesystem or GPU access.

---

## Task 3 -- `cogito_launch.py`: parameter wizard + argv build (test-first, pure)

**Goal:** the START/tune gate, the seven-field wizard, and the exact cogito argv.

**Files:** `cogito_launch.py`; add `tests/test_launch_wizard.py`.

**Steps (tests first):**
- `DEFAULTS`: `genesis_type="mirror"`, `cycles=50`, `context_size=16384`,
  `tokens_per_cycle=256`, `temperature=0.8`, `top_p=0.95`, `repeat_penalty=1.1`
  (match `cogito.py`'s own defaults so the wizard and the CLI agree).
- `_prompt_params(defaults, input_fn, out) -> params`:
  - Gate: `"Start now, or tune parameters? [START/tune]"` -- empty/`start` ->
    return `defaults`; `tune` -> walk the seven fields.
  - Each field prints `name [default]:`; empty accepts the default; a value is
    validated (int fields int + range, float fields float + range, `genesis_type`
    in `GENESIS_PROMPTS` keys + `custom`); invalid -> re-prompt (never crash).
  - `genesis_type == "custom"` -> prompt for the prompt text (non-empty).
- `_build_argv(model_path, params, gpu_layers, log_dir) -> list[str]`: the cogito
  CLI argv -- `--model`, `--genesis-type` (+`--genesis-prompt` when custom),
  `--cycles`, `--context-size`, `--tokens-per-cycle`, `--temperature`, `--top-p`,
  `--repeat-penalty`, `--gpu-layers`, `--log-dir`. Pure; this is the contract the
  launch seam and `--dry-run` render.
- **Tests first:** `--yes`/START bypass -> defaults verbatim; `tune` walking all
  seven (scripted stdin) incl. each Enter-default and each re-prompt-on-bad-input;
  `custom` genesis captures the prompt text; `_build_argv` for defaults and for
  every override produces the exact expected argv (incl. the custom-prompt pair
  and the resolved `--gpu-layers`).

**Verify:** `.venv/bin/python -m unittest tests.test_launch_wizard -v` green.

**Done when:** the gate, all seven fields (Enter + validation + custom), and the
argv contract are covered.

---

## Task 4 -- `cogito_launch.py`: flow + entry point (test-first)

**Goal:** wire the seams into `main()` and register `cogito-run`.

**Files:** `cogito_launch.py`; register `cogito-run = "cogito_launch:main"` in
`pyproject.toml` `[project.scripts]`; add `tests/test_launch_flow.py`.

**Steps:**
- `main(argv=None, *, detect_fn=None, catalog=None, input_fn=input, lister=None,
  run_fn=None, viz_fn=None, out=sys.stdout, now=None) -> int` with argparse:
  `--model`, `--dest` (default `./models`), `--yes`, `--dry-run`, plus
  pass-through overrides (`--genesis-type`, `--cycles`, `--context-size`,
  `--tokens-per-cycle`, `--temperature`, `--top-p`, `--repeat-penalty`) that
  pre-seed `DEFAULTS`.
- Flow: backend check (reuse S2's `_backend_installed()`; absent -> point at
  `cogito-install`, return non-zero) -> `_resolve_model` (None -> guidance,
  return non-zero) -> `detect_fn()` -> `_resolve_offload` -> per-run
  `log_dir = ./logs/run_<now>` -> `--yes` skips the gate (all defaults) else
  `_prompt_params` -> `_build_argv`.
  - `--dry-run`: print the resolved model, the full cogito command, and the
    planned `cogito-viz <log_dir>`; call neither seam; return 0.
  - else `run_fn(argv)` (default: `cogito.main(argv)`), wrapped so a
    `KeyboardInterrupt` is caught and treated as a normal end-of-run; then
    `viz_fn(log_dir)` (default: `visualize.main([log_dir])`) wrapped so any
    exception prints a warning naming the dir + the manual `uv run cogito-viz
    <dir>` command but still returns 0 (the run succeeded).
- **Tests first (fakes for every seam):** happy path calls `run_fn` with the
  argv `_build_argv` would produce and then `viz_fn(log_dir)`; `run_fn` raising
  `KeyboardInterrupt` still calls `viz_fn`; `viz_fn` raising -> warning printed,
  exit 0; backend-absent -> no `run_fn`, non-zero; no-model -> guidance, non-zero;
  `--dry-run` -> neither seam, plan printed; a pass-through override reaches the
  argv. `now` injected for a deterministic `log_dir`.

**Verify:** `.venv/bin/python -m unittest tests.test_launch_flow -v` green;
`uv run cogito-run --dry-run` on the box (with a model present in `./models`)
prints the resolved plan and launches nothing.

**Done when:** the flow drives every branch through injected seams; the entry
point resolves.

---

## Task 5 -- setup.sh chain + CHANGELOG + README

**Goal:** make the guided front door end in a running experiment; record it.

**Files:** `setup.sh`; `CHANGELOG.md`; `README.md`.

**Steps:**
- `setup.sh`: after the `cogito-model` step, chain
  `exec uv run cogito-run --yes` (all-defaults, no gate) so a fresh clone flows
  install -> model -> a live run. Keep a `|| echo "re-run 'uv run cogito-run'
  any time"` safety note consistent with the existing idiom.
- CHANGELOG `[Unreleased]`: a `First-run launcher (cogito-run)` section --
  discovers the model, reuses the fit for offload, START/tune wizard over the
  seven run parameters, auto-hands the finished run to `cogito-viz`.
- README: extend the Quick Start so the guided path is shown ending in a run
  (and `cogito-run` noted as re-runnable). Full README polish is the publish
  half of S3; this is just the accurate one-liner.

**Verify:** `.venv/bin/python -m unittest discover tests -v` all green;
`shellcheck setup.sh` clean.

**Done when:** `./setup.sh` (dry-read) chains zero -> backend -> model -> run.

---

## Task 6 -- live validation on the 1070 (ground truth) + full suite

**Goal:** prove the launcher end-to-end on real hardware.

**Files:** none (validation); record results in the CHANGELOG / a short log.

**Steps:**
- Full suite green: `.venv/bin/python -m unittest discover tests -v` (argv seam,
  launch resolve/wizard/flow) + S1/S2 suites + existing smoke tests.
- **Live launch (GPU), START path:** with a model already downloaded to
  `./models` (from S2's qwen2.5-1.5b) and the CUDA runtime restored per the
  gotcha, run `LD_LIBRARY_PATH=... uv run --no-sync cogito-run` -> accept START
  -> confirm the loop streams thoughts, the per-run `./logs/run_<ts>` dir fills
  (log + checkpoints + transcript), and `cogito-viz` fires automatically and
  writes its PNG into that dir. Use a short finite `--cycles` (e.g. `--cycles 3`)
  so the run completes and the auto-viz path (not just Ctrl-C) is exercised.
- **Tune path:** one run choosing `tune` and changing `genesis-type` (e.g. to
  `void`) -> confirm the override reaches the loop (the transcript's genesis
  matches) and offload still lands 29/29 on the 1070.
- **Ctrl-C path:** an infinite `--cycles 0` run interrupted after a couple of
  cycles -> confirm the launcher still hands off to `cogito-viz` on the partial
  run.

**Verify:** the streamed run completes, the per-run dir is self-contained, and
`cogito-viz` produced its PNG automatically on both the finite and the
interrupted run; full suite green.

**Done when:** discover -> model -> START launch -> live offloaded run -> auto
`cogito-viz` is proven on metal, and the tune + Ctrl-C paths are confirmed.

---

## Sequencing & dependencies

- Task 1 (argv seams) is independent and unblocks Task 4's default seams.
- Task 2 (resolve/offload) and Task 3 (wizard/argv) are independent pure logic;
  either order.
- Task 4 (flow) depends on 1, 2, 3. Task 5 depends on 4. Task 6 depends on all.
- Riskiest: **Task 4** (the KeyboardInterrupt-still-vizzes wiring and the
  per-run log-dir handoff -- the fake-seam tests are the guard) and **Task 6**
  (real loop + auto-viz on the 1070; the CUDA-runtime `--no-sync` gotcha is the
  known trap, mitigated by the documented restore recipe).

## Execution options

- **Solo (Claude), test-first, per-task commits** to
  `claude/first-run-launcher` -- the default, matching S1/S2. Pure cores (2, 3)
  get tests before code.
- Subagent-driven (superpowers:subagent-driven-development), one agent per task
  with review -- reasonable given the clean seams; Tasks 2 and 3 are
  parallel-friendly.
- Opus is right-sized for all of these; no Fable escalation needed.

Commit per task to the feature branch (standing OK). Merge to local `main` and
the eventual `origin` push of all commits (local `main` is 19 ahead from
S1+S2, and this launcher stacks on top): ask.
