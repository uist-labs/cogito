# COGITO First-Run Launcher (Installer Session 3) -- Design

Date: 2026-07-10
Status: Approved (brainstorm) -- pending implementation plan
Follows: `2026-07-10-installer-model-picker-design.md` (S2, merged @ 45a1373)

## Purpose

S1 (`cogito-install`) detects hardware and installs the matching
llama-cpp-python backend. S2 (`cogito-model`) recommends, downloads, and
verifies a hardware-appropriate GGUF -- and then dead-ends by *printing* a
`uv run cogito --model ... --genesis-type mirror --cycles 50` command for the
user to copy and paste.

S3 closes that last gap. `cogito-run` takes a downloaded model and actually
launches a ponder loop -- guiding the genesis prompt and run parameters, then
handing the finished run to `cogito-viz` automatically. A newcomer goes from a
fresh clone to a running experiment *and its visualization* without ever
copy-pasting a command or knowing which directory the logs landed in.

This is the third and final leg of the guided front door:
`setup.sh -> cogito-install -> cogito-model -> cogito-run`.

Target operator, as always: a junior admin at 2 AM. Enter-accepts every
default; fail early with the fix; never assume.

## Scope

In scope:
- A standalone `cogito-run` console command (module `cogito_launch.py`),
  chained from `cogito-model`.
- Model resolution: `--model`, else discover `*.gguf` under `./models`
  (`--dest`); handle zero / one / several files.
- GPU-offload resolution: reuse S2's `cogito_models.fit()` when the discovered
  file matches a catalog entry; otherwise fall back to cogito's offload-all
  default with a printed note.
- A two-tier launch prompt: `START` (recommended defaults) vs. `tune` (a full
  seven-field parameter wizard, every field Enter-accepts its default).
- In-process launch of the ponder loop (thoughts stream live).
- Automatic `cogito-viz` handoff on the run's log directory when the loop ends
  (normal completion or Ctrl-C).
- Runtime-first flags: `--model`, `--dest`, `--yes`, `--dry-run`, plus
  pass-through parameter overrides.
- A one-line touch-up to `cogito.py` and `visualize.py`: give each `main()` an
  `argv=None` parameter (matching `cogito_modelpick.main`) so both are callable
  in-process behind an injected seam.

Out of scope (deferred to publish, the other half of S3):
- GitHub topics, README polish, `origin` push, release tag. (Tracked
  separately; the launcher lands first, then the whole 19-commit update
  publishes together.)
- Any new run parameters beyond what `cogito.py` already exposes.
- Multi-run orchestration, scheduling, or resuming a checkpoint (cogito's own
  concern, not the launcher's).

## Key decisions (from the brainstorm)

1. **Full seven-field parameter wizard, not a minimal one.** The genesis prompt
   is the single knob that most changes what emerges (per the README's own
   analysis), and researchers reproduce runs by the sampling parameters -- so
   the wizard exposes every run-shaping knob `cogito` accepts:
   `genesis-type`, `cycles`, `context-size`, `tokens-per-cycle`, `temperature`,
   `top-p`, `repeat-penalty`. (`gpu-layers` is resolved from the fit, shown, and
   overridable.) Ken's call: seven Enter-default prompts are not overwhelming
   for the target audience. Rejected: a 2-3 field minimal wizard (hides the
   levers people actually tune); rejected: a one-keypress auto-launch (removes
   the genesis choice, the most interesting decision).

2. **Two-tier fast path: `START` vs. `tune`.** The wizard opens with a single
   gate -- "Start now with recommended settings, or tune parameters?
   `[START/tune]`". `START` (the default, Enter-accepts) launches immediately
   with all defaults; `tune` walks the full seven-field wizard. This reconciles
   "full control" with the 2 AM-admin ergonomic: a newcomer is never *forced*
   to reason about `top-p` before their first ponder, but every knob is one
   keystroke away. `--yes` skips the gate entirely (all defaults) for zero-touch
   automation. **Execution deviation (2026-07-10, Ken's call):** `setup.sh` ends
   at the *gate* (plain `cogito-run`, not `--yes`), not auto-launching -- a fresh
   clone stops for one keystroke before a multi-minute GPU run rather than
   starting it unprompted. A Ctrl-C at the gate defers cleanly ("start it when
   you're ready with 'uv run cogito-run'"). `--yes` remains for true zero-touch.

3. **Standalone `cogito-run`, chained from `cogito-model`, wizards stay
   decoupled.** Mirrors the S1/S2 shape
   (`setup.sh -> install -> model -> run`). No state is passed between the
   picker and the launcher: `cogito-run` independently discovers the model in
   `./models` and recomputes the GPU fit. Benefits: one clean testable module;
   re-runnable (`uv run cogito-run` any time to start another experiment);
   nothing to keep in sync between two wizards. Rejected: a coupled handoff
   (cogito-model writes model+fit to a state file the launcher reads) -- adds
   IPC/state the decoupled design deliberately avoids. Rejected: requiring an
   explicit `--model` always -- breaks the "setup.sh flows straight into a run"
   goal.

4. **Independent model discovery over `./models`.** `--model <path>` wins if
   given. Otherwise scan `--dest` (default `./models`) for `*.gguf`:
   - **0 files** -> print "run `uv run cogito-model` first to download a model"
     and exit non-zero. The message is guidance, not a stack trace, but a
     non-zero code keeps `setup.sh`'s chain honest -- no run happened.
   - **1 file** -> use it.
   - **N files** -> a numbered pick list (kiln style, Enter-accepts the
     most-recently-modified as `[1]`).

5. **GPU offload reuses S2's `fit()`, with an honest fallback.** If the resolved
   file's basename matches a catalog entry's `hf_file`, run
   `cogito_models.fit(model, detection)` to get `gpu_layers` (exactly what
   `cogito-model` showed). If it does not match any catalog entry (a
   hand-downloaded or renamed GGUF), we cannot know the model's layer count, so
   we fall back to cogito's own default (`--gpu-layers -1`, offload-all) and
   print the same note the picker uses: "attempting full GPU offload; lower
   `--gpu-layers` if you hit OOM." Never guess a wrong specific number.

6. **In-process launch behind an injected seam, not `uv run` shelling.** The
   default launch runner calls `cogito.main(argv=[...])` in the *same* venv --
   no nested `uv run`, and the seam is trivially mockable in tests. Same pattern
   for the viz handoff (`visualize.main([log_dir, ...])`). This requires giving
   `cogito.main` and `visualize.main` an `argv=None` parameter
   (`parser.parse_args(argv)`), matching the convention `cogito_modelpick.main`
   already follows. Rejected: `subprocess.run(["uv","run","cogito",...])` --
   spawns a second uv resolution inside an already-synced env, and streams/tests
   awkwardly.

7. **Automatic `cogito-viz` on completion -- "one less thing to forget."** When
   the loop returns (finite `--cycles` done, or a Ctrl-C on an infinite run),
   the launcher runs `cogito-viz` on the run's log directory automatically, so
   the dashboard PNG is waiting without the user remembering the directory or
   the command. A viz failure after a good run is a *warning*, never a failure
   -- the ponder run is the valuable artifact. Ken's call: removes the "did I
   run that?" gap.

8. **Per-run log directory, handed straight to viz.** The launcher chooses an
   explicit per-run `--log-dir` (e.g. `./logs/run_<timestamp>`) and passes the
   same path to `cogito-viz`, so each run's artifacts are self-contained and the
   handoff points at exactly the right directory (rather than a shared `./logs`
   that accumulates multiple runs and confuses the visualizer).

## Architecture

One new module plus two one-line touch-ups. Nothing else changes.

### `cogito_launch.py` (new) -- the `cogito-run` wizard

Single flat module, stdlib-only (no new dependency), following S1/S2 seams.

```
main(argv=None, *, detect_fn=None, catalog=None, input_fn=input,
     run_fn=None, viz_fn=None, out=sys.stdout, now=None) -> int
```

- `detect_fn` default -> `cogito_detect.detect` (VRAM/RAM for the fit).
- `catalog`   default -> `cogito_models.CATALOG` (for the fit lookup).
- `input_fn`  default -> builtin `input` (prompts; injected for tests).
- `run_fn`    default -> a thin wrapper calling `cogito.main(argv=[...])`.
- `viz_fn`    default -> a thin wrapper calling `visualize.main([log_dir])`.
- `now`       default -> a wall-clock stamp for the per-run log dir name
  (injected in tests for a deterministic path).
- `out`       default -> `sys.stdout` (all human text; injected for tests).

Internal helpers (each small, pure where possible, unit-tested in isolation):

- `_resolve_model(args, out) -> Path | None` -- flag/discover/pick logic
  (the 0/1/N cases). Pure given a listing function.
- `_resolve_offload(model_path, catalog, detection) -> (gpu_layers, note)` --
  catalog basename match -> `fit()`; else `(-1, fallback_note)`. Pure.
- `_prompt_params(defaults, input_fn, out) -> params` -- the START/tune gate and
  the seven-field wizard. `custom` genesis -> prompt for the prompt text. Each
  field validates and re-prompts on bad input; Enter accepts the default.
- `_build_argv(model_path, params, gpu_layers, log_dir) -> list[str]` -- the
  cogito CLI argv. Pure; the seam boundary that `--dry-run` prints and tests
  assert on.
- `_launch(argv, run_fn, out)` / `_visualize(log_dir, viz_fn, out)` -- call the
  seams; catch and downgrade viz failure to a warning.

### `cogito.py` (touch-up)

`def main(argv=None):` -> `parser.parse_args(argv)`. Behaviour unchanged when
called as a script (`argv=None` reads `sys.argv`). Enables the in-process seam.

### `visualize.py` (touch-up)

`def main(argv=None):` -> `parser.parse_args(argv)`. Same rationale.

### `setup.sh` (touch-up)

Append a step: after `cogito-model`, chain `exec uv run cogito-run` (at the
gate, per the execution deviation above -- not `--yes`) so the guided flow ends
ready to ponder, one keystroke from a run. Keep a `|| echo "re-run any time"`
safety idiom so a non-fatal hiccup still leaves the user with actionable next
steps.

### `pyproject.toml` (touch-up)

Add `cogito-run = "cogito_launch:main"` under `[project.scripts]`.

## Data flow

```
setup.sh --yes            interactive: uv run cogito-run
      |                                    |
      v                                    v
 _resolve_model  <-- --model / scan ./models (0->guide,1->use,N->pick)
      |
      v
 detect_fn (VRAM/RAM) --> _resolve_offload --> gpu_layers (+ note)
      |
      v
 _prompt_params:
   --yes -> all defaults (no gate)
   else  -> "START/tune?"
              START -> defaults
              tune  -> genesis,cycles,ctx,tokens,temp,top-p,repeat-penalty
      |
      v
 _build_argv --> [--dry-run prints it and stops here]
      |
      v
 run_fn(argv)  == cogito.main(argv)   (loop streams live to stdout)
      |
      v  (returns on completion or KeyboardInterrupt)
 viz_fn(log_dir) == visualize.main([log_dir])   (PNGs; failure -> warn only)
```

## Error handling

- **No backend installed** (llama-cpp-python import would fail): detected up
  front (reuse S2's `_backend_installed()` check), print
  "run `uv run cogito-install` first", exit non-zero. Do not attempt a load.
- **No model found**: print the `cogito-model` guidance, exit non-zero.
- **Bad `tune` input** (non-numeric cycles, unknown genesis, out-of-range
  sampling value): re-prompt with the valid set/range; never crash on input.
- **Ctrl-C during the loop**: a normal way to end an (often infinite) run --
  caught, treated as completion, still hands off to viz.
- **`cogito-viz` failure after a good run**: print a warning naming the log dir
  and the manual `uv run cogito-viz <dir>` command; exit success (the run
  succeeded).
- **`--dry-run`**: resolve model + offload + params, print the model, the full
  cogito command, and the planned viz command; launch nothing; exit 0.

## Testing

stdlib `unittest`, TDD RED->GREEN, matching `test_smoke.py` and the S1/S2
suites. All seams injected -- no real model load, no real download, no GPU
needed.

Table-driven / pure-logic coverage:
- `_resolve_model`: 0 files (guidance + non-zero), 1 file, N files (pick + bad
  pick re-prompt + Enter-default), explicit `--model` wins.
- `_resolve_offload`: catalog basename match -> fit's `gpu_layers`; non-match ->
  `(-1, note)`; detection-unknown -> conservative fallback.
- `_prompt_params`: `--yes` bypasses the gate; `START` uses defaults; `tune`
  walks all seven; `custom` genesis prompts for text; each field Enter-accepts
  and re-prompts on bad input.
- `_build_argv`: defaults and every override produce the exact expected cogito
  argv (this is the contract the launch seam depends on).
- Flow with fake `run_fn`/`viz_fn`: happy path calls both with the expected
  args; viz raising -> warning printed, exit 0; `run_fn` raising a
  `KeyboardInterrupt` -> still calls viz.
- `--dry-run`: prints the plan, calls neither seam.

Live validation (on the 1070, per the S1/S2 pattern): a real
`uv run cogito-run` after a `cogito-model` download -- START path, a short
finite `--cycles`, confirm the loop streams, the per-run log dir fills, and
`cogito-viz` produces its PNG automatically. One `tune`-path run changing the
genesis prompt to confirm the override reaches the loop.

## Deviations from the roadmap

The S3 roadmap paired "first-run launcher" with "publish". This spec covers the
launcher only; publish (GitHub topics, README polish, `origin` push of all 19
commits, release tag) follows as a separate, non-code step once the launcher is
merged, so the whole guided front door ships in one release.
