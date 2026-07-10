# COGITO Guided Model Picker (Installer Session 2) -- Design

Date: 2026-07-10
Status: Approved (brainstorm) -- pending implementation plan
Follows: `2026-07-09-installer-backend-screen-design.md` (S1, merged @ de12897)

## Purpose

S1 gave COGITO a guided backend installer (`cogito-install`): detect hardware,
recommend and install the matching llama-cpp-python backend, verify it loads.
Its exit currently tells the user "Get a GGUF model (see README)". S2 fills that
seam with a **guided model picker** so a newcomer goes from a fresh clone to a
downloaded, hardware-appropriate model and a ready-to-run command with no manual
Hugging Face spelunking.

This is the README's standing promise: "A guided model picker with size and
VRAM-fit checks is planned for a future release."

Target operator, as always: a junior admin at 2 AM. Fail early, fail with the
number and the fix, never assume.

## Scope

In scope:
- A curated GGUF catalog (code-as-config, mirroring `cogito_backends.py`).
- Memory detection (total VRAM, total system RAM) added to `cogito_detect.py`.
- Fit annotation of each catalog entry against detected memory.
- Storage-path prompt + free-space precheck before any download.
- A stdlib-only download engine with resume and progress.
- A ready-to-paste first-run command suggestion on success.
- A standalone `cogito-model` console command, chained from `cogito-install`.

Out of scope (deferred to S3): launching the ponder loop, genesis/cycle
selection, publish/README-polish/release-tag.

## Key decisions (from the brainstorm)

1. **Download = stdlib `urllib`, no new dependency.** Keeps the front door
   dependency-light (the core install every user pays for stays tiny; no
   supply-chain surface added). We write our own progress bar and `Range`-header
   resume (~20 lines). Declined: adding `huggingface_hub` to core; declined:
   detecting and shelling out to an installed `huggingface-cli` (a second
   download code path to test and keep in sync with its output, for mostly
   resume -- which we already implement).

2. **Gated models via `HF_TOKEN` passthrough.** If `HF_TOKEN` or
   `HF_HUB_TOKEN` is set in the environment, add an `Authorization: Bearer`
   header to the request. ~2 lines, zero dependencies, unlocks gated repos for
   anyone who has a token. Runtime-first (env var, no config file). The curated
   catalog itself lists only **ungated** repos, so auth is never required for
   the guided path -- the token is a pure convenience for advanced users.

3. **Standalone `cogito-model` command, chained from the installer.** Mirrors
   S1's shape (setup.sh -> cogito-install -> cogito-model). Benefits: one clean
   testable wizard module; re-runnable (`uv run cogito-model` to grab another
   model later); the full zero-to-run chain for a newcomer. Rejected: folding
   the picker into `cogito-install` as a second screen (fatter module, and you
   could not re-pick a model without re-touching the backend install).

4. **One curated quant per catalog entry (Q4_K_M).** Not a quant menu. If a
   model does not fit, the wizard steps the user **down the model ladder** (e.g.
   "Qwen2.5-32B will not fit your 8 GB card; Qwen2.5-7B will") rather than
   offering five quant variants of one model. Keeps the catalog and UI legible;
   honors "don't over-engineer". An advanced user can still hand any GGUF to
   `cogito --model` directly.

5. **Fit annotates, never hides.** Each model is tagged *fits on GPU* /
   *partial offload* / *CPU (RAM)* against detected memory. A CPU user can still
   run a 7B slowly; partial offload is legitimate. Label honestly rather than
   filtering the list down and hiding options.

6. **Default storage = `./models/`** (already gitignored). Prompted, with a
   free-space precheck. On the wks1 dev box this path resolves onto the ZFS dev
   zone (`/mnt/usb-single-1/dev/cogito/models`) automatically.

## Architecture

```
setup.sh -> cogito-install (S1) --success--> cogito-model (S2, new)
```

New / changed modules:

- **`cogito_models.py`** (new) -- the catalog. Code-as-config, same pattern as
  `cogito_backends.py`. A frozen `Model` dataclass plus `iter_models()`,
  `by_key()`, `keys()`.

  ```python
  @dataclass(frozen=True)
  class Model:
      key: str          # short stable id, e.g. "qwen2.5-7b"
      name: str         # human label, e.g. "Qwen2.5 7B Instruct"
      params_b: float   # parameter count in billions (fit + display)
      hf_repo: str      # ungated HF repo, e.g. "bartowski/Qwen2.5-7B-Instruct-GGUF"
      hf_file: str      # GGUF filename within the repo
      quant: str        # "Q4_K_M"
      size_bytes: int   # on-disk / download size (verified at build time)
      sha256: Optional[str] = None  # pinned when known; verified when present
      notes: str = ""
  ```

  Download URL is derived, not stored: `https://huggingface.co/{hf_repo}/
  resolve/main/{hf_file}`.

- **`cogito_detect.py`** (extend) -- add two fields to `Detection`:
  `vram_total_mb: Optional[int]` and `ram_total_mb: Optional[int]`. Add pure
  parsers `parse_vram(text)` (from `nvidia-smi --query-gpu=memory.total
  --format=csv,noheader,nounits`) and `parse_meminfo(text)` (MemTotal from
  `/proc/meminfo`; kB -> MB). `detect()` populates them via the existing
  injected `run`/`which` seams. RAM read is stdlib only (read `/proc/meminfo`,
  or `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` off-Linux); no
  `psutil`. The module stays cogito-agnostic and lift-ready.

- **`cogito_modeldl.py`** (new) -- the download engine, kept separate from the
  wizard so it is unit-testable in isolation. Pure logic + one injected
  side-effect seam (a URL-opener factory and the destination path). Responsible
  for: streaming to `<file>.part`, `Range`-header resume, progress reporting,
  optional `HF_TOKEN` header, final size (and optional sha256) verification,
  atomic rename `.part -> final` on success, bounded retries.

- **`cogito_modelpick.py`** (new) -- the `cogito-model` wizard. Orchestrates:
  detect memory -> annotate catalog by fit -> present -> pick -> resolve storage
  path + free-space precheck -> download (via `cogito_modeldl`) -> suggest the
  exact first-run command. All side effects injected (`detect_fn`, catalog,
  `input_fn`, downloader, `out`, `which`) exactly like S1's wizard.

- **`pyproject.toml`** -- add the `cogito-model` console entry point.
- **`setup.sh`** / `cogito_install.py` `_next_steps` -- chain into
  `cogito-model` on a successful, verified backend install (replacing the
  "see README" text). Re-runnable note preserved.

## Data flow

1. `detect()` -> memory facts (`vram_total_mb`, `ram_total_mb`, plus S1 fields).
2. For each catalog `Model`, `fit(model, detection)` -> a `Fit` result:
   - `gpu` if `size_bytes * headroom <= vram_total_mb` (fits fully on GPU),
   - `partial` if it exceeds VRAM but a meaningful fraction of layers offload,
   - `cpu` otherwise (runs from RAM; flag if it exceeds RAM too).
   `headroom` accounts for KV-cache/context overhead (a multiplicative factor,
   tuned in implementation; conservative).
3. Wizard presents the catalog with fit tags; recommends the largest model
   tagged `gpu` (or the best `cpu` fit when there is no GPU). Enter accepts.
4. If the chosen model does not fit, offer the next model down the ladder.
5. Resolve storage path (default `./models`, prompt). Create on request.
   `shutil.disk_usage(path).free` vs `size_bytes + headroom` -- fail clean if
   short, with the actual numbers.
6. Download via `cogito_modeldl` (resume if a `.part` exists; skip if the
   complete file is already present).
7. Print the ready-to-paste command, with `--gpu-layers` derived from the fit:
   `gpu` -> `-1` (all), `partial` -> a computed layer count, `cpu` -> `0`.

   ```
   uv run cogito --model ./models/<file>.gguf --gpu-layers <n> \
       --genesis-type mirror --cycles 50
   ```

## Curated catalog (initial)

All ungated; exact `hf_repo` / `hf_file` / `size_bytes` (and `sha256` where
pinned) are verified against Hugging Face at implementation time. Sizes below
are approximate Q4_K_M and for orientation only.

| key | Model | Params | ~Q4_K_M | Role |
|---|---|---|---|---|
| qwen2.5-0.5b | Qwen2.5 0.5B Instruct | 0.5B | ~0.4 GB | tiny / CPU / GTX 1070 |
| qwen2.5-1.5b | Qwen2.5 1.5B Instruct | 1.5B | ~1.1 GB | the 1070 test model |
| qwen2.5-7b | Qwen2.5 7B Instruct | 7B | ~4.7 GB | 8 GB cards; small default |
| qwen2.5-14b | Qwen2.5 14B Instruct | 14B | ~9 GB | 12-16 GB cards |
| qwen2.5-32b | Qwen2.5 32B Instruct | 32B | ~20 GB | the CHANGELOG model; 24 GB |
| mistral-7b | Mistral 7B Instruct v0.3 | 7B | ~4.4 GB | Apache-2.0; different lineage |
| phi-3.5-mini | Phi-3.5 mini Instruct | 3.8B | ~2.3 GB | MIT; synthetic-data voice |
| deepseek-r1-qwen-7b | DeepSeek-R1-Distill-Qwen 7B | 7B | ~4.7 GB | MIT; explicit reasoning mode |

The Qwen2.5 ladder gives reproducibility (every CHANGELOG run used Qwen2.5-32B)
and natural fit step-down. The three contrast picks add distinct cognitive
modes -- a different lab (Mistral), a synthetic-data model (Phi), and a
think-out-loud reasoning model (DeepSeek-R1 distill), which is the most
interesting subject for a recursive self-prompting loop.

## Error handling

Every failure exits clean and actionable:

- **Interrupted / dropped download** -- stream to `<file>.part`, resume via
  `Range` on re-run; bounded retries with backoff, then give up printing the
  direct URL, a `wget`/`curl -C -` fallback, and the HF page link. Ctrl-C leaves
  the `.part` intact for resume.
- **Integrity** -- always verify final size against `size_bytes` (catches
  truncation and HTML error pages masquerading as a GGUF). When the catalog
  pins `sha256`, verify it too. Do not require hashes for every entry (GGUF
  re-quantization drifts hashes and would break downloads): size always, hash
  when pinned.
- **Disk space** -- `shutil.disk_usage` on the target vs `size_bytes` + headroom
  BEFORE writing a byte; fail with the actual free/needed numbers.
- **Bad storage path** -- non-existent -> offer to create; not writable -> say
  so with the path.
- **Gated / 401 / 403** -- should not happen (ungated catalog); if it does, the
  message names `HF_TOKEN` as the fix.
- **Already downloaded** -- a complete file (size match) at the destination is
  detected; skip the download, go to the run-command suggestion. Idempotent.
- **Backend not installed yet** (ran `cogito-model` first) -- warn, point at
  `cogito-install`, but still allow the download (model is backend-independent).
- **Explicit opt-in downloads** -- `--model KEY` downloads that model (tty or
  not); `--yes` is an explicit accept and downloads the **recommended** model
  (the largest that fits), consistent with S1's `--yes`.
- **Implicit non-interactive** (no tty, and neither `--model` nor `--yes`) --
  print the fit-annotated catalog and exit WITHOUT downloading. We never pull
  multiple GB unattended as a surprise side effect; a deliberate flag is
  required to fetch under automation.

Runtime-first flags (mirror S1): `--model KEY`, `--dest PATH`, `--yes`,
`--dry-run`.

## Testing

Mirrors S1: stdlib `unittest`, all side effects injected, no network in tests.

- **Pure / table-driven:** `fit(model, detection)` across VRAM/RAM tiers (full /
  partial / cpu, and cpu-exceeds-RAM); `parse_vram` and `parse_meminfo` with
  fixtures; download-URL derivation; free-space decision; ladder step-down.
- **Download engine (`cogito_modeldl`):** inject the URL-opener + filesystem
  seam -- assert `Range` header sent when a `.part` exists, progress accounting,
  size-mismatch rejection, retry-then-give-up, atomic rename on success, and
  `HF_TOKEN` header present only when the env var is set.
- **Wizard (`cogito_modelpick`):** inject `detect_fn`, catalog, `input_fn`,
  downloader, `out` -- pick flow, doesn't-fit -> ladder step-down, path precheck
  failure, already-present skip, non-interactive behavior, and each flag
  (`--model` / `--dest` / `--yes` / `--dry-run`).

## Reuse note (future, not built now)

`cogito_detect.py` stays cogito-agnostic and lift-ready (the memory fields are
generic GPU/host facts). `cogito_modeldl.py` is a generic resumable HF GGUF
downloader -- a candidate shared module for other llama.cpp consumers. Do not
extract until a second real consumer exists (premature-abstraction guard).

## STIG / host note

Model files are data and land under `./models` -> on the wks1 dev box that is
the ZFS dev zone (`/mnt/usb-single-1/dev`), the correct place for multi-GB data
(the boot flash is small). No new executable surface; no daemon; no privileged
action. Any STIG friction encountered during implementation gets documented and
loosened only if justified, per the workspace posture.
