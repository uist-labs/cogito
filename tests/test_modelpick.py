#!/usr/bin/env python3
"""TDD tests for cogito_modelpick -- the cogito-model wizard.

Every side effect is injected: detect_fn, catalog, input_fn (scripted stdin),
downloader (recording), disk_usage, isatty, backend_installed. No hardware, no
network, no real disk writes beyond a temp dir. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_modelpick
"""

import io
import tempfile
import types
import unittest
from pathlib import Path

import cogito_detect as d
import cogito_modelpick as pick

EIGHT_GB = d.Detection(vram_total_mb=8192, ram_total_mb=15719)
NO_GPU = d.Detection(vram_total_mb=None, ram_total_mb=15719)


class Recorder:
    """A stand-in downloader that records calls instead of hitting the network."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, dest, *, size_bytes, sha256=None, out=None):
        self.calls.append(types.SimpleNamespace(
            url=url, dest=str(dest), size_bytes=size_bytes))
        return Path(dest)


def ample(_path):
    return types.SimpleNamespace(free=10 ** 12)  # 1 TB free


def tiny(_path):
    return types.SimpleNamespace(free=1000)  # 1 KB free


class PickTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.out = io.StringIO()
        self.rec = Recorder()

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, argv, inputs=(), detect=EIGHT_GB, isatty=True,
                 disk_usage=ample, backend_installed=True):
        it = iter(inputs)
        return pick.main(
            ["--dest", self.tmp, *argv],
            detect_fn=lambda: detect,
            downloader=self.rec,
            input_fn=lambda _p: next(it),
            out=self.out,
            isatty=lambda: isatty,
            disk_usage=disk_usage,
            backend_installed=lambda: backend_installed,
        )

    # --- --model flag -----------------------------------------------------
    def test_unknown_model_key_errors(self):
        rc = self.run_main(["--model", "no-such"])
        self.assertEqual(rc, 2)
        self.assertEqual(self.rec.calls, [])
        self.assertIn("no-such", self.out.getvalue())

    def test_model_flag_downloads_that_model(self):
        rc = self.run_main(["--model", "qwen2.5-1.5b"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.rec.calls), 1)
        self.assertIn("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", self.rec.calls[0].dest)
        self.assertIn("--gpu-layers -1", self.out.getvalue())  # 1.5B fits fully

    # --- recommendation / --yes ------------------------------------------
    def test_yes_downloads_recommended(self):
        rc = self.run_main(["--yes"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.rec.calls), 1)
        # largest full-GPU fit on 8 GB is a 7B.
        self.assertIn("7B", self.rec.calls[0].dest)

    # --- dry run ----------------------------------------------------------
    def test_dry_run_downloads_nothing_but_shows_plan(self):
        rc = self.run_main(["--yes", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.rec.calls, [])
        out = self.out.getvalue()
        self.assertIn("huggingface.co", out)
        self.assertIn("cogito --model", out)

    # --- implicit non-interactive ----------------------------------------
    def test_noninteractive_without_flags_lists_and_exits(self):
        rc = self.run_main([], isatty=False)
        self.assertEqual(rc, 0)
        self.assertEqual(self.rec.calls, [])
        # It listed the catalog rather than downloading.
        self.assertIn("qwen2.5-32b", self.out.getvalue())

    # --- interactive selection -------------------------------------------
    def test_enter_accepts_recommended(self):
        rc = self.run_main([], inputs=[""])  # Enter
        self.assertEqual(rc, 0)
        self.assertIn("7B", self.rec.calls[0].dest)

    def test_number_selects_that_model(self):
        # The menu is catalog order; item 1 is qwen2.5-0.5b.
        rc = self.run_main([], inputs=["1"])
        self.assertEqual(rc, 0)
        self.assertIn("0.5B", self.rec.calls[0].dest)

    def test_oversized_pick_offers_step_down_and_accepts(self):
        # Choose the 32B (item 5) on an 8 GB card, then accept the step-down.
        rc = self.run_main([], inputs=["5", "y"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.rec.calls), 1)
        self.assertIn("14B", self.rec.calls[0].dest)  # stepped 32B -> 14B

    # --- precheck / errors ------------------------------------------------
    def test_insufficient_space_aborts_before_download(self):
        rc = self.run_main(["--model", "qwen2.5-7b"], disk_usage=tiny)
        self.assertEqual(rc, 1)
        self.assertEqual(self.rec.calls, [])
        self.assertIn("space", self.out.getvalue().lower())

    def test_already_present_skips_download(self):
        target = Path(self.tmp) / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        import cogito_models as m
        with open(target, "wb") as fobj:  # sparse file of the exact pinned size
            fobj.truncate(m.by_key("qwen2.5-1.5b").size_bytes)
        rc = self.run_main(["--model", "qwen2.5-1.5b"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.rec.calls, [])  # not re-downloaded
        self.assertIn("--gpu-layers", self.out.getvalue())

    def test_backend_missing_warns_but_proceeds(self):
        rc = self.run_main(["--model", "qwen2.5-1.5b"], backend_installed=False)
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.rec.calls), 1)
        self.assertIn("cogito-install", self.out.getvalue())


if __name__ == "__main__":
    unittest.main()
