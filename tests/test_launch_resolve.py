#!/usr/bin/env python3
"""TDD tests for cogito_launch's pure resolution helpers.

These cover the two decisions the launcher makes before any prompting:
which model to run (--model / discover / pick) and how many layers to offload
(reuse the catalog fit, or fall back for an unknown file). Every side effect is
injected -- the directory listing (lister) and stdin (input_fn) are fakes; only
the real-glob test touches a temp dir. No GPU, no model load. Stdlib unittest.

Run with:  .venv/bin/python -m unittest tests.test_launch_resolve
"""

import io
import os
import tempfile
import unittest
from pathlib import Path

import cogito_detect as d
import cogito_models as models
import cogito_launch as launch

EIGHT_GB = d.Detection(vram_total_mb=8192, ram_total_mb=15719)


def scripted(responses):
    """An input_fn returning each queued response in turn."""
    it = iter(responses)
    return lambda _prompt="": next(it)


def never_prompt(_prompt=""):
    raise AssertionError("should not have prompted")


class DiscoverTest(unittest.TestCase):
    def test_globs_ggufs_only_and_sorts_most_recent_first(self):
        with tempfile.TemporaryDirectory() as dd:
            old = Path(dd) / "old.gguf"; old.write_bytes(b"x")
            new = Path(dd) / "new.gguf"; new.write_bytes(b"x")
            note = Path(dd) / "note.txt"; note.write_bytes(b"x")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(launch.discover_ggufs(dd), [new, old])

    def test_empty_dir_or_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as dd:
            self.assertEqual(launch.discover_ggufs(dd), [])
        self.assertEqual(launch.discover_ggufs("/no/such/dir"), [])


class ResolveModelTest(unittest.TestCase):
    def test_explicit_flag_that_exists_is_used(self):
        with tempfile.TemporaryDirectory() as dd:
            p = Path(dd) / "m.gguf"; p.write_bytes(b"x")
            got = launch.resolve_model(str(p), dd, lister=lambda _d: [], out=io.StringIO())
            self.assertEqual(got, p)

    def test_explicit_flag_missing_returns_none_with_message(self):
        out = io.StringIO()
        got = launch.resolve_model("/nope/x.gguf", "/models",
                                   lister=lambda _d: [], out=out)
        self.assertIsNone(got)
        self.assertIn("not found", out.getvalue().lower())

    def test_zero_discovered_points_at_cogito_model(self):
        out = io.StringIO()
        got = launch.resolve_model(None, "/models", lister=lambda _d: [], out=out)
        self.assertIsNone(got)
        self.assertIn("cogito-model", out.getvalue())

    def test_one_discovered_used_without_prompting(self):
        f = Path("/models/only.gguf")
        got = launch.resolve_model(None, "/models", lister=lambda _d: [f],
                                   input_fn=never_prompt, out=io.StringIO())
        self.assertEqual(got, f)

    def test_many_discovered_enter_accepts_first(self):
        files = [Path("/m/a.gguf"), Path("/m/b.gguf")]
        got = launch.resolve_model(None, "/m", lister=lambda _d: files,
                                   input_fn=scripted([""]), out=io.StringIO())
        self.assertEqual(got, files[0])

    def test_many_discovered_number_overrides(self):
        files = [Path("/m/a.gguf"), Path("/m/b.gguf")]
        got = launch.resolve_model(None, "/m", lister=lambda _d: files,
                                   input_fn=scripted(["2"]), out=io.StringIO())
        self.assertEqual(got, files[1])

    def test_many_discovered_bad_input_reprompts(self):
        files = [Path("/m/a.gguf"), Path("/m/b.gguf")]
        got = launch.resolve_model(None, "/m", lister=lambda _d: files,
                                   input_fn=scripted(["9", "x", "1"]), out=io.StringIO())
        self.assertEqual(got, files[0])

    def test_many_discovered_non_interactive_uses_most_recent_without_prompt(self):
        files = [Path("/m/a.gguf"), Path("/m/b.gguf")]
        out = io.StringIO()
        got = launch.resolve_model(None, "/m", lister=lambda _d: files,
                                   input_fn=never_prompt, out=out, interactive=False)
        self.assertEqual(got, files[0])
        self.assertIn("most recent", out.getvalue().lower())


class ResolveOffloadTest(unittest.TestCase):
    def test_catalog_match_uses_fit_values(self):
        m = next(iter(models.iter_models()))
        path = Path("/models") / m.hf_file
        catalog = list(models.iter_models())
        gpu_layers, note = launch.resolve_offload(path, catalog, EIGHT_GB)
        expected = models.fit(m, EIGHT_GB)
        self.assertEqual(gpu_layers, expected.gpu_layers)
        self.assertIn(expected.reason, note)

    def test_unknown_file_falls_back_to_full_offload_with_note(self):
        path = Path("/models/mystery-model.gguf")
        catalog = list(models.iter_models())
        gpu_layers, note = launch.resolve_offload(path, catalog, EIGHT_GB)
        self.assertEqual(gpu_layers, -1)
        self.assertIn("catalog", note.lower())
        self.assertIn("gpu-layers", note.lower())


if __name__ == "__main__":
    unittest.main()
