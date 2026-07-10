#!/usr/bin/env python3
"""TDD tests for the argv=None seams on cogito.main and visualize.main.

The launcher (cogito_launch) drives both entry points in-process, passing an
explicit argv list. These tests prove each main() threads that argv through
argparse (rather than reading sys.argv) without loading a model or a figure.
argparse's own output is captured so the suite stays pristine. Stdlib unittest.

Run with:  .venv/bin/python -m unittest tests.test_argv_seam
"""

import contextlib
import io
import unittest

import cogito
import visualize


class ArgvSeamTest(unittest.TestCase):
    def test_cogito_main_accepts_argv_and_errors_without_model(self):
        # Explicit empty argv -> argparse sees no --model (required) -> exit 2.
        # Proves argv is parsed, not sys.argv (which under unittest is not empty).
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            cogito.main([])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("model", buf.getvalue().lower())

    def test_cogito_main_help_exits_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            cogito.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_visualize_main_accepts_argv_and_errors_without_dir(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit) as cm:
            visualize.main([])
        self.assertEqual(cm.exception.code, 2)

    def test_visualize_main_help_exits_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            visualize.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
