#!/usr/bin/env python3
"""Lightweight smoke tests for COGITO packaging.

These verify the package is importable and its console-facing surfaces are wired
up. They deliberately do NOT exercise the model loop (that needs a GGUF and a
backend). Stdlib only -- no pytest dependency.

Run with:  uv run python -m unittest tests.test_smoke
"""

import subprocess
import sys
import unittest


class TestBackendCatalog(unittest.TestCase):
    def test_curated_keys(self):
        import cogito_backends as b
        self.assertEqual(
            b.keys(), ["cpu", "cu124", "cu130", "metal", "vulkan", "rocm72"]
        )

    def test_by_key(self):
        import cogito_backends as b
        self.assertEqual(b.by_key("cu124").cmake_flag, "-DGGML_CUDA=on")
        self.assertIsNone(b.by_key("does-not-exist"))

    def test_every_backend_has_an_https_index(self):
        import cogito_backends as b
        for be in b.BACKENDS:
            self.assertTrue(be.index_url.startswith("https://"), be.key)


class TestImportsWithoutBackend(unittest.TestCase):
    def test_cogito_imports_without_llama_cpp(self):
        # cogito.py must import (and expose main) without the inference backend.
        import cogito
        self.assertTrue(callable(cogito.main))

    def test_visualizers_import(self):
        import visualize
        import visualize_advanced
        self.assertTrue(callable(visualize.main))
        self.assertTrue(callable(visualize_advanced.main))

    def test_cli_help_exits_zero(self):
        code = "import sys; sys.argv = ['cogito', '--help']; import cogito; cogito.main()"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)


if __name__ == "__main__":
    unittest.main()
