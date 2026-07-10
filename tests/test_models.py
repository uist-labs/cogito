#!/usr/bin/env python3
"""TDD tests for cogito_models -- the curated GGUF catalog (code-as-config).

Pure data + accessors; no hardware, no network. Structural invariants only --
exact size_bytes/sha256 are pinned live at build time, so we assert shape, not
byte counts. Stdlib unittest, matching the rest of tests/.

Run with:  uv run python -m unittest tests.test_models
"""

import re
import unittest

import cogito_models as m

EXPECTED_KEYS = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "mistral-7b",
    "phi-3.5-mini",
    "deepseek-r1-qwen-7b",
]


class TestAccessors(unittest.TestCase):
    def test_keys_are_the_curated_catalog_in_order(self):
        self.assertEqual(m.keys(), EXPECTED_KEYS)

    def test_by_key_round_trips(self):
        for key in EXPECTED_KEYS:
            self.assertEqual(m.by_key(key).key, key)

    def test_by_key_unknown_returns_none(self):
        self.assertIsNone(m.by_key("no-such-model"))

    def test_iter_models_yields_every_entry(self):
        self.assertEqual([mod.key for mod in m.iter_models()], EXPECTED_KEYS)


class TestDownloadUrl(unittest.TestCase):
    def test_url_is_the_hf_resolve_path(self):
        model = m.Model(
            key="x", name="X", params_b=1.0, hf_repo="acme/X-GGUF",
            hf_file="x-q4.gguf", quant="Q4_K_M", size_bytes=100, n_layers=10,
        )
        self.assertEqual(
            m.download_url(model),
            "https://huggingface.co/acme/X-GGUF/resolve/main/x-q4.gguf",
        )


class TestCatalogInvariants(unittest.TestCase):
    def test_keys_are_unique(self):
        keys = m.keys()
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_entry_is_well_formed(self):
        for mod in m.iter_models():
            with self.subTest(key=mod.key):
                self.assertTrue(mod.hf_repo, "hf_repo must be set")
                self.assertTrue(mod.hf_file.endswith(".gguf"), "hf_file is a .gguf")
                self.assertGreater(mod.params_b, 0)
                self.assertGreater(mod.n_layers, 0)
                self.assertTrue(mod.quant, "quant must be set")
                self.assertGreaterEqual(mod.size_bytes, 0)

    def test_repos_are_ungated_providers(self):
        # The whole auth-free story depends on curating ungated repos. Guard that
        # nobody adds a known-gated org (meta-llama / google) without noticing.
        gated = ("meta-llama/", "google/")
        for mod in m.iter_models():
            with self.subTest(key=mod.key):
                self.assertFalse(
                    mod.hf_repo.startswith(gated),
                    f"{mod.hf_repo} looks gated; catalog must stay auth-free",
                )

    def test_sha256_is_none_or_64_hex(self):
        for mod in m.iter_models():
            with self.subTest(key=mod.key):
                if mod.sha256 is not None:
                    self.assertRegex(mod.sha256, re.compile(r"^[0-9a-f]{64}$"))


if __name__ == "__main__":
    unittest.main()
