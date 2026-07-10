#!/usr/bin/env python3
"""TDD tests for cogito_detect.recommend() -- the backend decision (crown jewel).

Pure: synthetic Detection in, (key, rationale, caveats, source_build_key) out. The
headline case is the Pascal regression -- a GTX 1070 whose driver advertises CUDA 13.0
must still get cu124, because CUDA 13 dropped pre-Turing GPU support. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_recommend
"""

import unittest

import cogito_detect as d


def rec(**kw):
    return d.recommend(d.Detection(**kw))


class TestApple(unittest.TestCase):
    def test_apple_silicon_metal(self):
        self.assertEqual(rec(system="Darwin", machine="arm64").key, "metal")

    def test_intel_mac_falls_back_to_cpu(self):
        self.assertEqual(rec(system="Darwin", machine="x86_64").key, "cpu")


class TestNvidia(unittest.TestCase):
    def test_pascal_1070_with_cuda13_driver_gets_cu124_REGRESSION(self):
        # The wks1 GTX 1070: Pascal CC 6.1, driver max-CUDA 13.0, modern glibc.
        # CUDA 13 dropped pre-Turing, so cu130 is wrong -> must recommend cu124.
        r = rec(nvidia=True, nvidia_compute_cap=6.1,
                nvidia_max_cuda=(13, 0), glibc=(2, 39))
        self.assertEqual(r.key, "cu124")
        self.assertIsNone(r.source_build_key)
        self.assertIn("6.1", r.rationale)

    def test_turing_plus_with_cuda13_driver_gets_cu130(self):
        r = rec(nvidia=True, nvidia_compute_cap=8.6,
                nvidia_max_cuda=(13, 0), glibc=(2, 35))
        self.assertEqual(r.key, "cu130")

    def test_turing_plus_but_driver_maxes_at_cuda12_gets_cu124(self):
        r = rec(nvidia=True, nvidia_compute_cap=8.6,
                nvidia_max_cuda=(12, 4), glibc=(2, 35))
        self.assertEqual(r.key, "cu124")

    def test_compute_cap_unreadable_is_conservative_cu124(self):
        r = rec(nvidia=True, nvidia_compute_cap=None,
                nvidia_max_cuda=(13, 0), glibc=(2, 39))
        self.assertEqual(r.key, "cu124")

    def test_nvidia_on_old_glibc_gets_cpu_with_source_build_option(self):
        r = rec(nvidia=True, nvidia_compute_cap=6.1,
                nvidia_max_cuda=(13, 0), glibc=(2, 17))
        self.assertEqual(r.key, "cpu")
        self.assertEqual(r.source_build_key, "cu124")
        self.assertTrue(any("source" in c.lower() for c in r.caveats))

    def test_nvidia_unknown_glibc_is_conservative_cpu(self):
        r = rec(nvidia=True, nvidia_compute_cap=6.1,
                nvidia_max_cuda=(13, 0), glibc=None)
        self.assertEqual(r.key, "cpu")


class TestAmd(unittest.TestCase):
    def test_amd_with_rocm_and_modern_glibc_gets_rocm72(self):
        r = rec(amd_gpu=True, rocm_ok=True, glibc=(2, 35))
        self.assertEqual(r.key, "rocm72")

    def test_amd_rocm_but_old_glibc_falls_to_vulkan(self):
        r = rec(amd_gpu=True, rocm_ok=True, glibc=(2, 17), vulkan=True)
        self.assertEqual(r.key, "vulkan")

    def test_amd_without_rocm_but_vulkan_present_gets_vulkan(self):
        r = rec(amd_gpu=True, rocm_ok=False, vulkan=True, glibc=(2, 35))
        self.assertEqual(r.key, "vulkan")

    def test_amd_without_rocm_or_vulkan_gets_cpu(self):
        r = rec(amd_gpu=True, rocm_ok=False, vulkan=False)
        self.assertEqual(r.key, "cpu")


class TestFallback(unittest.TestCase):
    def test_all_unknown_gets_cpu(self):
        r = rec()
        self.assertEqual(r.key, "cpu")

    def test_every_recommendation_names_a_known_backend_key(self):
        # Guard against typo'd keys drifting from the catalog's six.
        known = {"cpu", "cu124", "cu130", "metal", "vulkan", "rocm72"}
        for det in [
            d.Detection(system="Darwin", machine="arm64"),
            d.Detection(nvidia=True, nvidia_compute_cap=6.1,
                        nvidia_max_cuda=(13, 0), glibc=(2, 39)),
            d.Detection(amd_gpu=True, rocm_ok=True, glibc=(2, 35)),
            d.Detection(),
        ]:
            self.assertIn(d.recommend(det).key, known)


if __name__ == "__main__":
    unittest.main()
