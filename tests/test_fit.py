#!/usr/bin/env python3
"""TDD tests for the model-fit decision (cogito_models.fit + helpers).

Pure: (Model, Detection) -> Fit. No hardware. Table-driven across VRAM/RAM
tiers, with exact gpu_layers values locked so the estimate cannot silently
drift. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_fit
"""

import unittest

import cogito_detect as d
import cogito_models as m

EIGHT_GB = 8192      # GTX 1070 / typical 8 GB card (MiB)
TWENTYFOUR_GB = 24564  # RTX 4090/5090-class (MiB)


def model(key, size_bytes, n_layers, params_b):
    return m.Model(key=key, name=key, params_b=params_b, hf_repo="x/y-GGUF",
                   hf_file="y.gguf", quant="Q4_K_M", size_bytes=size_bytes,
                   n_layers=n_layers)


# Real catalog sizes for the two 7B/14B/32B cases.
M_7B = model("m7", 4683074240, 28, 7.0)
M_14B = model("m14", 8988110976, 48, 14.0)
M_32B = model("m32", 19851336576, 64, 32.0)


class TestFitGpu(unittest.TestCase):
    def test_7b_fits_fully_on_an_8gb_card(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        fit = m.fit(M_7B, det)
        self.assertEqual(fit.tier, "gpu")
        self.assertEqual(fit.gpu_layers, -1)

    def test_32b_fits_fully_on_a_24gb_card(self):
        det = d.Detection(vram_total_mb=TWENTYFOUR_GB, ram_total_mb=64000)
        self.assertEqual(m.fit(M_32B, det).tier, "gpu")


class TestFitPartial(unittest.TestCase):
    def test_14b_partial_on_8gb_with_a_clamped_layer_estimate(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        fit = m.fit(M_14B, det)
        self.assertEqual(fit.tier, "partial")
        # floor(48 * 8589934592 / int(8988110976 * 1.2)) = 38, clamped to [1, 47].
        self.assertEqual(fit.gpu_layers, 38)

    def test_partial_layers_stay_within_bounds(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        fit = m.fit(M_32B, det)
        self.assertEqual(fit.tier, "partial")
        self.assertGreaterEqual(fit.gpu_layers, 1)
        self.assertLessEqual(fit.gpu_layers, M_32B.n_layers - 1)


class TestFitCpu(unittest.TestCase):
    def test_7b_runs_on_cpu_when_no_gpu(self):
        det = d.Detection(vram_total_mb=None, ram_total_mb=15719)
        fit = m.fit(M_7B, det)
        self.assertEqual(fit.tier, "cpu")
        self.assertEqual(fit.gpu_layers, 0)

    def test_32b_exceeds_ram_is_cpu_oversized(self):
        det = d.Detection(vram_total_mb=None, ram_total_mb=15719)
        self.assertEqual(m.fit(M_32B, det).tier, "cpu_oversized")

    def test_unknown_memory_is_conservative_cpu(self):
        det = d.Detection(vram_total_mb=None, ram_total_mb=None)
        fit = m.fit(M_7B, det)
        self.assertEqual(fit.tier, "cpu")
        self.assertEqual(fit.gpu_layers, 0)


class TestRecommendedModel(unittest.TestCase):
    def test_8gb_recommends_largest_full_gpu_fit(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        rec = m.recommended_model(list(m.iter_models()), det)
        self.assertEqual(m.fit(rec, det).tier, "gpu")
        self.assertEqual(rec.params_b, 7.0)

    def test_24gb_recommends_the_32b(self):
        det = d.Detection(vram_total_mb=TWENTYFOUR_GB, ram_total_mb=64000)
        rec = m.recommended_model(list(m.iter_models()), det)
        self.assertEqual(rec.key, "qwen2.5-32b")


class TestStepDown(unittest.TestCase):
    def test_32b_steps_down_to_the_largest_smaller_fit(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        catalog = list(m.iter_models())
        chosen = m.by_key("qwen2.5-32b")
        nxt = m.step_down(catalog, det, chosen)
        self.assertIsNotNone(nxt)
        self.assertLess(nxt.params_b, chosen.params_b)
        # 14B is partial (acceptable) on 8 GB -> the closest usable step down.
        self.assertEqual(nxt.key, "qwen2.5-14b")

    def test_smallest_model_has_no_step_down(self):
        det = d.Detection(vram_total_mb=EIGHT_GB, ram_total_mb=15719)
        catalog = list(m.iter_models())
        chosen = m.by_key("qwen2.5-0.5b")
        self.assertIsNone(m.step_down(catalog, det, chosen))


if __name__ == "__main__":
    unittest.main()
