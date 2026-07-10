#!/usr/bin/env python3
"""TDD tests for cogito_detect pure parsers (the parse-text -> value layer).

These never touch hardware or subprocess -- they feed captured/synthetic text to
pure functions and assert the parsed value. Stdlib unittest only (no pytest dep),
matching tests/test_smoke.py.

Run with:  uv run python -m unittest tests.test_detect_parsers
"""

import pathlib
import unittest

import cogito_detect as d

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text()


class TestParseNvidiaSmiCuda(unittest.TestCase):
    def test_real_1070_banner_reports_cuda_13_0(self):
        self.assertEqual(
            d.parse_nvidia_smi_cuda(fixture("nvidia_smi_gtx1070.txt")), (13, 0)
        )

    def test_missing_cuda_line_returns_none(self):
        self.assertIsNone(d.parse_nvidia_smi_cuda("no cuda here"))

    def test_empty_returns_none(self):
        self.assertIsNone(d.parse_nvidia_smi_cuda(""))


class TestParseDriverVersion(unittest.TestCase):
    def test_real_1070_banner(self):
        self.assertEqual(
            d.parse_driver_version(fixture("nvidia_smi_gtx1070.txt")), "580.173.02"
        )

    def test_missing_returns_none(self):
        self.assertIsNone(d.parse_driver_version("nothing"))


class TestParseComputeCap(unittest.TestCase):
    def test_real_1070_is_pascal_6_1(self):
        self.assertEqual(d.parse_compute_cap(fixture("compute_cap_6_1.txt")), 6.1)

    def test_turing_plus_8_6(self):
        self.assertEqual(d.parse_compute_cap(fixture("compute_cap_8_6.txt")), 8.6)

    def test_first_gpu_wins_on_multi_gpu(self):
        # Multiple GPUs: take the first line's value (single-GPU is the norm).
        self.assertEqual(d.parse_compute_cap("7.5\n8.9\n"), 7.5)

    def test_garbage_returns_none(self):
        self.assertIsNone(d.parse_compute_cap("N/A\n"))

    def test_empty_returns_none(self):
        self.assertIsNone(d.parse_compute_cap(""))


class TestParseGlibc(unittest.TestCase):
    def test_bare_version(self):
        self.assertEqual(d.parse_glibc("2.39"), (2, 39))

    def test_confstr_form(self):
        self.assertEqual(d.parse_glibc("glibc 2.35"), (2, 35))

    def test_empty_returns_none(self):
        self.assertIsNone(d.parse_glibc(""))


class TestParseLspciAmd(unittest.TestCase):
    def test_amd_gpu_line_true(self):
        self.assertTrue(d.parse_lspci_amd(fixture("lspci_amd.txt")))

    def test_intel_gpu_line_false(self):
        self.assertFalse(d.parse_lspci_amd(fixture("lspci_intel.txt")))

    def test_empty_false(self):
        self.assertFalse(d.parse_lspci_amd(""))


class TestParseRocminfoAgents(unittest.TestCase):
    def test_gpu_agent_present_true(self):
        blob = "Agent 2\n  Name:  gfx1100\n  Device Type:  GPU\n"
        self.assertTrue(d.parse_rocminfo_agents(blob))

    def test_cpu_only_false(self):
        blob = "Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n"
        self.assertFalse(d.parse_rocminfo_agents(blob))

    def test_empty_false(self):
        self.assertFalse(d.parse_rocminfo_agents(""))


class TestDetectionRecord(unittest.TestCase):
    def test_detection_is_frozen_with_absent_defaults(self):
        det = d.Detection()
        self.assertFalse(det.nvidia)
        self.assertIsNone(det.nvidia_compute_cap)
        with self.assertRaises(Exception):
            det.nvidia = True  # frozen


if __name__ == "__main__":
    unittest.main()
