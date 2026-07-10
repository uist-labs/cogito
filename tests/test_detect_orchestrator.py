#!/usr/bin/env python3
"""TDD tests for cogito_detect.detect() -- the probe orchestrator.

detect() takes injectable ``which`` and ``run`` callables (and optional system/machine
overrides) so these tests simulate every hardware shape with zero real subprocess or
hardware. Stdlib unittest only.

Run with:  uv run python -m unittest tests.test_detect_orchestrator
"""

import pathlib
import unittest

import cogito_detect as d

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text()


def make_env(tools):
    """Build (which, run) fakes from a dict: tool-name -> output text (present) .

    A tool absent from the dict is treated as not installed.
    """
    def which(name):
        return f"/usr/bin/{name}" if name in tools else None

    def run(cmd):
        name = cmd[0]
        if name not in tools:
            return ""
        # nvidia-smi is called twice (banner + compute_cap query); disambiguate.
        if name == "nvidia-smi" and "--query-gpu=compute_cap" in cmd:
            return tools.get("nvidia-smi:compute_cap", "")
        return tools[name]

    return which, run


class TestDetectNvidia(unittest.TestCase):
    def test_real_1070(self):
        which, run = make_env({
            "nvidia-smi": fixture("nvidia_smi_gtx1070.txt"),
            "nvidia-smi:compute_cap": fixture("compute_cap_6_1.txt"),
        })
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64")
        self.assertTrue(det.nvidia)
        self.assertEqual(det.nvidia_compute_cap, 6.1)
        self.assertEqual(det.nvidia_max_cuda, (13, 0))
        self.assertEqual(det.nvidia_driver, "580.173.02")
        self.assertFalse(det.amd_gpu)

    def test_nvidia_smi_present_but_no_device_is_not_nvidia(self):
        # Driver tool installed but errors out (no GPU) -> nothing parses -> not nvidia.
        which, run = make_env({"nvidia-smi": "No devices were found"})
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64")
        self.assertFalse(det.nvidia)


class TestDetectAmd(unittest.TestCase):
    def test_amd_with_rocm(self):
        which, run = make_env({
            "rocminfo": "Agent 2\n  Name: gfx1100\n  Device Type:  GPU\n",
            "lspci": fixture("lspci_amd.txt"),
        })
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64")
        self.assertTrue(det.amd_gpu)
        self.assertTrue(det.rocm_ok)

    def test_amd_without_rocm(self):
        # No rocminfo tool; lspci still shows the AMD GPU.
        which, run = make_env({"lspci": fixture("lspci_amd.txt")})
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64")
        self.assertTrue(det.amd_gpu)
        self.assertFalse(det.rocm_ok)


class TestDetectVulkanAndPlatform(unittest.TestCase):
    def test_vulkan_tool_present(self):
        which, run = make_env({"vulkaninfo": "Vulkan Instance Version: 1.3"})
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64")
        self.assertTrue(det.vulkan)

    def test_apple_silicon(self):
        which, run = make_env({})
        det = d.detect(which=which, run=run, system="Darwin", machine="arm64")
        self.assertEqual(det.system, "Darwin")
        self.assertEqual(det.machine, "arm64")
        self.assertFalse(det.nvidia)
        self.assertFalse(det.amd_gpu)

    def test_bare_cpu_linux_has_glibc(self):
        which, run = make_env({})
        det = d.detect(
            which=which, run=run, system="Linux", machine="x86_64",
            libc=("glibc", "2.39"),
        )
        self.assertEqual(det.glibc, (2, 39))
        self.assertFalse(det.nvidia)
        self.assertFalse(det.vulkan)


if __name__ == "__main__":
    unittest.main()
