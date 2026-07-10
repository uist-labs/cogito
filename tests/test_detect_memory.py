#!/usr/bin/env python3
"""TDD tests for the memory-detection layer added to cogito_detect (S2).

Covers the pure parsers (parse_vram, parse_meminfo), the _ram_mb helper with
its meminfo->sysconf fallback, and detect() populating vram_total_mb /
ram_total_mb. No hardware; all side effects injected. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_detect_memory
"""

import pathlib
import unittest

import cogito_detect as d

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text()


class TestParseVram(unittest.TestCase):
    def test_single_gpu_line(self):
        self.assertEqual(d.parse_vram("8192\n"), 8192)

    def test_takes_first_gpu(self):
        self.assertEqual(d.parse_vram("8192\n24564\n"), 8192)

    def test_empty_is_none(self):
        self.assertIsNone(d.parse_vram(""))

    def test_garbage_is_none(self):
        self.assertIsNone(d.parse_vram("[N/A]"))


class TestParseMeminfo(unittest.TestCase):
    def test_real_meminfo_to_mib(self):
        # 16096556 kB // 1024 = 15719 MiB.
        self.assertEqual(d.parse_meminfo(fixture("proc_meminfo.txt")), 15719)

    def test_missing_memtotal_is_none(self):
        self.assertIsNone(d.parse_meminfo("MemFree: 100 kB\n"))

    def test_empty_is_none(self):
        self.assertIsNone(d.parse_meminfo(""))


class TestRamMb(unittest.TestCase):
    def test_uses_meminfo_when_present(self):
        ram = d._ram_mb(read=lambda: fixture("proc_meminfo.txt"))
        self.assertEqual(ram, 15719)

    def test_falls_back_to_sysconf_when_meminfo_empty(self):
        # 4 KiB pages * 4,000,000 pages = ~15625 MiB.
        fake_sysconf = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4_000_000}.get
        ram = d._ram_mb(read=lambda: "", sysconf=fake_sysconf)
        self.assertEqual(ram, 4096 * 4_000_000 // (1024 * 1024))

    def test_none_when_both_sources_fail(self):
        def boom(_name):
            raise ValueError("no sysconf")
        self.assertIsNone(d._ram_mb(read=lambda: "", sysconf=boom))


class TestDetectMemory(unittest.TestCase):
    def _env(self, tools):
        def which(name):
            return f"/usr/bin/{name}" if name in tools else None

        def run(cmd):
            name = cmd[0]
            if name not in tools:
                return ""
            if name == "nvidia-smi" and "--query-gpu=compute_cap" in cmd:
                return tools.get("nvidia-smi:compute_cap", "")
            if name == "nvidia-smi" and "--query-gpu=memory.total" in cmd:
                return tools.get("nvidia-smi:memory", "")
            return tools[name]

        return which, run

    def test_populates_vram_and_ram_on_nvidia_box(self):
        which, run = self._env({
            "nvidia-smi": fixture("nvidia_smi_gtx1070.txt"),
            "nvidia-smi:compute_cap": fixture("compute_cap_6_1.txt"),
            "nvidia-smi:memory": fixture("nvidia_memory_total_8192.txt"),
        })
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64",
                       ram_mb_fn=lambda: 15719)
        self.assertEqual(det.vram_total_mb, 8192)
        self.assertEqual(det.ram_total_mb, 15719)

    def test_no_nvidia_leaves_vram_none_but_ram_set(self):
        which, run = self._env({})
        det = d.detect(which=which, run=run, system="Linux", machine="x86_64",
                       ram_mb_fn=lambda: 15719)
        self.assertIsNone(det.vram_total_mb)
        self.assertEqual(det.ram_total_mb, 15719)


if __name__ == "__main__":
    unittest.main()
