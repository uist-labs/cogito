#!/usr/bin/env python3
"""TDD tests for the guarded source-build path in cogito_install.

Covers cogito_install.source_build() directly (pre-flight, build, always-fall-back-to-cpu)
plus the wizard-level offer. No compiler, toolkit, or network is touched -- ``which`` and
the ``runner`` are fakes. Stdlib unittest.

Run with:  uv run python -m unittest tests.test_source_build
"""

import io
import unittest

import cogito_detect as d
import cogito_install as ci

# A full CUDA build toolchain present on an EL-style box.
FULL_CUDA = {"cc", "c++", "cmake", "make", "ninja", "nvcc", "dnf"}

BUILD_CMD = [
    "env", "CMAKE_ARGS=-DGGML_CUDA=on",
    "uv", "pip", "install", "llama-cpp-python",
    "--reinstall-package", "llama-cpp-python", "--no-cache",
]
CPU_SYNC = ["uv", "sync", "--extra", "cpu"]
INEXACT = ["uv", "sync", "--inexact"]

# NVIDIA GPU on an old-glibc box -> cpu recommended, cu124 buildable from source.
PASCAL_OLD_GLIBC = d.Detection(
    system="Linux", machine="x86_64", nvidia=True, nvidia_driver="580.173.02",
    nvidia_max_cuda=(13, 0), nvidia_compute_cap=6.1, glibc=(2, 17),
)


def which_factory(present):
    return lambda name: (f"/usr/bin/{name}" if name in present else None)


class SeqRunner:
    def __init__(self, fail_substr=None):
        self.cmds = []
        self.fail_substr = fail_substr

    def __call__(self, cmd):
        self.cmds.append(cmd)
        if self.fail_substr and self.fail_substr in " ".join(cmd):
            return 1
        return 0


class TestSourceBuildDirect(unittest.TestCase):
    def test_all_present_builds_then_inexact_sync(self):
        rec = SeqRunner()
        rc = ci.source_build("cu124", runner=rec, which=which_factory(FULL_CUDA),
                             out=io.StringIO(), system="Linux")
        self.assertEqual(rc, 0)
        self.assertEqual(rec.cmds, [BUILD_CMD, INEXACT])

    def test_missing_nvcc_links_vendor_and_falls_back_to_cpu(self):
        out = io.StringIO()
        rec = SeqRunner()
        rc = ci.source_build("cu124", runner=rec,
                             which=which_factory(FULL_CUDA - {"nvcc"}),
                             out=out, system="Linux")
        # No build attempted; only the cpu fallback ran.
        self.assertEqual(rec.cmds, [CPU_SYNC])
        text = out.getvalue().lower()
        self.assertIn("nvcc", text)
        self.assertIn("nvidia.com", text)  # vendor installer link, not a guess

    def test_missing_cmake_prints_pkg_command_and_falls_back(self):
        out = io.StringIO()
        rec = SeqRunner()
        rc = ci.source_build("cu124", runner=rec,
                             which=which_factory(FULL_CUDA - {"cmake"}),
                             out=out, system="Linux")
        self.assertEqual(rec.cmds, [CPU_SYNC])
        text = out.getvalue()
        self.assertIn("dnf install", text)
        self.assertIn("cmake", text)

    def test_build_failure_falls_back_to_cpu(self):
        out = io.StringIO()
        rec = SeqRunner(fail_substr="pip install")
        rc = ci.source_build("cu124", runner=rec, which=which_factory(FULL_CUDA),
                             out=out, system="Linux")
        self.assertEqual(rec.cmds, [BUILD_CMD, CPU_SYNC])
        self.assertIn("failed", out.getvalue().lower())

    def test_never_runs_sudo_or_package_install_itself(self):
        # The guidance is printed; the wizard must not execute a privileged install.
        rec = SeqRunner()
        ci.source_build("cu124", runner=rec, which=which_factory(FULL_CUDA - {"cmake"}),
                        out=io.StringIO(), system="Linux")
        for cmd in rec.cmds:
            self.assertNotIn("sudo", cmd)
            self.assertNotIn("dnf", cmd)


class TestWizardOffer(unittest.TestCase):
    def _run(self, inputs, installed=False):
        rec = SeqRunner()
        it = iter(inputs)
        code = ci.main(
            [], detect_fn=lambda: PASCAL_OLD_GLIBC, runner=rec,
            input_fn=lambda prompt="": next(it),
            is_installed=lambda key: installed,
            which=which_factory(FULL_CUDA),
            isatty=lambda: True, out=io.StringIO(),
        )
        return code, rec

    def test_offer_accepted_routes_to_source_build(self):
        code, rec = self._run(inputs=["y"])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [BUILD_CMD, INEXACT])

    def test_offer_declined_falls_through_to_menu_cpu(self):
        # Decline the build, then Enter accepts the cpu recommendation.
        code, rec = self._run(inputs=["n", ""])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [CPU_SYNC])


if __name__ == "__main__":
    unittest.main()
