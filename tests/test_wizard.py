#!/usr/bin/env python3
"""TDD tests for cogito_install -- the wizard core.

All side effects are injected: a fake ``runner`` records the command that WOULD run,
``detect_fn`` returns a synthetic Detection, ``input_fn`` replays keystrokes, and
``is_installed`` fakes the current env. No real detection, install, or stdin. Stdlib
unittest.

Run with:  uv run python -m unittest tests.test_wizard
"""

import io
import unittest

import cogito_detect as d
import cogito_install as ci

PASCAL = d.Detection(
    system="Linux", machine="x86_64", nvidia=True, nvidia_driver="580.173.02",
    nvidia_max_cuda=(13, 0), nvidia_compute_cap=6.1, glibc=(2, 39),
)


class Recorder:
    def __init__(self, rc=0):
        self.cmds = []
        self.rc = rc

    def __call__(self, cmd):
        self.cmds.append(cmd)
        return self.rc


def run_wizard(argv, inputs=(), detection=PASCAL, installed=False, rc=0, isatty=True,
               verify=(True, ""), out=None):
    rec = Recorder(rc=rc)
    it = iter(inputs)
    out = out if out is not None else io.StringIO()

    def input_fn(prompt=""):
        return next(it)

    code = ci.main(
        argv,
        detect_fn=lambda: detection,
        runner=rec,
        input_fn=input_fn,
        is_installed=lambda key: installed,
        isatty=lambda: isatty,
        verify_fn=lambda key: verify,
        out=out,
    )
    return code, rec


class TestInteractiveChoice(unittest.TestCase):
    def test_enter_accepts_recommended_cu124(self):
        code, rec = run_wizard([], inputs=[""])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cu124"]])

    def test_number_selects_a_different_backend(self):
        # Floated menu: [1]=cu124 (recommended), [2]=cpu ...
        code, rec = run_wizard([], inputs=["2"])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cpu"]])

    def test_invalid_then_valid_reprompts(self):
        code, rec = run_wizard([], inputs=["nope", ""])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cu124"]])


class TestNonInteractiveFlags(unittest.TestCase):
    def test_yes_accepts_recommendation_without_input(self):
        code, rec = run_wizard(["--yes"], inputs=[])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cu124"]])

    def test_backend_flag_installs_named_backend(self):
        code, rec = run_wizard(["--backend", "cpu", "--yes"], inputs=[])
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cpu"]])

    def test_unknown_backend_flag_errors_without_running(self):
        code, rec = run_wizard(["--backend", "bogus", "--yes"], inputs=[])
        self.assertNotEqual(code, 0)
        self.assertEqual(rec.cmds, [])


class TestSwitchBackend(unittest.TestCase):
    def test_already_installed_appends_reinstall_flags(self):
        code, rec = run_wizard(["--yes"], installed=True)
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [[
            "uv", "sync", "--extra", "cu124",
            "--reinstall-package", "llama-cpp-python", "--no-cache",
        ]])


class TestNonInteractiveRobustness(unittest.TestCase):
    def test_no_tty_accepts_recommendation_without_prompting(self):
        # Piped/redirected stdin (no tty), no flags, no input queued: must not crash.
        code, rec = run_wizard([], inputs=[], isatty=False)
        self.assertEqual(code, 0)
        self.assertEqual(rec.cmds, [["uv", "sync", "--extra", "cu124"]])


class TestPostInstallVerification(unittest.TestCase):
    def test_verify_success_prints_next_steps(self):
        out = io.StringIO()
        code, rec = run_wizard(["--yes"], verify=(True, ""), out=out)
        self.assertEqual(code, 0)
        self.assertIn("Next steps", out.getvalue())

    def test_next_steps_points_at_the_model_picker(self):
        out = io.StringIO()
        run_wizard(["--yes"], verify=(True, ""), out=out)
        self.assertIn("cogito-model", out.getvalue())

    def test_verify_failure_reports_actionable_guidance_and_nonzero(self):
        out = io.StringIO()
        code, rec = run_wizard(
            ["--yes"],
            verify=(False, "libcudart.so.12: cannot open shared object file"),
            out=out,
        )
        self.assertNotEqual(code, 0)
        text = out.getvalue()
        # The install ran, but we must NOT claim success.
        self.assertNotIn("Next steps", text)
        # Actionable: name the runtime and a concrete fix for cu12.
        self.assertIn("nvidia-cuda-runtime-cu12", text)
        self.assertIn("cuda-downloads", text)


class TestDryRunner(unittest.TestCase):
    def test_default_runner_dry_run_executes_nothing(self):
        out = io.StringIO()
        rc = ci._default_runner(["uv", "sync", "--extra", "cu124"],
                                dry_run=True, out=out)
        self.assertEqual(rc, 0)
        printed = out.getvalue()
        self.assertIn("uv sync --extra cu124", printed)
        self.assertIn("dry run", printed.lower())


if __name__ == "__main__":
    unittest.main()
