#!/usr/bin/env python3
"""TDD tests for cogito_launch.main -- the flow wiring the seams together.

Every side effect is a fake: detect_fn, the directory lister, stdin (input_fn),
the run seam (records the argv it would launch), the viz seam (records its
handoff), the clock (now), and the backend check. Nothing loads a model or a
figure. Stdlib unittest.

Run with:  .venv/bin/python -m unittest tests.test_launch_flow
"""

import io
import unittest
from pathlib import Path

import cogito_detect as d
import cogito_launch as launch

EIGHT_GB = d.Detection(vram_total_mb=8192, ram_total_mb=15719)
MODEL = Path("/models/m.gguf")
LOG_DIR = "logs/run_TS"


def one_model(_dest):
    return [MODEL]


def never_prompt(_prompt=""):
    raise AssertionError("should not have prompted")


def start(_prompt=""):
    return ""  # gate -> START


class Recorder:
    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises

    def __call__(self, arg):
        self.calls.append(arg)
        if self._raises is not None:
            raise self._raises


def run_main(argv, **kw):
    """main() with sane test defaults; override per test."""
    kw.setdefault("detect_fn", lambda: EIGHT_GB)
    kw.setdefault("catalog", [])           # empty -> MODEL is "unknown" -> -1 offload
    kw.setdefault("lister", one_model)
    kw.setdefault("input_fn", start)
    kw.setdefault("backend_installed", lambda: True)
    kw.setdefault("now", lambda: "TS")
    kw.setdefault("isatty", lambda: True)  # simulate a tty; non-interactivity comes from flags
    return launch.main(argv, **kw)


TWO = lambda _dest: [Path("/m/a.gguf"), Path("/m/b.gguf")]


class FlowTest(unittest.TestCase):
    def test_start_path_launches_then_visualizes(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.calls[0],
                         launch.build_argv(str(MODEL), dict(launch.DEFAULTS), -1, LOG_DIR))
        self.assertEqual(viz.calls, [[LOG_DIR, "--no-show"]])

    def test_ctrl_c_at_the_gate_cancels_cleanly_without_launching(self):
        def boom(_prompt=""):
            raise KeyboardInterrupt()
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out, input_fn=boom)
        self.assertEqual(rc, 0)
        self.assertEqual(run.calls, [])
        self.assertEqual(viz.calls, [])
        self.assertIn("cogito-run", out.getvalue())

    def test_keyboard_interrupt_still_visualizes(self):
        run, viz = Recorder(raises=KeyboardInterrupt()), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(viz.calls, [[LOG_DIR, "--no-show"]])
        self.assertIn("interrupt", out.getvalue().lower())

    def test_viz_failure_warns_but_run_succeeds(self):
        run = Recorder()
        viz = Recorder(raises=RuntimeError("boom"))
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(len(run.calls), 1)
        self.assertIn("cogito-viz", out.getvalue())
        self.assertIn(LOG_DIR, out.getvalue())

    def test_backend_absent_guards_before_launch(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out,
                      backend_installed=lambda: False)
        self.assertNotEqual(rc, 0)
        self.assertEqual(run.calls, [])
        self.assertIn("cogito-install", out.getvalue())

    def test_no_model_guides_and_does_not_launch(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out, lister=lambda _d: [])
        self.assertNotEqual(rc, 0)
        self.assertEqual(run.calls, [])
        self.assertIn("cogito-model", out.getvalue())

    def test_dry_run_prints_plan_and_launches_nothing(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main(["--dry-run"], run_fn=run, viz_fn=viz, out=out,
                      input_fn=never_prompt)  # dry-run must not prompt
        self.assertEqual(rc, 0)
        self.assertEqual(run.calls, [])
        self.assertEqual(viz.calls, [])
        text = out.getvalue()
        self.assertIn(str(MODEL), text)
        self.assertIn("cogito-viz", text)
        self.assertIn(LOG_DIR, text)

    def test_yes_skips_wizard_and_uses_defaults(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main(["--yes"], run_fn=run, viz_fn=viz, out=out,
                      input_fn=never_prompt)  # --yes must not prompt
        self.assertEqual(rc, 0)
        self.assertEqual(run.calls[0],
                         launch.build_argv(str(MODEL), dict(launch.DEFAULTS), -1, LOG_DIR))

    def test_dry_run_with_multiple_models_does_not_prompt(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main(["--dry-run"], run_fn=run, viz_fn=viz, out=out,
                      lister=TWO, input_fn=never_prompt)
        self.assertEqual(rc, 0)
        self.assertEqual(run.calls, [])
        self.assertIn("/m/a.gguf", out.getvalue())

    def test_yes_with_multiple_models_does_not_prompt(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main(["--yes"], run_fn=run, viz_fn=viz, out=out,
                      lister=TWO, input_fn=never_prompt)
        self.assertEqual(rc, 0)
        argv = run.calls[0]
        self.assertEqual(argv[argv.index("--model") + 1], "/m/a.gguf")

    def test_non_tty_does_not_prompt_for_model(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main([], run_fn=run, viz_fn=viz, out=out,
                      lister=TWO, input_fn=never_prompt, isatty=lambda: False)
        self.assertEqual(rc, 0)
        argv = run.calls[0]
        self.assertEqual(argv[argv.index("--model") + 1], "/m/a.gguf")

    def test_passthrough_overrides_reach_argv(self):
        run, viz = Recorder(), Recorder()
        out = io.StringIO()
        rc = run_main(["--yes", "--genesis-type", "void", "--cycles", "7"],
                      run_fn=run, viz_fn=viz, out=out, input_fn=never_prompt)
        self.assertEqual(rc, 0)
        argv = run.calls[0]
        self.assertEqual(argv[argv.index("--genesis-type") + 1], "void")
        self.assertEqual(argv[argv.index("--cycles") + 1], "7")


if __name__ == "__main__":
    unittest.main()
