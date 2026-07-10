#!/usr/bin/env python3
"""TDD tests for cogito_launch's parameter wizard and argv builder.

The START/tune gate, the seven Enter-default fields (validated, with a custom
genesis path), and build_argv -- the exact cogito CLI contract the launch seam
and --dry-run render. stdin is scripted; nothing runs. Stdlib unittest.

Run with:  .venv/bin/python -m unittest tests.test_launch_wizard
"""

import io
import unittest

import cogito
import cogito_launch as launch


def scripted(responses):
    it = iter(responses)
    return lambda _prompt="": next(it)


class GateTest(unittest.TestCase):
    def test_empty_starts_with_defaults(self):
        params = launch.prompt_params(launch.DEFAULTS, scripted([""]), io.StringIO())
        self.assertEqual(params, dict(launch.DEFAULTS))

    def test_start_keyword_starts_with_defaults(self):
        params = launch.prompt_params(launch.DEFAULTS, scripted(["start"]), io.StringIO())
        self.assertEqual(params, dict(launch.DEFAULTS))


class TuneTest(unittest.TestCase):
    def test_tune_all_enter_yields_defaults(self):
        # gate=tune, then 7 fields all Enter (genesis default is not custom)
        script = ["tune"] + [""] * 7
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params, dict(launch.DEFAULTS))

    def test_tune_overrides_each_field(self):
        script = ["tune", "void", "10", "8192", "128", "0.7", "0.9", "1.3"]
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params["genesis_type"], "void")
        self.assertEqual(params["cycles"], 10)
        self.assertEqual(params["context_size"], 8192)
        self.assertEqual(params["tokens_per_cycle"], 128)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["repeat_penalty"], 1.3)

    def test_custom_genesis_captures_prompt_text(self):
        script = ["tune", "custom", "What is here, really?"] + [""] * 6
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params["genesis_type"], "custom")
        self.assertEqual(params["genesis_prompt"], "What is here, really?")

    def test_bad_genesis_reprompts(self):
        script = ["tune", "not-a-genesis", "void"] + [""] * 6
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params["genesis_type"], "void")

    def test_bad_int_reprompts(self):
        # genesis Enter, cycles "abc" then "10", rest Enter
        script = ["tune", "", "abc", "10", "", "", "", "", ""]
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params["cycles"], 10)

    def test_out_of_range_top_p_reprompts(self):
        # top_p "2" (>1) then "0.9"; walk to it with Enters
        script = ["tune", "", "", "", "", "", "2", "0.9", ""]
        params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
        self.assertEqual(params["top_p"], 0.9)

    def test_genesis_menu_shown_up_front_with_custom_hint(self):
        # The field must not read as a free-text box: show the named seeds and
        # how to write your own before the first prompt.
        out = io.StringIO()
        launch.prompt_params(launch.DEFAULTS, scripted(["tune"] + [""] * 7), out)
        text = out.getvalue()
        self.assertIn("custom", text.lower())
        self.assertIn("mirror", text)
        self.assertIn("void", text)

    def test_invalid_genesis_message_names_input_and_points_at_custom(self):
        out = io.StringIO()
        launch.prompt_params(
            launch.DEFAULTS,
            scripted(["tune", "my own idea", "void"] + [""] * 6), out)
        text = out.getvalue()
        self.assertIn("my own idea", text)   # actionable: names what they typed
        self.assertIn("custom", text.lower())  # ...and how to write their own

    def test_genesis_choices_come_from_cogito(self):
        # every built-in cogito genesis is accepted by the wizard
        for key in cogito.GENESIS_PROMPTS:
            script = ["tune", key] + [""] * 6
            params = launch.prompt_params(launch.DEFAULTS, scripted(script), io.StringIO())
            self.assertEqual(params["genesis_type"], key)


class BuildArgvTest(unittest.TestCase):
    def test_defaults_produce_expected_argv(self):
        argv = launch.build_argv("/models/m.gguf", dict(launch.DEFAULTS), -1, "./logs/run_x")
        self.assertEqual(argv, [
            "--model", "/models/m.gguf",
            "--genesis-type", "mirror",
            "--cycles", "50",
            "--context-size", "16384",
            "--tokens-per-cycle", "256",
            "--temperature", "0.8",
            "--top-p", "0.95",
            "--repeat-penalty", "1.1",
            "--gpu-layers", "-1",
            "--log-dir", "./logs/run_x",
        ])

    def test_custom_genesis_includes_prompt_pair(self):
        params = dict(launch.DEFAULTS)
        params["genesis_type"] = "custom"
        params["genesis_prompt"] = "Begin, quietly."
        argv = launch.build_argv("/m/x.gguf", params, 20, "./logs/run_y")
        self.assertIn("--genesis-prompt", argv)
        i = argv.index("--genesis-prompt")
        self.assertEqual(argv[i + 1], "Begin, quietly.")

    def test_non_custom_omits_prompt_pair(self):
        argv = launch.build_argv("/m/x.gguf", dict(launch.DEFAULTS), 0, "./logs/z")
        self.assertNotIn("--genesis-prompt", argv)

    def test_gpu_layers_and_log_dir_are_threaded(self):
        argv = launch.build_argv("/m/x.gguf", dict(launch.DEFAULTS), 29, "./logs/run_z")
        self.assertEqual(argv[argv.index("--gpu-layers") + 1], "29")
        self.assertEqual(argv[argv.index("--log-dir") + 1], "./logs/run_z")


if __name__ == "__main__":
    unittest.main()
