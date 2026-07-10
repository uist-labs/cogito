#!/usr/bin/env python3
"""TDD tests for visualize's headless-backend decision.

DISPLAY being set is a poor proxy for "a usable display exists" -- e.g. running
as root inherits the user's DISPLAY but not their X authorization cookie, so a
GUI backend then fails with "Authorization required". When the caller asked for
--no-show there is no window wanted at all, so Agg should be forced regardless
of DISPLAY. Stdlib unittest.

Run with:  .venv/bin/python -m unittest tests.test_visualize_backend
"""

import unittest

import visualize


class BackendChoiceTest(unittest.TestCase):
    def test_no_display_forces_agg(self):
        self.assertTrue(visualize._should_use_agg(None, no_show=False))
        self.assertTrue(visualize._should_use_agg("", no_show=False))

    def test_display_and_window_wanted_keeps_gui(self):
        self.assertFalse(visualize._should_use_agg(":0", no_show=False))

    def test_no_show_forces_agg_even_with_display(self):
        # root-with-DISPLAY-but-no-xauth, and the launcher's auto-viz, pass --no-show
        self.assertTrue(visualize._should_use_agg(":0", no_show=True))


if __name__ == "__main__":
    unittest.main()
