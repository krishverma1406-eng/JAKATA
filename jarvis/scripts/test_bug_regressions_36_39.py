"""Regression coverage for bugs 36-39."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import weather_tool, web_search
from utils.interrupts import register_keyboard_interrupt


class BugRegressionTests(unittest.TestCase):
    def test_direct_fetch_strips_html_tags_scripts_and_styles(self) -> None:
        html_doc = """
        <html><head><style>.x{color:red}</style><script>alert('x')</script></head>
        <body><h1>Heading</h1><p>Weather &amp; Climate</p></body></html>
        """

        mock_response = Mock()
        mock_response.read.return_value = html_doc.encode("utf-8")
        mock_cm = Mock()
        mock_cm.__enter__ = Mock(return_value=mock_response)
        mock_cm.__exit__ = Mock(return_value=False)

        with patch("tools.web_search.urlopen", return_value=mock_cm):
            text = web_search._direct_fetch("https://example.com")

        self.assertIn("Heading", text)
        self.assertIn("Weather & Climate", text)
        self.assertNotIn("<html>", text)
        self.assertNotIn("alert('x')", text)
        self.assertNotIn(".x{color:red}", text)

    def test_infer_location_reuses_single_memory_instance(self) -> None:
        weather_tool._MEMORY = None
        memory_instance = Mock()
        memory_instance.recall.return_value = ["I currently live in Austin, Texas."]

        with patch("tools.weather_tool.Memory", return_value=memory_instance) as memory_ctor:
            first = weather_tool._infer_location()
            second = weather_tool._infer_location()

        self.assertEqual(first, "Austin, Texas")
        self.assertEqual(second, "Austin, Texas")
        self.assertEqual(memory_ctor.call_count, 1)
        self.assertEqual(memory_instance.recall.call_count, 2)

    def test_keyboard_interrupt_needs_two_presses_to_exit(self) -> None:
        count, should_exit = register_keyboard_interrupt(0)
        self.assertEqual(count, 1)
        self.assertFalse(should_exit)

        count, should_exit = register_keyboard_interrupt(count)
        self.assertEqual(count, 2)
        self.assertTrue(should_exit)


if __name__ == "__main__":
    unittest.main()
