"""Regression coverage for image generation tool and viewer task wiring."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import image_gen_tool
import server


class ImageGenToolRegressionTests(unittest.TestCase):
    def test_execute_requires_prompt(self) -> None:
        result = image_gen_tool.execute({})
        self.assertFalse(result["ok"])
        self.assertIn("prompt is required", result["error"])

    def test_generate_nvidia_saves_base64_artifact_and_returns_viewer_url(self) -> None:
        image_bytes = b"fake-image-bytes"
        payload = {"artifacts": [{"base64": base64.b64encode(image_bytes).decode("ascii")}]}

        mock_response = Mock()
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_response.headers.get.return_value = "application/json"
        mock_cm = Mock()
        mock_cm.__enter__ = Mock(return_value=mock_response)
        mock_cm.__exit__ = Mock(return_value=False)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "tools.image_gen_tool.SCREENSHOTS_DIR",
            Path(temp_dir),
        ), patch("tools.image_gen_tool.urlopen", return_value=mock_cm):
            result = image_gen_tool._generate_nvidia(
                prompt="city skyline",
                styled_prompt="city skyline, cinematic",
                style="cinematic",
                size="landscape",
                width=1344,
                height=768,
                api_key="test-key",
            )
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["path"]).exists())
            self.assertEqual(Path(result["path"]).read_bytes(), image_bytes)
            self.assertTrue(str(result["url"]).startswith("/generated/generated_"))


class ImageTaskServerRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        server._TASKS.clear()

    def tearDown(self) -> None:
        server._TASKS.clear()

    def test_complete_image_task_marks_task_ready_for_viewer(self) -> None:
        task = server._start_image_task("city skyline")

        server._complete_image_task(
            str(task["task_id"]),
            {
                "ok": True,
                "url": "/generated/generated_123.png",
                "path": r"C:\tmp\generated_123.png",
                "prompt": "city skyline",
                "provider": "nvidia",
                "model": "black-forest-labs/flux.1-schnell",
                "style": "cinematic",
                "size": "landscape",
            },
        )

        stored = server._TASKS[str(task["task_id"])]
        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["result"]["type"], "image")
        self.assertEqual(stored["result"]["url"], "/generated/generated_123.png")

    def test_complete_image_task_records_failure(self) -> None:
        task = server._start_image_task("city skyline")

        server._complete_image_task(str(task["task_id"]), {"ok": False, "error": "provider down"})

        stored = server._TASKS[str(task["task_id"])]
        self.assertEqual(stored["status"], "failed")
        self.assertIn("provider down", stored["error"])


if __name__ == "__main__":
    unittest.main()
