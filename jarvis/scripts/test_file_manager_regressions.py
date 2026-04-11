"""Regression checks for file-manager path resolution, scanning caps, and truncation hints."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import file_manager


class FileManagerRegressionTests(unittest.TestCase):
    def test_resolve_path_prefers_runtime_context_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target = temp_path / "test.py"
            target.write_text("print('runtime cwd file')\n", encoding="utf-8")

            resolved = file_manager._resolve_path(
                "test.py",
                {"_runtime_context": {"cwd": str(temp_path)}},
            )

        self.assertEqual(resolved, target.resolve())

    def test_resolve_path_prefers_current_working_directory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target = temp_path / "test.py"
            target.write_text("print('cwd file')\n", encoding="utf-8")
            os.chdir(temp_path)
            try:
                resolved = file_manager._resolve_path("test.py")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(resolved, target.resolve())

    def test_search_text_stops_after_max_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            files: list[Path] = []
            for index in range(file_manager._MAX_FILES_TO_SCAN + 5):
                path = temp_path / f"file_{index:03d}.txt"
                text = "needle\n" if index == file_manager._MAX_FILES_TO_SCAN + 1 else "haystack\n"
                path.write_text(text, encoding="utf-8")
                files.append(path)

            with patch.object(file_manager, "_iter_candidate_files", return_value=iter(files)):
                result = file_manager._search_text(
                    temp_path,
                    "needle",
                    pattern="",
                    recursive=True,
                    max_results=25,
                    case_sensitive=False,
                    include_hidden=True,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["files_scanned"], file_manager._MAX_FILES_TO_SCAN)
        self.assertTrue(result["scan_limited"])

    def test_read_file_embeds_truncation_notice_in_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "long.txt"
            target.write_text("a" * (file_manager._READ_CHAR_LIMIT + 250), encoding="utf-8")

            result = file_manager._read_file(target, summarize=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["content_truncated"])
        self.assertIn("[FILE TRUNCATED - showing first", result["content"])
        self.assertIn(str(file_manager._READ_CHAR_LIMIT), result["content"])


if __name__ == "__main__":
    unittest.main()
