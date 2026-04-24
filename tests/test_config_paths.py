from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.core.config import get_settings


class ConfigPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_app_data_dir = os.environ.get("APP_DATA_DIR")
        get_settings.cache_clear()

    def tearDown(self) -> None:
        if self.original_app_data_dir is None:
            os.environ.pop("APP_DATA_DIR", None)
        else:
            os.environ["APP_DATA_DIR"] = self.original_app_data_dir
        get_settings.cache_clear()

    def test_runtime_data_dir_overrides_mutable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["APP_DATA_DIR"] = tmpdir
            get_settings.cache_clear()
            settings = get_settings()

            self.assertEqual(settings.runtime_data_dir, Path(tmpdir))
            self.assertEqual(settings.uploads_dir, Path(tmpdir) / "uploads")
            self.assertEqual(settings.logs_dir, Path(tmpdir) / "logs")
            self.assertEqual(
                settings.rag_vector_store_dir,
                Path(tmpdir) / "rag_data" / "vector_store",
            )


if __name__ == "__main__":
    unittest.main()
