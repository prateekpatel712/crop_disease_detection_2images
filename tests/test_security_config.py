from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_application


class SecurityConfigTests(unittest.TestCase):
    _managed_keys = {
        "API_KEY",
        "REQUIRE_API_KEY",
        "ENABLE_API_DOCS",
    }

    def setUp(self) -> None:
        self._original_env = {
            key: os.environ.get(key)
            for key in self._managed_keys
        }
        for key in self._managed_keys:
            os.environ.pop(key, None)
        get_settings.cache_clear()

    def tearDown(self) -> None:
        for key in self._managed_keys:
            os.environ.pop(key, None)
        for key, value in self._original_env.items():
            if value is not None:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_docs_are_disabled_by_default(self) -> None:
        app = create_application()

        self.assertIsNone(app.docs_url)
        self.assertIsNone(app.redoc_url)
        self.assertIsNone(app.openapi_url)

    def test_api_key_is_enforced_when_configured(self) -> None:
        os.environ["API_KEY"] = "top-secret"
        get_settings.cache_clear()
        client = TestClient(create_application())

        public_health_response = client.get("/api/v1/health")
        unauthorized_response = client.post(
            "/api/v1/feedback",
            json={"request_id": "req-1", "verdict": "correct"},
        )
        authorized_response = client.post(
            "/api/v1/feedback",
            json={"request_id": "req-1", "verdict": "correct"},
            headers={"X-API-Key": "top-secret"},
        )

        self.assertEqual(public_health_response.status_code, 200)
        self.assertEqual(unauthorized_response.status_code, 401)
        self.assertEqual(authorized_response.status_code, 200)

    def test_missing_required_api_key_raises_startup_config_error(self) -> None:
        os.environ["REQUIRE_API_KEY"] = "true"
        get_settings.cache_clear()

        with self.assertRaises(RuntimeError):
            create_application()


if __name__ == "__main__":
    unittest.main()
