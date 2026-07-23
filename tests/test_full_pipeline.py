from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import full_pipeline  # noqa: E402


class TestSendTelegramMessage(unittest.TestCase):
    def test_handles_http_error_without_raising(self):
        err = HTTPError(url="http://example.com", code=404, msg="Not Found", hdrs=None, fp=None)
        with patch("full_pipeline.urllib.request.urlopen", side_effect=err):
            self.assertIsNone(full_pipeline.send_telegram_message("test"))


if __name__ == "__main__":
    unittest.main()
