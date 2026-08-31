import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "deep_ce_rerank.py"
SPEC = importlib.util.spec_from_file_location("deep_ce_rerank", MODULE_PATH)
deep_ce = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deep_ce)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class DeepCeRerankTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_cache = deep_ce.CACHE
        deep_ce.CACHE = self.tempdir.name

    def tearDown(self):
        deep_ce.CACHE = self.old_cache
        self.tempdir.cleanup()

    def test_live_fetch_populates_shared_cache(self):
        response = FakeResponse({"doc": {"text": "retrieved document text"}})
        with patch.object(deep_ce.urllib.request, "urlopen", return_value=response):
            self.assertTrue(deep_ce.fetch_doc("shard_00001_1", "secret", 0))
        self.assertEqual(deep_ce.read_doc("shard_00001_1"), "retrieved document text")

    def test_cached_document_avoids_network(self):
        deep_ce.write_doc("shard_00001_2", True, "cached text")
        with patch.object(deep_ce.urllib.request, "urlopen") as urlopen:
            self.assertTrue(deep_ce.fetch_doc("shard_00001_2", "secret", 0))
        urlopen.assert_not_called()

    def test_excessive_missing_text_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing text 99/100"):
            deep_ce.require_coverage(1, 99, 0.01)


if __name__ == "__main__":
    unittest.main()
