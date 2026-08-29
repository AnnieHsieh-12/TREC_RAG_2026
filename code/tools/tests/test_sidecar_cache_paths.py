from pathlib import Path
import sys
import tempfile
import unittest


SIDECAR_ROOT = Path(__file__).resolve().parents[3] / "sidecar"
sys.path.insert(0, str(SIDECAR_ROOT))

from src.cache_paths import doc_cache_path, raw_pool_path


class SidecarCachePathTests(unittest.TestCase):
    def test_docid_cannot_escape_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = Path(doc_cache_path(str(root), "../../../../tmp/secret")).resolve()
            self.assertTrue(path.is_relative_to(root))
            self.assertRegex(path.name, r"^[a-f0-9]{64}\.txt$")

    def test_unsafe_qid_has_no_raw_pool_path(self):
        self.assertIsNone(raw_pool_path("/cache", "../../secret", ""))
        self.assertEqual(
            raw_pool_path("/cache", "rag2026-1", "-k500"),
            "/cache/rag2026-1-k500.json",
        )


if __name__ == "__main__":
    unittest.main()
