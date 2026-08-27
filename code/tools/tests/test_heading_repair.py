import copy
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_uncited_heading_repair.py"
SPEC = importlib.util.spec_from_file_location("heading_repair", MODULE_PATH)
assert SPEC and SPEC.loader
heading_repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(heading_repair)


class HeadingRepairTests(unittest.TestCase):
    def test_ordinary_uncited_sentence_is_not_merged(self):
        record = {
            "metadata": {"narrative_id": "new-run-topic"},
            "answer": [
                {"text": "An ordinary uncited sentence.", "citations": []},
                {"text": "A supported sentence.", "citations": [0]},
            ],
        }
        expected = copy.deepcopy(record)
        repaired, merges = heading_repair.repair(record)
        self.assertEqual(merges, 0)
        self.assertEqual(repaired, expected)

    def test_manifest_requires_exact_text_hash(self):
        qid, index, digest = next(iter(heading_repair.OFFICIAL_HEADINGS))
        record = {
            "metadata": {"narrative_id": qid},
            "answer": [
                {"text": f"filler {position}", "citations": [0]}
                for position in range(index)
            ]
            + [
                {"text": "not the frozen heading", "citations": []},
                {"text": "A supported sentence.", "citations": [0]},
            ],
        }
        self.assertNotEqual(
            heading_repair.hashlib.sha256(
                record["answer"][index]["text"].encode("utf-8")
            ).hexdigest(),
            digest,
        )
        repaired, merges = heading_repair.repair(record)
        self.assertEqual(merges, 0)
        self.assertEqual(len(repaired["answer"]), index + 2)


if __name__ == "__main__":
    unittest.main()
