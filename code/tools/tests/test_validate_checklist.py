import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_checklist.py"
SPEC = importlib.util.spec_from_file_location("validate_checklist", MODULE_PATH)
assert SPEC and SPEC.loader
validate_checklist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_checklist)


class ValidateChecklistTests(unittest.TestCase):
    def write_inputs(self, root: Path, checklist: str):
        topics = root / "topics.tsv"
        checklist_path = root / "checklist.jsonl"
        topics.write_text("1\tFirst\n2\tSecond\n", encoding="utf-8")
        checklist_path.write_text(checklist, encoding="utf-8")
        return topics, checklist_path

    def test_exact_coverage_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            topics, checklist = self.write_inputs(
                Path(tmp),
                '{"qid":"1","items":["a"]}\n'
                '{"qid":"2","items":["b"]}\n',
            )
            validate_checklist.validate(topics, checklist)

    def test_missing_extra_and_duplicate_ids_fail(self):
        cases = [
            ('{"qid":"1","items":["a"]}\n', "missing topic IDs"),
            (
                '{"qid":"1","items":["a"]}\n'
                '{"qid":"2","items":["b"]}\n'
                '{"qid":"3","items":["c"]}\n',
                "absent from topics",
            ),
            (
                '{"qid":"1","items":["a"]}\n'
                '{"qid":"1","items":["b"]}\n',
                "Duplicate checklist qid",
            ),
        ]
        for contents, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                topics, checklist = self.write_inputs(Path(tmp), contents)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_checklist.validate(topics, checklist)


if __name__ == "__main__":
    unittest.main()
