import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_format_check.py"
SPEC = importlib.util.spec_from_file_location("official_format_check", MODULE_PATH)
assert SPEC and SPEC.loader
official_format_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(official_format_check)


class OfficialFormatCheckTests(unittest.TestCase):
    def test_fixed_depth_is_warning_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.tsv"
            run.write_text(
                "1 Q0 shard_00001_1 1 1.0 fixture\n"
                "2 Q0 shard_00002_2 1 1.0 fixture\n",
                encoding="utf-8",
            )
            errors, warnings = official_format_check.check_r(run, {"1", "2"})
            self.assertEqual(errors, [])
            self.assertTrue(any("同一個 k" in warning for warning in warnings))

    def test_invalid_retrieval_docid_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.tsv"
            run.write_text("1 Q0 bad-docid 1 1.0 fixture\n", encoding="utf-8")
            errors, _ = official_format_check.check_r(run, {"1"})
            self.assertTrue(any("非 ClimbMix docid" in error for error in errors))

    def test_rag_accepts_direct_docid_empty_citations_and_uncited_refs(self):
        value = {
            "metadata": {
                "team_id": "pi-serini",
                "run_id": "fixture",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
                "generator": "extra metadata is allowed",
            },
            "references": ["shard_00001_1", "shard_00002_2"],
            "answer": [
                {"text": "Supported.", "citations": ["shard_00001_1"]},
                {"text": "No citation is structurally valid.", "citations": []},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            output.write_text(json.dumps(value) + "\n", encoding="utf-8")
            errors, warnings = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertEqual((errors, warnings), ([], []))

    def test_word_limit_and_bad_citation_fail(self):
        value = {
            "metadata": {
                "team_id": "pi-serini",
                "run_id": "fixture",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
            },
            "references": ["shard_00001_1"],
            "answer": [{"text": " ".join(["word"] * 1025), "citations": [9]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            output.write_text(json.dumps(value) + "\n", encoding="utf-8")
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertTrue(any("1024" in error for error in errors))
            self.assertTrue(any("越界" in error for error in errors))

    def test_two_runtime_validators_share_the_same_official_rules(self):
        code_root = Path(__file__).resolve().parents[2]
        paths = [
            code_root / "src/trec-rag-2026/agentic-rag-baseline/validation.ts",
            code_root / "rag/src/trec-rag-2026/agentic-rag-baseline/validation.ts",
        ]
        self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_shell_gate_propagates_failure(self):
        code_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            topics = tmp_path / "topics.tsv"
            retrieval = tmp_path / "bad.tsv"
            rag = tmp_path / "rag.jsonl"
            topics.write_text("1\tExample narrative\n", encoding="utf-8")
            retrieval.write_text("1 Q0 bad-docid 1 1.0 fixture\n", encoding="utf-8")
            rag.write_text(
                json.dumps({
                    "metadata": {
                        "team_id": "pi-serini",
                        "run_id": "fixture",
                        "narrative_id": "1",
                        "narrative": "Example narrative",
                        "run_desc": "fixture",
                    },
                    "references": [],
                    "answer": [{"text": "Answer.", "citations": []}],
                }) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(code_root / "scripts/validate_submission.sh"),
                    str(retrieval),
                    str(rag),
                    str(topics),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
