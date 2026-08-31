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
            self.assertTrue(any("all 2 topics use k" in warning for warning in warnings))

    def test_invalid_retrieval_docid_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.tsv"
            run.write_text("1 Q0 bad-docid 1 1.0 fixture\n", encoding="utf-8")
            errors, _ = official_format_check.check_r(run, {"1"})
            self.assertTrue(any("invalid ClimbMix docid" in error for error in errors))

    def test_retrieval_rows_must_be_in_rank_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.tsv"
            run.write_text(
                "1 Q0 shard_00001_2 2 1.0 fixture\n"
                "1 Q0 shard_00001_1 1 2.0 fixture\n",
                encoding="utf-8",
            )
            errors, _ = official_format_check.check_r(run, {"1"})
            self.assertTrue(any("rows are not ordered" in error for error in errors))

    def test_rag_accepts_direct_docid_empty_citations_and_uncited_refs(self):
        value = {
            "metadata": {
                "team_id": "example-team",
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
                "team_id": "example-team",
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
            self.assertTrue(any("out of range" in error for error in errors))

    def test_non_list_citations_are_reported_without_crashing(self):
        value = {
            "metadata": {
                "team_id": "cfda",
                "run_id": "fixture",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
            },
            "references": [],
            "answer": [{"text": "Invalid citations value.", "citations": None}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            output.write_text(json.dumps(value) + "\n", encoding="utf-8")
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertIn("L1: answer[0].citations must be a list", errors)

    def test_malformed_json_types_are_reported_without_crashing(self):
        cases = [
            (None, "RAG record must be an object"),
            ([], "RAG record must be an object"),
            (
                {"metadata": None, "references": [], "answer": []},
                "metadata must be an object",
            ),
            (
                {
                    "metadata": {
                        "team_id": "cfda",
                        "run_id": "fixture",
                        "narrative_id": [],
                        "narrative": "Example narrative",
                        "run_desc": "fixture",
                    },
                    "references": [],
                    "answer": [],
                },
                "metadata.narrative_id must be a non-empty string",
            ),
            (
                {
                    "metadata": {
                        "team_id": "cfda",
                        "run_id": "fixture",
                        "narrative_id": "1",
                        "narrative": "Example narrative",
                        "run_desc": "fixture",
                    },
                    "references": [],
                    "answer": [{"text": 17, "citations": []}],
                },
                "answer[0].text must be a non-empty string",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            for value, expected in cases:
                output.write_text(json.dumps(value) + "\n", encoding="utf-8")
                errors, _ = official_format_check.check_rag(
                    output, {"1": "Example narrative"}
                )
                self.assertTrue(any(expected in error for error in errors))

    def test_citations_field_is_required_and_boolean_is_not_an_index(self):
        base = {
            "metadata": {
                "team_id": "cfda",
                "run_id": "fixture",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
            },
            "references": ["shard_00001_1", "shard_00001_2"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            missing = {**base, "answer": [{"text": "Missing citations."}]}
            output.write_text(json.dumps(missing) + "\n", encoding="utf-8")
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertIn("L1: answer[0].citations is missing", errors)

            boolean = {
                **base,
                "answer": [{"text": "Boolean citation.", "citations": [True]}],
            }
            output.write_text(json.dumps(boolean) + "\n", encoding="utf-8")
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertTrue(any("invalid citation type" in error for error in errors))

    def test_retrieval_score_must_be_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.tsv"
            for score in ("NaN", "Infinity", "-Infinity"):
                run.write_text(
                    f"1 Q0 shard_00001_1 1 {score} fixture\n", encoding="utf-8"
                )
                errors, _ = official_format_check.check_r(run, {"1"})
                self.assertTrue(any("score must be finite" in error for error in errors))

    def test_rag_rejects_nonstandard_json_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            output.write_text(
                '{"metadata":{"team_id":"cfda","narrative_id":"1",'
                '"narrative":"Example narrative","run_id":"fixture",'
                '"run_desc":"fixture"},"references":[],"answer":'
                '[{"text":"Invalid numeric value.","citations":[]}],'
                '"extra":NaN}\n',
                encoding="utf-8",
            )
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertTrue(any("invalid JSON" in error for error in errors))

    def test_rag_item_must_not_contain_multiple_sentences(self):
        value = {
            "metadata": {
                "team_id": "cfda",
                "run_id": "fixture",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
            },
            "references": [],
            "answer": [
                {"text": "First factual sentence. Second factual sentence.", "citations": []}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rag.jsonl"
            output.write_text(json.dumps(value) + "\n", encoding="utf-8")
            errors, _ = official_format_check.check_rag(
                output, {"1": "Example narrative"}
            )
            self.assertTrue(any("multiple sentences" in error for error in errors))

    def test_duplicate_topic_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            topics = Path(tmp) / "topics.tsv"
            topics.write_text(
                "1\tFirst narrative\n1\tSecond narrative\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate topic_id"):
                official_format_check.load_topics(topics)

    def test_runtime_uses_one_canonical_validator(self):
        code_root = Path(__file__).resolve().parents[2]
        canonical = code_root / "src/trec-rag-2026/shared-rag/validation.ts"
        duplicate = code_root / "rag/src/trec-rag-2026/shared-rag/validation.ts"
        self.assertTrue(canonical.is_file())
        self.assertFalse(duplicate.exists())

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
                        "team_id": "example-team",
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
                    str(code_root / "scripts/validate_outputs.sh"),
                    str(retrieval),
                    str(rag),
                    str(topics),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_rag_only_gate_rejects_missing_topics(self):
        code_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            topics = tmp_path / "topics.tsv"
            rag = tmp_path / "rag.jsonl"
            topics.write_text(
                "1\tFirst narrative\n2\tSecond narrative\n", encoding="utf-8"
            )
            rag.write_text(
                json.dumps({
                    "metadata": {
                        "team_id": "cfda",
                        "run_id": "fixture",
                        "narrative_id": "1",
                        "narrative": "First narrative",
                        "run_desc": "fixture",
                    },
                    "references": [],
                    "answer": [{"text": "Answer.", "citations": []}],
                }) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    str(MODULE_PATH),
                    "--rag-only",
                    str(rag),
                    str(topics),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing 1 topics", completed.stdout)


if __name__ == "__main__":
    unittest.main()
