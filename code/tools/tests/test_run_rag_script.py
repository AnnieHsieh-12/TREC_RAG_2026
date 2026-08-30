import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_rag.sh"


class RunRagScriptTests(unittest.TestCase):
    def run_with(self, root: Path, **values: str):
        topics = root / "topics.tsv"
        checklist = root / "checklist.jsonl"
        topics.write_text("1\tExample narrative\n", encoding="utf-8")
        checklist.write_text(
            '{"qid":"1","items":["aspect"]}\n', encoding="utf-8"
        )
        env = {
            **os.environ,
            "OPENAI_API_KEY": "test-only",
            "TOPICS": str(topics),
            "CHECKLIST": str(checklist),
            "OUT": str(root / "out"),
            "SUBMISSION_OUT": str(root / "submission"),
            **values,
        }
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_shards_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.run_with(Path(tmp), SHARDS="0")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("SHARDS must be a positive integer", completed.stderr)

    def test_run_id_cannot_escape_default_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.run_with(Path(tmp), RUN_ID="../../escape")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("RUN_ID must contain only", completed.stderr)

    def test_assembly_does_not_disable_quality_gate(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("ANSWER_QUALITY_GATE=0", script)
        self.assertIn('REPLAY_OFFICIAL_W5C_REPAIR:-0', script)

    def test_openai_model_can_be_overridden(self):
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6-sol}"', script)
        self.assertIn('--llm-model "$OPENAI_MODEL"', script)


if __name__ == "__main__":
    unittest.main()
