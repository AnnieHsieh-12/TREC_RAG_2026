import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "finalize_submissions.py"


class FinalizeSubmissionsTests(unittest.TestCase):
    def test_rag_identity_arguments_are_preserved(self):
        record = {
            "metadata": {
                "team_id": "old-team",
                "run_id": "old-run",
                "narrative_id": "1",
                "narrative": "Example narrative",
                "run_desc": "fixture",
            },
            "references": [],
            "answer": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.jsonl"
            output_dir = root / "final"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--rag",
                    str(source),
                    "--team-id",
                    "public-team",
                    "--rag-tag",
                    "public-run",
                    "--outdir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            finalized = json.loads(
                (output_dir / "rag_output_trec_rag_2026.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(finalized["metadata"]["team_id"], "public-team")
            self.assertEqual(finalized["metadata"]["run_id"], "public-run")


if __name__ == "__main__":
    unittest.main()
