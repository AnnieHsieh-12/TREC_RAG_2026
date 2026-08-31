import os
from pathlib import Path
import subprocess
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = CODE_ROOT / "scripts" / "build_retrieval_submissions.sh"


class BuildRetrievalSubmissionTests(unittest.TestCase):
    def test_invalid_docid_fails_without_publishing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            deep_dir = run_dir / "deepce"
            deep_dir.mkdir(parents=True)
            row = "1 Q0 bad-docid 1 1.0 fixture\n"
            (run_dir / "candidate_pool_top5000.trec").write_text(
                row, encoding="utf-8"
            )
            (deep_dir / "candidate_pool_top5000.trec").write_text(
                row, encoding="utf-8"
            )
            topics = root / "topics.tsv"
            topics.write_text("1\tExample narrative\n", encoding="utf-8")
            output = root / "published"
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=CODE_ROOT,
                env={
                    **os.environ,
                    "RUN_DIR": str(run_dir),
                    "TOPICS": str(topics),
                    "OUT": str(output),
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(
                (output / "cfda-final-unc/r_output_trec_rag_2026.tsv").exists()
            )
            self.assertFalse(
                (output / "cfda-final-deep/r_output_trec_rag_2026.tsv").exists()
            )


if __name__ == "__main__":
    unittest.main()
