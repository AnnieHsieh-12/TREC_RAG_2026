import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

os.environ.setdefault("PYSERINI_API_TOKEN", "test-token")

from fastembed.rerank.cross_encoder import TextCrossEncoder
from src import sidecar


class SidecarRuntimeTests(unittest.TestCase):
    def test_fastembed_cross_encoder_api_is_available(self):
        self.assertEqual(TextCrossEncoder.__name__, "TextCrossEncoder")

    def test_codex_bridge_uses_read_only_guarded_execution(self):
        def fake_run(command, **kwargs):
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text('{"enough":true}', encoding="utf-8")
            self.assertIn("--sandbox", command)
            self.assertIn("read-only", command)
            self.assertIn("Do not use any tools", command[-1])
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("subprocess.run", side_effect=fake_run):
            result = sidecar.handle_llm(
                {"prompt": "Return JSON.", "model": "gpt-5.6-sol"}
            )

        self.assertEqual(result["text"], '{"enough":true}')
        self.assertIsInstance(result["calls_total"], int)


if __name__ == "__main__":
    unittest.main()
