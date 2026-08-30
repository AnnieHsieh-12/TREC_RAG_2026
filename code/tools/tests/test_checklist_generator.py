import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "sidecar" / "src" / "checklist.py"
)
SPEC = importlib.util.spec_from_file_location("checklist_generator", MODULE_PATH)
checklist_generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checklist_generator)


class FakeClient:
    def chat(self, **_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"sub_aspects": ['
                            '{"title":"Primary aspect","importance":"vital"},'
                            '{"title":"Secondary aspect","importance":"okay"},'
                            '{"title":"Third aspect","importance":"vital"}'
                            "]}"
                        )
                    }
                }
            ]
        }


class ChecklistGeneratorTests(unittest.TestCase):
    def test_generator_emits_controller_items_without_expected_facts(self):
        record = checklist_generator.generate_checklist(
            "topic-1", "Example narrative", FakeClient(), "test-model"
        )
        self.assertEqual(
            record,
            {
                "qid": "topic-1",
                "items": [
                    "Primary aspect (vital)",
                    "Secondary aspect",
                    "Third aspect (vital)",
                ],
            },
        )
        self.assertNotIn("expected_facts", checklist_generator.DECOMPOSE_USER_TEMPLATE)
        self.assertIn(
            "Every explicitly named domain",
            checklist_generator.DECOMPOSE_USER_TEMPLATE,
        )
        self.assertIn(
            'must be marked "vital"',
            checklist_generator.DECOMPOSE_USER_TEMPLATE,
        )


if __name__ == "__main__":
    unittest.main()
