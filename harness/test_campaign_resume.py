import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class ResumeIntegrityTests(unittest.TestCase):
    def fixture(self, folder):
        results = folder / "results"
        results.mkdir()
        selector = "strata/qwen3.8-flash-next-iq2_xs"
        trial_id = "B01_" + selector.replace("/", "__")
        usage = [{"key": trial_id + "/message-one", "cost": 2}]
        native_answer = "24 projects; Tribology has the most items."
        rows = [
            {"type": "trial_start", "trial_id": trial_id, "fingerprint": "frozen", "task": {"id": "B01", "ask": "How many projects?"}, "selector": selector, "thinking": "off"},
            {"type": "message_end", "messageId": "message-one", "message": {"role": "assistant", "provider": "strata", "model": "qwen3.8-flash-next-iq2_xs", "content": [{"type": "text", "text": native_answer}], "stopReason": "stop", "usage": {"cost": {"total": 2}}}},
            {"type": "usage", **usage[0]},
            {"type": "prompt_result", "status": "completed", "sessionSettled": True},
            {"type": "native_session_verified", "isSettled": True},
        ]
        journal = results / (trial_id + ".jsonl")
        result = {"trial_id": trial_id, "task": "B01", "selector": selector, "thinking": "off", "status": "completed", "answer": native_answer, "fingerprint": "frozen", "catalog_estimated_cost_usd": 2, "usage_records": usage}
        rows.append({"type": "result", **result})
        from campaign_worker import Journal
        writer = Journal(journal, "xb")
        try:
            for row in rows:
                writer.emit(row)
        finally:
            writer.close()
        result["journal_sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
        sidecar = results / (trial_id + ".json")
        sidecar.write_text(json.dumps(result), encoding="utf-8")
        settings = {"task_ids": ["B01"], "models": [{"selector": selector, "thinking": "off"}]}
        return settings, sidecar, result

    def test_preserved_trial_cost_counts_toward_admission_threshold(self):
        coordinator = importlib.import_module("run_campaign")
        from campaign_core import Budget
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            settings, _sidecar, _result = self.fixture(folder)
            budget = Budget(2)
            coordinator.existing_results(folder, settings, "frozen", budget)
            self.assertEqual(budget.total, 2)
            self.assertFalse(budget.can_schedule)

    def test_rewritten_sidecar_cannot_hide_recorded_native_cost(self):
        coordinator = importlib.import_module("run_campaign")
        from campaign_core import Budget
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            settings, sidecar, result = self.fixture(folder)
            result["usage_records"][0]["cost"] = 0
            result["catalog_estimated_cost_usd"] = 0
            sidecar.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(ValueError):
                coordinator.existing_results(folder, settings, "frozen", Budget(50))

    def test_rewritten_answer_cannot_replace_the_native_final_answer(self):
        coordinator = importlib.import_module("run_campaign")
        from campaign_core import Budget
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            settings, sidecar, result = self.fixture(folder)
            result["answer"] = "Invented answer with unsupported measurements."
            sidecar.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(ValueError):
                coordinator.existing_results(folder, settings, "frozen", Budget(50))


if __name__ == "__main__":
    unittest.main()
