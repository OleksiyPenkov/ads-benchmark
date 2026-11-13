import importlib
import json
import tempfile
import unittest
from pathlib import Path


class InterruptedAccountingTests(unittest.TestCase):
    def test_conflicting_native_usage_blocks_further_admission(self):
        from campaign_core import Budget
        budget = Budget(50)
        budget.record("trial/response", 2)
        self.assertTrue(budget.can_schedule)
        operation = getattr(budget, "note_failure", None)
        self.assertIsNotNone(operation, "Native accounting failures must block scheduling")
        operation("conflicting_native_message_duplicate")
        self.assertFalse(budget.can_schedule)
        self.assertEqual(budget.total, 2)

    def test_native_cost_survives_interruption_before_derived_usage_record(self):
        core = importlib.import_module("campaign_core")
        recovery = getattr(core, "iter_native_usage", None)
        self.assertIsNotNone(recovery, "Interrupted native message usage must be recoverable")
        from campaign_worker import Journal
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trial.jsonl"
            writer = Journal(path, "xb")
            writer.emit({"type": "message_end", "messageId": "response-one", "message": {"role": "assistant", "provider": "deepseek", "model": "deepseek-v4-pro", "usage": {"cost": {"total": 0.25}}, "content": [], "timestamp": 1}})
            writer.close()
            self.assertEqual(dict(recovery(path, "B01_deepseek__deepseek-v4-pro")), {"B01_deepseek__deepseek-v4-pro/response-one": 0.25})

    def test_duplicate_native_frames_do_not_double_count_and_conflicts_fail(self):
        core = importlib.import_module("campaign_core")
        recovery = getattr(core, "iter_native_usage", None)
        self.assertIsNotNone(recovery, "Native recovery must enforce stable response identities")
        from campaign_worker import Journal
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trial.jsonl"
            frame = {"type": "message_end", "messageId": "same-id", "message": {"role": "assistant", "usage": {"cost": {"total": 0.25}}, "content": [], "timestamp": 1}}
            writer = Journal(path, "xb")
            writer.emit(frame)
            writer.emit(frame)
            writer.close()
            self.assertEqual(sum(cost for _key, cost in recovery(path, "trial")), 0.25)
            writer = Journal(path, "ab")
            frame["message"]["usage"]["cost"]["total"] = 0.5
            writer.emit(frame)
            writer.close()
            with self.assertRaises(ValueError):
                list(recovery(path, "trial"))


if __name__ == "__main__":
    unittest.main()
