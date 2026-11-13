import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path


class CampaignRecoveryTests(unittest.TestCase):
    def configuration(self):
        return {"models": [{"selector": "deepseek/deepseek-v4-pro", "thinking": "high"}], "task_ids": ["B01"], "task_pattern": "^B[0-9]{2}$", "repetitions": 1, "parallelism": 3, "cost_threshold_usd": 50, "timeouts": {"short": 900}, "max_output_tokens": 32768, "tool_prefix": "xrc_", "tools": [{"name": "get_item", "cls": "read"}], "mcp_server": {"command": "/opt/mcp/server.exe"}, "skill_snapshot": "inputs/skill", "input_sha256": {"inputs/system_prompt.md": "unchanged"}, "runner_sha256": {"worker": "old"}}

    def test_completed_trials_survive_runner_only_recovery_without_reexecution(self):
        core = importlib.import_module("campaign_core")
        operation = getattr(core, "accepted_fingerprints", None)
        self.assertIsNotNone(operation, "Compatible completed evidence must remain reusable after a harness fix")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = self.configuration()
            path = root / "prior.json"
            path.write_text(json.dumps(prior), encoding="utf-8")
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            current = self.configuration()
            current["runner_sha256"] = {"worker": "fixed"}
            current["accepted_prior_configurations"] = [{"path": "prior.json", "sha256": fingerprint}]
            self.assertEqual(operation(root, current, "current"), {"current", fingerprint})
            current["input_sha256"]["inputs/system_prompt.md"] = "changed prompt"
            with self.assertRaises(ValueError):
                operation(root, current, "current")
            for field, value in (("timeouts", {"short": 1200}), ("tools", [{"name": "get_item", "cls": "write"}]), ("tool_prefix", "xrr_")):
                with self.subTest(field=field):
                    changed = {**self.configuration(), "runner_sha256": {"worker": "fixed"}, field: value,
                               "accepted_prior_configurations": current["accepted_prior_configurations"]}
                    with self.assertRaises(ValueError):
                        operation(root, changed, "current")

    def test_archived_incomplete_trial_cost_still_counts_against_budget(self):
        core = importlib.import_module("campaign_core")
        operation = getattr(core, "seed_recovery_costs", None)
        self.assertIsNotNone(operation, "Archived setup failures cannot disappear from spending control")
        from campaign_worker import Journal
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal_path = root / "failed.jsonl"
            trial_id = "B01_deepseek__deepseek-v4-pro"
            writer = Journal(journal_path, "xb")
            writer.emit({"type": "message_end", "messageId": "msg-2", "message": {"role": "assistant", "usage": {"cost": {"total": 0.25}}, "content": [], "timestamp": 1}})
            writer.close()
            result = {"status": "failed", "trial_id": trial_id, "fingerprint": "prior", "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(), "usage_records": [{"key": trial_id + "/msg-2", "cost": 0.25}], "catalog_estimated_cost_usd": 0.25}
            result_path = root / "failed.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            entry = {"result_file": "failed.json", "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(), "journal_file": "failed.jsonl", "journal_sha256": result["journal_sha256"]}
            budget = core.Budget(50)
            operation(root, [entry], {"prior"}, budget)
            budget.record(trial_id + "/msg-2", 0.5)
            self.assertEqual(budget.total, 0.75)
            result_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                operation(root, [entry], {"prior"}, core.Budget(50))


if __name__ == "__main__":
    unittest.main()
