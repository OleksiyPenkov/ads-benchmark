import importlib
import math
import unittest


class CampaignBoundaryTests(unittest.TestCase):
    def core(self):
        try:
            return importlib.import_module("campaign_core")
        except ModuleNotFoundError:
            self.fail("Campaign safety and budget behavior has not been implemented")

    def guard(self):
        declarations = [{"name": "get_item", "parameters": {"type": "object", "properties": {"item_id": {"type": "integer"}}, "required": ["item_id"], "additionalProperties": False}}]
        return self.core().ToolGuard(["get_item"], declarations)

    def test_mixed_read_write_batch_is_denied_before_any_read(self):
        guard = self.guard()
        calls = [{"id": "read", "name": "get_item", "arguments": {"item_id": 685}}, {"id": "write", "name": "update_item", "arguments": {"item_id": 685}}]
        with self.assertRaises(PermissionError):
            guard.prepare(calls, set())

    def test_boolean_item_id_and_extra_arguments_are_denied(self):
        for arguments in [{"item_id": True}, {"item_id": 685, "title": "changed"}]:
            with self.subTest(arguments=arguments):
                with self.assertRaises(PermissionError):
                    self.guard().prepare([{"id": "read", "name": "get_item", "arguments": arguments}], set())

    def test_repeated_call_identifiers_are_denied(self):
        call = {"id": "read", "name": "get_item", "arguments": {"item_id": 685}}
        for calls, previous in [([call, call], set()), ([call], {"read"})]:
            with self.assertRaises(PermissionError):
                self.guard().prepare(calls, previous)

    def test_ambiguous_json_and_nonfinite_values_are_denied(self):
        for text in ['{"item_id":685,"item_id":686}', '{"item_id":NaN}', '{"item_id":Infinity}']:
            with self.assertRaises(ValueError):
                self.core().parse_json(text)

    def test_schemas_cannot_fetch_external_references(self):
        with self.assertRaises(ValueError):
            self.core().ToolGuard(["get_item"], [{"name": "get_item", "parameters": {"$ref": "https://example.invalid/schema.json"}}])

    def test_budget_stops_admission_at_threshold_without_discarding_inflight_cost(self):
        budget = self.core().Budget(50)
        budget.record("trial-one/response-one", 49)
        self.assertTrue(budget.can_schedule)
        budget.record("trial-two/response-one", 1)
        self.assertFalse(budget.can_schedule)
        budget.record("trial-three/already-in-flight", 3)
        self.assertEqual(budget.total, 53)

    def test_duplicate_usage_is_counted_once_and_conflicting_duplicates_fail(self):
        budget = self.core().Budget(50)
        budget.record("response", 2)
        budget.record("response", 2)
        self.assertEqual(budget.total, 2)
        with self.assertRaises(ValueError):
            budget.record("response", 3)

    def test_invalid_costs_cannot_hide_spending(self):
        for cost in [-1, math.nan, math.inf, True]:
            with self.subTest(cost=cost):
                with self.assertRaises(ValueError):
                    self.core().Budget(50).record("response", cost)


    def test_optional_null_and_string_null_stripped_to_match_omp_sanitizer(self):
        declarations = [{
            "name": "aps_start_recipe",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parametersJson": {"type": ["string", "null"], "default": None},
                },
                "required": ["name"],
            },
        }]
        guard = self.core().ToolGuard(["aps_start_recipe"], declarations)
        native = guard.prepare([{"id": "c1", "name": "aps_start_recipe", "arguments": {"name": "ARM-A-test", "parametersJson": None}}], set())
        host = guard.prepare([{"id": "c1", "name": "aps_start_recipe", "arguments": {"name": "ARM-A-test"}}], set())
        self.assertEqual(native, host)
        self.assertEqual(native["c1"]["arguments"], {"name": "ARM-A-test"})

if __name__ == "__main__":
    unittest.main()
