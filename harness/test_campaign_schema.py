import importlib
import unittest
import jsonschema


class PortableSchemaTests(unittest.TestCase):
    def test_explicit_array_defaults_preserve_the_original_argument_contract(self):
        core = importlib.import_module("campaign_core")
        operation = getattr(core, "portable_declarations", None)
        self.assertIsNotNone(operation, "Provider-portable schemas must preserve source argument semantics")
        declaration = {"name": "get_file_columns", "parameters": {"type": "object", "properties": {"columns": {"type": "array"}}, "required": ["columns"], "additionalProperties": False}}
        portable = operation([declaration], "xrr_")[0]
        validator = jsonschema.validators.validator_for(declaration["parameters"])
        original_validator = validator(declaration["parameters"])
        portable_validator = validator(portable["parameters"])
        for columns in [[], [0, "Substrate"], [None, {"x": 1}], True, "Substrate", 0]:
            with self.subTest(columns=columns):
                self.assertEqual(original_validator.is_valid({"columns": columns}), portable_validator.is_valid({"columns": columns}))
        guard = core.ToolGuard(["xrr_get_file_columns"], [portable])
        with self.assertRaises(PermissionError):
            guard.prepare([{"id": "write", "name": "xrr_update_item", "arguments": {}}], set())


if __name__ == "__main__":
    unittest.main()
