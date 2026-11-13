import { test } from "node:test";
import assert from "node:assert/strict";
import { activeToolsFor, capGeminiRequest, restoreGeminiToolSchemas } from "./guard_policy.ts";

const manifest = {
  tool_prefix: "xrr_",
  tools: [
    { name: "list_measurements", cls: "read" },
    { name: "fit_xrr", cls: "write" },
    { name: "skill_read", cls: "local" },
  ],
};
const suite = {
  tasks: [
    { id: "X01", tools: ["list_measurements"] },
    { id: "XC1", tools: ["skill_read", "list_measurements", "fit_xrr"] },
    { id: "X99", tools: ["list_measurements", "bash"] },
  ],
};

test("returns prefixed names for X01", () => {
  assert.deepEqual(activeToolsFor(manifest, suite, "X01"), ["xrr_list_measurements"]);
});

test("throws for an unknown task", () => {
  assert.throws(() => activeToolsFor(manifest, suite, "X02"), /X02/);
});

test("throws for a task tool absent from the manifest", () => {
  assert.throws(() => activeToolsFor(manifest, suite, "X99"), /bash/);
});

test("order follows the manifest", () => {
  assert.deepEqual(activeToolsFor(manifest, suite, "XC1"),
    ["xrr_list_measurements", "xrr_fit_xrr", "xrr_skill_read"]);
});

test("Gemini output cap and schema restoration", () => {
  const payload = {
    model: "gemini-pro-agent",
    request: {
      generationConfig: { maxOutputTokens: 65535 },
      tools: [{ functionDeclarations: [{ name: "xrr_fit_xrr", parameters: { type: "object", properties: {} } }] }],
    },
  };
  const capped = capGeminiRequest(payload, "gemini-pro-agent", 32768);
  assert.equal(capped.outputTokenCap, 32768);
  const restored = restoreGeminiToolSchemas(capped.payload, {
    xrr_fit_xrr: { type: "object", properties: { structure: { type: "object" } }, required: ["structure"] },
  });
  const tools = restored.request.tools as { functionDeclarations: { parametersJsonSchema: { required: string[] } }[] }[];
  assert.deepEqual(tools[0].functionDeclarations[0].parametersJsonSchema.required, ["structure", "i"]);
});
