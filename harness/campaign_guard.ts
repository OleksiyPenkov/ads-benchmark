import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { activeToolsFor, capGeminiRequest, restoreGeminiToolSchemas } from "./guard_policy.ts";

// The campaign directory comes from BENCH_CAMPAIGN_DIR; only the harness is fixed.
const HARNESS = resolve(fileURLToPath(new URL(".", import.meta.url))).replaceAll("\\", "/");
const GUARD = `${HARNESS}/campaign_guard.ts`;
const POLICY = `${HARNESS}/guard_policy.ts`;
const SUITE = "inputs/suite.json";
const TOOLS = "inputs/tools.json";
const PROMPT = "inputs/system_prompt.md";
const REQUIRED_INPUTS = [SUITE, TOOLS, PROMPT, "inputs/runner-config.yml"];

interface ModelSelection {
  selector: string;
  thinking: string;
}

interface Manifest {
  models: ModelSelection[];
  max_output_tokens: number;
  tool_prefix: string;
  tools: { name: string; cls: "read" | "write" | "local" }[];
  input_sha256: Record<string, string>;
  runner_sha256: Record<string, string>;
}

interface Environment {
  model: string;
  fingerprint: string;
  thinking: string;
  directory: string;
  task: string;
}

interface LiveModel {
  provider: string;
  id: string;
  api: "anthropic-messages" | "openai-completions" | "openai-codex-responses" | "google-gemini-cli";
  maxTokens: number;
  compat?: { maxTokensField?: "max_tokens" | "max_completion_tokens" };
  requestModelId?: string;
  thinking?: { effortRouting?: Record<string, string> };
}
interface FrozenFile {
  path: string;
  hash: string;
  realPath: string;
}

interface Snapshot {
  manifest: Manifest;
  environment: Environment;
  selection: ModelSelection;
  prompt: string;
  activeTools: string[];
  toolSchemas: Record<string, Record<string, unknown>>;
  files: FrozenFile[];
}

function normalizedPath(path: string): string {
  return resolve(path).replaceAll("\\", "/").toLowerCase();
}

function containedPath(root: string, path: string): boolean {
  const suffix = relative(root, path);
  return suffix !== "" && suffix !== ".." && !suffix.startsWith("..\\") &&
    !suffix.startsWith("../") && !isAbsolute(suffix);
}

function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function sameNames(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && new Set(left).size === left.length &&
    left.every(name => right.includes(name));
}

export default function campaignGuard(pi: ExtensionAPI): void {
  const z = pi.zod;
  const hashSchema = z.string().regex(/^[0-9a-f]{64}$/);
  const selectorSchema = z.string().min(3).refine(value =>
    value === value.trim() && /^[a-z0-9][a-z0-9-]*\/[^\s]+$/.test(value));
  const thinkingSchema = z.string().min(1);
  const modelSelectionSchema = z.object({
    selector: selectorSchema,
    thinking: thinkingSchema,
  }).strict();
  // Field types only: campaign-specific values (model and task counts, limits)
  // live in campaign.json and are checked by manifest.py.
  const manifestSchema = z.object({
    models: z.array(modelSelectionSchema).min(1),
    task_ids: z.array(z.string().min(1)).min(1),
    task_pattern: z.string().min(1),
    repetitions: z.number().int().positive(),
    parallelism: z.number().int().positive(),
    cost_threshold_usd: z.number().positive(),
    timeouts: z.record(z.string(), z.number().int().positive()),
    max_output_tokens: z.number().int().positive(),
    tool_prefix: z.string().min(1),
    tools: z.array(z.object({
      name: z.string().min(1),
      cls: z.enum(["read", "write", "local"]),
    }).strict()).min(1),
    mcp_server: z.object({ command: z.string().min(1) }).strict(),
    skill_snapshot: z.string().min(1),
    input_sha256: z.record(z.string(), hashSchema),
    runner_sha256: z.record(z.string(), hashSchema),
  }).passthrough();
  const environmentSchema = z.object({
    model: selectorSchema,
    fingerprint: hashSchema,
    thinking: thinkingSchema,
    directory: z.string().min(1).refine(value => isAbsolute(value)),
    task: z.string().min(1),
  }).strict();
  const toolDeclarationSchema = z.object({
    name: z.string(),
    description: z.string(),
    parameters: z.object({ type: z.literal("object") }).passthrough(),
    loadMode: z.literal("essential"),
  }).strict();
  const tokenSchema = z.number().int().positive().refine(value => Number.isFinite(value));
  const liveModelSchema = z.object({
    provider: z.string().min(1),
    id: z.string().min(1),
    api: z.enum(["anthropic-messages", "openai-completions", "openai-codex-responses", "google-gemini-cli"]),
    maxTokens: tokenSchema,
    compat: z.object({
      maxTokensField: z.enum(["max_tokens", "max_completion_tokens"]).optional(),
    }).passthrough().optional(),
    requestModelId: z.string().min(1).optional(),
    thinking: z.object({
      effortRouting: z.record(z.string(), z.string().min(1)).optional(),
    }).passthrough().optional(),
  }).passthrough();
  const payloadSchema = z.object({ model: z.string().optional() }).passthrough();
  const anthropicPayloadSchema = z.object({
    model: z.string().optional(),
    max_tokens: tokenSchema.optional(),
    thinking: z.object({
      type: z.enum(["enabled", "disabled", "adaptive"]),
      budget_tokens: tokenSchema.optional(),
    }).passthrough().optional(),
  }).passthrough();
  const completionPayloadSchema = z.object({
    model: z.string().optional(),
    max_tokens: tokenSchema.optional(),
    max_completion_tokens: tokenSchema.optional(),
  }).passthrough();


  let snapshot: Snapshot | undefined;
  let failed = false;
  let prepared = false;
  let observedRequests = 0;

  function cancel(abort: () => unknown, category: string): void {
    failed = true;
    prepared = false;
    // Do not await abort: settling the agent can depend on this handler returning.
    process.stderr.write(`BENCH_CAMPAIGN_DENIED ${category}\n`);
    try {
      Promise.resolve(abort()).catch(() => {
        process.stderr.write("BENCH_CAMPAIGN_DENIED abort_failed\n");
      });
    } catch {
      process.stderr.write("BENCH_CAMPAIGN_DENIED abort_failed\n");
    }
  }

  function environment(): Environment {
    return environmentSchema.parse({
      model: process.env.BENCH_CAMPAIGN_MODEL,
      fingerprint: process.env.BENCH_CAMPAIGN_FINGERPRINT,
      thinking: process.env.BENCH_CAMPAIGN_THINKING,
      directory: process.env.BENCH_CAMPAIGN_DIR,
      task: process.env.BENCH_CAMPAIGN_TASK,
    });
  }

  function loadSnapshot(): Snapshot {
    const expected = environment();
    if (normalizedPath(fileURLToPath(import.meta.url)) !== normalizedPath(GUARD)) {
      throw new Error("Campaign guard path mismatch.");
    }
    const campaign = resolve(expected.directory);
    const inputs = resolve(campaign, "inputs");
    const manifestPath = resolve(campaign, "campaign.json");
    if (!containedPath(campaign, realpathSync(manifestPath))) {
      throw new Error("Campaign manifest path mismatch.");
    }
    const bytes = readFileSync(manifestPath);
    if (createHash("sha256").update(bytes).digest("hex") !== expected.fingerprint) {
      throw new Error("Campaign fingerprint mismatch.");
    }
    const data: unknown = JSON.parse(bytes.toString("utf8"));
    const manifest = manifestSchema.parse(data);
    const selectors = manifest.models.map(model => model.selector);
    const toolNames = manifest.tools.map(tool => tool.name);
    if (new Set(selectors).size !== selectors.length || new Set(manifest.task_ids).size !== manifest.task_ids.length ||
      new Set(toolNames).size !== toolNames.length || !manifest.task_ids.includes(expected.task)) {
      throw new Error("Campaign selection mismatch.");
    }
    const selection = manifest.models.find(model => model.selector === expected.model);
    if (!selection || selection.thinking !== expected.thinking) {
      throw new Error("Campaign model configuration mismatch.");
    }
    for (const path of REQUIRED_INPUTS) {
      if (!Object.hasOwn(manifest.input_sha256, path)) {
        throw new Error("Required frozen input missing.");
      }
    }
    const files: FrozenFile[] = [{
      path: manifestPath, hash: expected.fingerprint, realPath: realpathSync(manifestPath),
    }];
    for (const [path, hash] of Object.entries(manifest.input_sha256)) {
      const absolute = resolve(campaign, path);
      if (isAbsolute(path) || !path.startsWith("inputs/") ||
        !containedPath(inputs, absolute) || !containedPath(inputs, realpathSync(absolute))) {
        throw new Error("Frozen input path mismatch.");
      }
      files.push({ path: absolute, hash, realPath: realpathSync(absolute) });
    }
    let ownsGuard = false;
    let ownsPolicy = false;
    for (const [path, hash] of Object.entries(manifest.runner_sha256)) {
      if (!isAbsolute(path) || !containedPath(HARNESS, path) ||
        !containedPath(HARNESS, realpathSync(path))) {
        throw new Error("Frozen runner path mismatch.");
      }
      if (normalizedPath(path) === normalizedPath(GUARD)) ownsGuard = true;
      if (normalizedPath(path) === normalizedPath(POLICY)) ownsPolicy = true;
      files.push({ path, hash, realPath: realpathSync(path) });
    }
    if (!ownsGuard || !ownsPolicy) throw new Error("Campaign guard fingerprint missing.");
    for (const file of files) {
      if (sha256File(file.path) !== file.hash) throw new Error("Frozen file changed.");
    }
    const declarationsData: unknown = JSON.parse(readFileSync(resolve(campaign, TOOLS), "utf8"));
    const declarations = z.array(toolDeclarationSchema).parse(declarationsData);
    const declared = declarations.map(tool => tool.name);
    if (declared.length !== toolNames.length || declared.some((name, index) => name !== toolNames[index])) {
      throw new Error("Frozen tool declaration mismatch.");
    }
    const toolSchemas: Record<string, Record<string, unknown>> = {};
    for (const tool of declarations) {
      toolSchemas[manifest.tool_prefix + tool.name] = tool.parameters;
    }
    const suite: unknown = JSON.parse(readFileSync(resolve(campaign, SUITE), "utf8"));
    const activeTools = activeToolsFor(manifest, suite, expected.task);
    const prompt = z.string().min(1).parse(readFileSync(resolve(campaign, PROMPT), "utf8"));
    return { manifest, environment: expected, selection, prompt, activeTools, toolSchemas, files };
  }

  function immutableSnapshot(): Snapshot {
    if (failed) throw new Error("Campaign admission already denied.");
    if (!snapshot) snapshot = loadSnapshot();
    const current = environment();
    if (current.model !== snapshot.environment.model ||
      current.fingerprint !== snapshot.environment.fingerprint ||
      current.thinking !== snapshot.environment.thinking ||
      current.directory !== snapshot.environment.directory ||
      current.task !== snapshot.environment.task) {
      throw new Error("Campaign environment changed.");
    }
    for (const file of snapshot.files) {
      if (normalizedPath(realpathSync(file.path)) !== normalizedPath(file.realPath) ||
        sha256File(file.path) !== file.hash) {
        throw new Error("Frozen campaign file changed.");
      }
    }
    return snapshot;
  }

  function selectedModel(value: unknown, frozen: Snapshot): LiveModel {
    const model = liveModelSchema.parse(value);
    if (`${model.provider}/${model.id}` !== frozen.selection.selector ||
      pi.getThinkingLevel() !== frozen.selection.thinking) {
      throw new Error("Live model configuration mismatch.");
    }
    return model;
  }

  pi.on("session_start", (_event, ctx) => {
    try {
      if (ctx.mode !== "rpc" || ctx.agent.kind !== "main") {
        throw new Error("Campaign session mismatch.");
      }
      selectedModel(ctx.model, immutableSnapshot());
    } catch {
      cancel(() => ctx.abort(), "session_admission_failed");
    }
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    try {
      if (ctx.mode !== "rpc" || ctx.agent.kind !== "main") {
        throw new Error("Campaign session mismatch.");
      }
      const frozen = immutableSnapshot();
      selectedModel(ctx.model, frozen);
      const available = new Set(pi.getAllTools().map(tool => tool.name));
      if (frozen.activeTools.some(name => !available.has(name))) {
        throw new Error("Campaign host tools missing.");
      }
      if (!sameNames(pi.getActiveTools(), frozen.activeTools)) {
        await pi.setActiveTools([...frozen.activeTools]);
      }
      if (!sameNames(pi.getActiveTools(), frozen.activeTools)) {
        throw new Error("Campaign active tools mismatch.");
      }
      prepared = true;
      // A full replacement, not an append: no generated instructions survive.
      return { systemPrompt: frozen.prompt };
    } catch {
      cancel(() => ctx.abort(), "prompt_admission_failed");
      throw new Error("Campaign prompt admission denied.");
    }
  });

  pi.on("tool_call", (event, ctx) => {
    try {
      if (!prepared || ctx.mode !== "rpc" || ctx.agent.kind !== "main") {
        throw new Error("Tool not admitted.");
      }
      const frozen = immutableSnapshot();
      if (!frozen.activeTools.includes(event.toolName)) {
        throw new Error("Tool not in the task's tool set.");
      }
      selectedModel(ctx.model, frozen);
      // Python separately validates the entire assistant batch and each dispatch.
      // The dedicated namespace blocks native MCP tools and built-in aliases.
    } catch {
      cancel(() => ctx.abort(), "tool_admission_failed");
      return { block: true, reason: "Campaign tool policy denied this tool call." };
    }
  });

  pi.on("cache_warming_decision", () => ({ action: "stop" }));
  pi.on("session_before_compact", () => ({ cancel: true }));

  pi.on("before_provider_request", (event, ctx) => {
    try {
      if (!prepared || ctx.mode !== "rpc" || ctx.agent.kind !== "main") {
        throw new Error("Provider request not admitted.");
      }
      const frozen = immutableSnapshot();
      const model = selectedModel(ctx.model, frozen);
      if (!sameNames(pi.getActiveTools(), frozen.activeTools)) {
        throw new Error("Campaign active tools changed.");
      }
      const body = payloadSchema.parse(event.payload);
      const wireModel = model.api === "google-gemini-cli" ?
        model.thinking?.effortRouting?.[frozen.selection.thinking] ?? model.requestModelId ?? model.id :
        model.id;
      if (body.model !== undefined && body.model !== wireModel) {
        throw new Error("Provider payload model mismatch.");
      }
      const cap = Math.min(frozen.manifest.max_output_tokens, model.maxTokens);
      let payload: Record<string, unknown> = body;
      let actualCap: number | null;
      switch (model.api) {
        case "anthropic-messages": {
          const request = anthropicPayloadSchema.parse(body);
          actualCap = Math.min(request.max_tokens ?? cap, cap);
          request.max_tokens = actualCap;
          const thinking = request.thinking;
          if (thinking?.type === "enabled") {
            if (frozen.selection.thinking !== "high" ||
              thinking.budget_tokens === undefined || thinking.budget_tokens < 1024 ||
              actualCap <= 1024) {
              throw new Error("Anthropic thinking budget mismatch.");
            }
            request.thinking = {
              ...thinking, budget_tokens: Math.min(thinking.budget_tokens, actualCap - 1),
            };
          } else if (thinking?.budget_tokens !== undefined) {
            throw new Error("Unexpected Anthropic thinking budget.");
          }
          payload = request;
          break;
        }
        case "openai-completions": {
          const request = completionPayloadSchema.parse(body);
          if (request.max_tokens !== undefined && request.max_completion_tokens !== undefined) {
            throw new Error("Conflicting completion output limits.");
          }
          const field = request.max_completion_tokens !== undefined ? "max_completion_tokens" :
            request.max_tokens !== undefined ? "max_tokens" :
              model.compat?.maxTokensField ?? "max_tokens";
          actualCap = Math.min(request[field] ?? cap, cap);
          request[field] = actualCap;
          payload = request;
          break;
        }
        case "google-gemini-cli": {
          const request = capGeminiRequest(body, wireModel, cap);
          payload = restoreGeminiToolSchemas(request.payload, frozen.toolSchemas);
          actualCap = request.outputTokenCap;
          break;
        }
        case "openai-codex-responses":
          // Codex rejects caller output caps. Only the host deadline bounds it.
          delete payload.max_output_tokens;
          delete payload.max_completion_tokens;
          delete payload.max_tokens;
          actualCap = null;
          break;
      }
      observedRequests += 1;
      process.stderr.write(`BENCH_CAMPAIGN_POLICY ${JSON.stringify({
        selector: frozen.selection.selector,
        model: `${model.provider}/${model.id}`,
        provider: model.provider,
        api: model.api,
        output_token_cap: actualCap,
        request: observedRequests,
      })}\n`);
      // This counter observes hook calls; it is not a transport-hard request budget.
      return payload;
    } catch {
      // Provider-hook exceptions alone are swallowed by the host. Abort explicitly.
      cancel(() => ctx.abort(), "provider_admission_failed");
      throw new Error("Campaign provider admission denied.");
    }
  });
}
