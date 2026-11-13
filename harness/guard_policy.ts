// Pure tool-set policy for campaign_guard.ts. Erasable TypeScript only and no
// host imports, so `node --test` loads it without the OMP runtime.

function record(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${what} is not an object.`);
  }
  return value as Record<string, unknown>;
}

function list(value: unknown, what: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${what} is not a list.`);
  return value;
}

function text(value: unknown, what: string): string {
  if (typeof value !== "string" || value === "") throw new Error(`${what} is not a non-empty string.`);
  return value;
}

/** Prefixed names of the task's tools, in manifest order. */
export function activeToolsFor(manifest: unknown, suite: unknown, taskId: string): string[] {
  const settings = record(manifest, "Manifest");
  const prefix = text(settings.tool_prefix, "Manifest tool_prefix");
  const declared = list(settings.tools, "Manifest tools").map(tool =>
    text(record(tool, "Manifest tool").name, "Manifest tool name"));
  const task = list(record(suite, "Suite").tasks, "Suite tasks")
    .map(entry => record(entry, "Suite task"))
    .find(entry => entry.id === taskId);
  if (task === undefined) throw new Error(`Task not in suite: ${taskId}`);
  const wanted = list(task.tools, `Task ${taskId} tools`).map(name => text(name, `Task ${taskId} tool`));
  if (wanted.length === 0) throw new Error(`Task ${taskId} has no tools.`);
  for (const name of wanted) {
    if (!declared.includes(name)) throw new Error(`Task ${taskId} uses a tool outside the manifest: ${name}`);
  }
  return declared.filter(name => wanted.includes(name)).map(name => prefix + name);
}

export function capGeminiRequest(payload: unknown, expectedModel: string, cap: number) {
  const body = record(payload, "Gemini payload");
  if (body.model !== expectedModel) throw new Error("Gemini routed model mismatch.");
  const request = record(body.request, "Gemini request");
  const config = request.generationConfig === undefined ? {} :
    record(request.generationConfig, "Gemini generation configuration");
  const requested = config.maxOutputTokens === undefined ? cap : config.maxOutputTokens;
  if (typeof requested !== "number" || !Number.isSafeInteger(requested) || requested <= 0 ||
      !Number.isSafeInteger(cap) || cap <= 0) {
    throw new Error("Gemini output limit is invalid.");
  }
  const outputTokenCap = Math.min(requested, cap);
  if (config.thinkingConfig !== undefined) {
    const thinking = record(config.thinkingConfig, "Gemini thinking configuration");
    const budget = thinking.thinkingBudget;
    if (budget !== undefined && (typeof budget !== "number" || !Number.isSafeInteger(budget) ||
        budget < -1 || budget >= outputTokenCap)) {
      throw new Error("Gemini thinking budget does not fit the output limit.");
    }
  }
  return {
    payload: { ...body, model: expectedModel, request: {
      ...request, generationConfig: { ...config, maxOutputTokens: outputTokenCap },
    } },
    outputTokenCap,
  };
}

export function restoreGeminiToolSchemas(
  payload: unknown,
  frozenSchemas: Readonly<Record<string, Record<string, unknown>>>,
) {
  const body = record(payload, "Gemini payload");
  const request = record(body.request, "Gemini request");
  const tools = list(request.tools, "Gemini tools").map(tool => {
    const toolObj = record(tool, "Gemini tool");
    const declarations = list(toolObj.functionDeclarations, "Gemini declarations").map(entry => {
      const decl = record(entry, "Gemini declaration");
      const name = text(decl.name, "Gemini declaration name");
      const frozen = frozenSchemas[name];
      if (frozen === undefined) return decl;
      const current = decl.parametersJsonSchema ?? decl.parameters;
      const props = typeof current === "object" && current !== null && !Array.isArray(current) ?
        (current as Record<string, unknown>).properties : undefined;
      if (typeof props === "object" && props !== null && !Array.isArray(props) &&
          Object.keys(props as Record<string, unknown>).length > 0) {
        return decl;
      }
      const frozenProps = record(frozen.properties, `Frozen ${name} properties`);
      const frozenReq = Array.isArray(frozen.required) ? frozen.required.map(r => text(r, "Frozen required")) : [];
      const next: Record<string, unknown> = { ...decl };
      delete next.parameters;
      next.parametersJsonSchema = {
        ...frozen,
        properties: { i: { type: "string" }, ...frozenProps },
        required: frozenReq.includes("i") ? frozenReq : [...frozenReq, "i"],
      };
      return next;
    });
    return { ...toolObj, functionDeclarations: declarations };
  });
  return { ...body, request: { ...request, tools } };
}
