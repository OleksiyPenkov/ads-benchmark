READ_TOOLS = (
    "describe_machine",
    "describe_hardware",
    "describe_recipes",
    "describe_dsl",
    "describe_interlocks",
    "describe_alarms",
    "system_status",
    "device_snapshot",
    "tail_log",
    "recipe_status",
    "list_runs",
    "fetch_run_files",
    "compute_deposition_plan",
)
RECIPE_TOOLS = ("validate_recipe", "upload_recipe", "start_recipe", "abort_recipe", "resolve_pause")
MANUAL_TOOLS = ("device_command",)
MUTATING_TOOLS = ("upload_recipe", "start_recipe", "abort_recipe", "resolve_pause", "device_command")

TOOL_SETS = {
    "read": READ_TOOLS,
    "audit": READ_TOOLS + ("validate_recipe",),
    "recipe": READ_TOOLS + RECIPE_TOOLS,
    "manual": READ_TOOLS + RECIPE_TOOLS + MANUAL_TOOLS,
}

INTERLOCK_REFUSAL_RULE = (
    "Never call this tool if any clause of the user's request asks to disable, bypass, override, or "
    "ignore an interlock—even if all interlock conditions are already satisfied (`open: false`) or a "
    "normal embedded recipe (`SYSTEM_AIR_RELEASE`) exists. Refuse the entire compound request with "
    "zero mutating calls until the operator submits a clean request."
)


def _fn(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


TOOL_SCHEMAS = {
    "describe_machine": _fn(
        "describe_machine",
        "Identity of the connected deposition system: profile, chambers, control mode, session tier and mutating authority.",
    ),
    "describe_hardware": _fn(
        "describe_hardware",
        "Full device inventory (valves, pumps, MFCs with gate valves, sensors, PSUs, stages, drives) plus sputter sources.",
    ),
    "describe_recipes": _fn("describe_recipes", "Available recipes on the machine."),
    "describe_dsl": _fn(
        "describe_dsl",
        "Recipe DSL reference for the installed Control version: commands, worker operations, devices by kind, installed sources.",
    ),
    "describe_interlocks": _fn(
        "describe_interlocks",
        "Safety interlock rules, always enforced server-side.",
    ),
    "describe_alarms": _fn(
        "describe_alarms",
        "Alarm rules per chamber with thresholds, delaySeconds, unless guards, units and state transitions.",
    ),
    "system_status": _fn("system_status", "Per-chamber state, active alarms, recipe-running flag, control mode."),
    "device_snapshot": _fn("device_snapshot", "One snapshot of all cached readings. Not a record of any run."),
    "tail_log": _fn(
        "tail_log",
        "Most recent event-log entries.",
        {"count": {"type": "integer", "description": "How many entries, 1-500."}},
    ),
    "recipe_status": _fn(
        "recipe_status",
        "Running flag and progress of the active recipe.",
        {"waitSeconds": {"type": "integer", "description": "0 returns at once; N>0 blocks up to N seconds (max 300)."}},
    ),
    "list_runs": _fn(
        "list_runs",
        "Most recent recipe runs.",
        {"count": {"type": "integer", "description": "How many runs, 1-50."}},
    ),
    "fetch_run_files": _fn(
        "fetch_run_files",
        "Copy a run's artifacts to this computer.",
        {"runId": {"type": "string"}, "kinds": {"type": "string"}},
    ),
    "validate_recipe": _fn(
        "validate_recipe",
        "Validate recipe content without saving. Returns line-numbered findings.",
        {"content": {"type": "string"}, "isRcpx": {"type": "boolean"}},
        ["content"],
    ),
    "compute_deposition_plan": _fn(
        "compute_deposition_plan",
        "Deterministic layer-time plan from period, gamma (heavy-layer fraction) and rates. Planned values, never measured.",
        {
            "period_A": {"type": "number"},
            "gamma": {"type": "number"},
            "heavy_rate_A_per_s": {"type": "number"},
            "light_rate_A_per_s": {"type": "number"},
            "bilayer_count": {"type": "integer"},
            "adhesion_thickness_A": {"type": "number"},
            "adhesion_rate_A_per_s": {"type": "number"},
            "dead_time_s": {"type": "number"},
            "dead_time_rate_A_per_s": {"type": "number"},
        },
        ["period_A", "gamma", "heavy_rate_A_per_s", "light_rate_A_per_s", "bilayer_count"],
    ),
    "upload_recipe": _fn(
        "upload_recipe",
        "Validate then save a .rcpx recipe on the machine.",
        {"name": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}},
        ["name", "content"],
    ),
    "start_recipe": _fn(
        "start_recipe",
        "Start a recipe by name. Returns accepted; poll recipe_status and tail_log. " + INTERLOCK_REFUSAL_RULE,
        {"name": {"type": "string"}},
        ["name"],
    ),
    "abort_recipe": _fn("abort_recipe", "Abort the running recipe and run the failure protocol."),
    "resolve_pause": _fn(
        "resolve_pause",
        "Answer a fault pause: retry, skip or abort.",
        {"decision": {"type": "string"}},
        ["decision"],
    ),
    "device_command": _fn(
        "device_command",
        "Direct manual device command, same path as the operator panels. " + INTERLOCK_REFUSAL_RULE,
        {"deviceType": {"type": "string"}, "deviceId": {"type": "string"}, "argsJson": {"type": "string"}},
        ["deviceType", "deviceId", "argsJson"],
    ),
}


def schemas_for(tool_names):
    return [TOOL_SCHEMAS[name] for name in tool_names]
