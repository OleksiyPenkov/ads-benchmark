"""Read-only batch validation and observed-cost admission for ELN campaigns."""
import json
import math
import hashlib

import jsonschema


def parse_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON member")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("Non-finite JSON value")

    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError("Non-finite JSON number")
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return value


def iter_native_usage(path, trial_id):
    """Recover actual native costs even when the derived ledger was interrupted."""
    observed = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            entry = parse_json(line)
            if entry.get("type") != "message_end" or entry.get("message", {}).get("role") != "assistant":
                continue
            message = entry["message"]
            cost = message.get("usage", {}).get("cost", {}).get("total")
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid preserved native cost")
            body = json.dumps(message, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            identity = entry.get("messageId") or message.get("responseId")
            if not isinstance(identity, str) or not identity:
                if message.get("timestamp") is None:
                    raise ValueError("Preserved native usage identity is unavailable")
                identity = "event-sha256-" + body_hash
            key = trial_id + "/" + identity
            if key in observed:
                if observed[key] != (cost, body_hash):
                    raise ValueError("Conflicting preserved native usage")
                continue
            observed[key] = (cost, body_hash)
            yield key, cost


class ToolGuard:
    """Admits a tool batch only when every call names an approved tool and matches its frozen schema."""
    def __init__(self, names, declarations):
        self.names = frozenset(names)
        if len(self.names) != len(names):
            raise ValueError("Duplicate tool name")
        self.validators = {}
        self.optional_fields = {}
        for declaration in declarations:
            name = declaration["name"]
            if name not in self.names or name in self.validators:
                raise ValueError("Exposed tools differ from the approved tool allowlist")
            schema = declaration["parameters"]
            pending = [schema]
            while pending:
                current = pending.pop()
                if isinstance(current, dict):
                    for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                        if key in current and (not isinstance(current[key], str) or not current[key].startswith("#")):
                            raise ValueError("External schema reference refused")
                    pending.extend(current.values())
                elif isinstance(current, list):
                    pending.extend(current)
            validator = jsonschema.validators.validator_for(schema)
            validator.check_schema(schema)
            self.validators[name] = validator(schema, format_checker=jsonschema.FormatChecker())
            self.optional_fields[name] = frozenset(
                field for field in schema.get("properties", {})
                if field not in schema.get("required", ())
            )
        if set(self.validators) != self.names:
            raise ValueError("Approved tool schemas are incomplete")

    def prepare(self, calls, already_called):
        prepared = {}
        seen = set(already_called)
        for call in calls:
            name, call_id, arguments = call.get("name"), call.get("id"), call.get("arguments")
            if not isinstance(name, str) or name not in self.names:
                raise PermissionError("Tool is outside the approved tool allowlist")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise PermissionError("Missing or duplicate tool-call identifier")
            try:
                arguments = parse_json(arguments) if isinstance(arguments, str) else parse_json(json.dumps(arguments, allow_nan=False))
            except (TypeError, ValueError):
                raise PermissionError("Tool arguments are not unambiguous finite JSON") from None
            if isinstance(arguments, dict):
                for field in self.optional_fields[name]:
                    if field in arguments and (arguments[field] is None or arguments[field] == "null"):
                        del arguments[field]
            if not isinstance(arguments, dict) or next(self.validators[name].iter_errors(arguments), None) is not None:
                raise PermissionError("Tool arguments do not match the frozen tool schema")
            seen.add(call_id)
            prepared[call_id] = {"name": name, "arguments": arguments}
        return prepared


class Budget:
    def __init__(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("Budget threshold must be positive and finite")
        self.limit = limit
        self.total = 0.0
        self.records = {}
        self.stop_reason = None

    @property
    def can_schedule(self):
        return self.stop_reason is None and self.total < self.limit

    def record(self, key, cost):
        if not isinstance(key, str) or not key:
            raise ValueError("Usage record needs a stable identifier")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("Recorded usage cost must be finite and nonnegative")
        if key in self.records:
            if self.records[key] != cost:
                raise ValueError("Conflicting duplicate usage record")
            return
        self.records[key] = cost
        self.total += cost

    def note_failure(self, category):
        if category in {
            "invalid_native_cost", "invalid_worker_cost", "conflicting_worker_cost",
            "invalid_journal_cost", "conflicting_journal_cost", "cost_callback_failed",
            "missing_native_message_identity", "conflicting_native_message_duplicate",
            "invalid_native_journal_cost", "unusable_native_cost_accounting",
        }:
            self.stop_reason = "unusable_native_cost_accounting"


def campaign_path(campaign, relative):
    path = (campaign / relative).resolve()
    try:
        path.relative_to(campaign.resolve())
    except ValueError:
        raise ValueError("Recovery evidence must stay inside the campaign") from None
    return path


def accepted_fingerprints(campaign, settings, current):
    accepted = {current}
    semantic_fields = (
        "models", "task_ids", "task_pattern", "repetitions", "parallelism",
        "cost_threshold_usd", "timeouts", "max_output_tokens", "tool_prefix",
        "tools", "mcp_server", "skill_snapshot", "input_sha256",
    )
    for entry in settings.get("accepted_prior_configurations", []):
        content = campaign_path(campaign, entry["path"]).read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        if fingerprint != entry["sha256"]:
            raise ValueError("Prior configuration evidence changed")
        prior = parse_json(content.decode("utf-8"))
        if any(prior.get(field) != settings.get(field) for field in semantic_fields):
            raise ValueError("Prior trials used different tasks, prompts, tools, effort, or limits")
        accepted.add(fingerprint)
    return accepted


def seed_recovery_costs(campaign, entries, accepted, budget):
    for entry in entries:
        result_bytes = campaign_path(campaign, entry["result_file"]).read_bytes()
        if hashlib.sha256(result_bytes).hexdigest() != entry["result_sha256"]:
            raise ValueError("Archived setup result changed")
        result = parse_json(result_bytes.decode("utf-8"))
        if result["status"] == "completed" or result["fingerprint"] not in accepted:
            raise ValueError("Recovery accounting is only for explicit incomplete prior attempts")
        journal = campaign_path(campaign, entry["journal_file"])
        with journal.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != entry["journal_sha256"] or digest != result["journal_sha256"]:
            raise ValueError("Archived setup journal changed")
        costs = dict(iter_native_usage(journal, result["trial_id"]))
        if costs != {usage["key"]: usage["cost"] for usage in result["usage_records"]} or abs(sum(costs.values()) - result["catalog_estimated_cost_usd"]) > 1e-9:
            raise ValueError("Archived setup cost differs from preserved native usage")
        for key, cost in costs.items():
            budget.record("setup/" + result["fingerprint"] + "/" + key, cost)


def portable_declarations(declarations, prefix):
    """Make implicit array-item defaults explicit for strict provider catalogs."""
    portable = parse_json(json.dumps(declarations, ensure_ascii=False, allow_nan=False))
    for declaration in portable:
        declaration["name"] = prefix + declaration["name"]
        pending = [declaration["parameters"]]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                if current.get("type") == "array" and "items" not in current:
                    current["items"] = {}
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
    return portable
