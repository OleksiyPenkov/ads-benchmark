"""Owned campaign trials using installed OMP RPC and the installed MCP SDK."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from ads_stack import PROFILE, StackError
from bridge import BRIDGE, EndpointError, bridge_params, check_endpoint, default_secrets_dir
from campaign_core import ToolGuard, iter_native_usage, parse_json, portable_declarations
from local_tools import SOURCE_ROOTS, LocalTools
from manifest import HARNESS, load_manifest, load_suite, skill_snapshot_path, task_timeout

# Campaign, MCP server, tool prefix, timeouts and limits come from the frozen
# campaign.json (manifest.py); the ELN origin hard-coded them here.
OMP = Path(os.environ.get("ADS_BENCH_OMP_BIN") or (Path(os.environ.get("ProgramFiles") or (Path.home() / "Programs")) / "OMP" / "omp.exe"))
AGY = Path(os.environ.get("ADS_BENCH_AGY_BIN") or (Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "agy" / "bin" / "agy.exe"))
GUARD = HARNESS / "campaign_guard.ts"
TEARDOWN_TIMEOUT = 15
FRAME_LIMIT = 1048576
LOGICAL_LIMIT = 67108864
AGY_MCP_SERVER_NAME = "aps"
AGY_MAX_RECIPE_WAIT_SECONDS = 60
AGY_ACP_STARTUP_TIMEOUT_SECONDS = 120.0
AGY_PASSTHROUGH_MODELS = frozenset(("gemini-pro-orchestrator", "gemini-pro-orchestrator-argon"))
AGY_EFFORT_SUFFIXES = ("-low", "-medium", "-high", "-xhigh")
# API-equivalent USD per 1M tokens (uncached input, output, cached read).
AGY_TOKEN_RATES_USD = {
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-4-carbon": (4.0, 20.0, 0.2),
    "gemini-4-argon": (4.0, 20.0, 0.2),
    "gemini-4-hydrogen": (4.0, 20.0, 0.2),
}
REQUIRED_INPUTS = frozenset(("inputs/suite.json", "inputs/system_prompt.md", "inputs/tools.json",
                             "inputs/runner-config.yml"))
# Categories campaign_guard.ts writes after BENCH_CAMPAIGN_DENIED; anything else is journalled as "unrecognized".
DENIED_PREFIX = b"BENCH_CAMPAIGN_DENIED "
GUARD_DENIALS = frozenset(("session_admission_failed", "prompt_admission_failed", "tool_admission_failed",
                           "provider_admission_failed", "abort_failed"))
EVENTS = [
    "agent_start", "agent_end", "turn_start", "turn_end", "message_end",
    "tool_execution_start", "tool_execution_end", "auto_compaction_start",
    "auto_compaction_end", "auto_retry_start", "auto_retry_end",
    "cache_warming_start", "cache_warming_end", "retry_fallback_applied",
    "retry_fallback_succeeded", "model_changed", "thinking_level_changed",
]


class TrialFailure(Exception):
    """A fixed, safe category, never a provider or credential error message."""


def failure_category(error):
    pending = [error]
    while pending:
        current = pending.pop()
        if isinstance(current, TrialFailure):
            return str(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
    return type(error).__name__


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _normalize_js_numbers(value):
    if isinstance(value, dict):
        return {k: _normalize_js_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_js_numbers(v) for v in value]
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def trial_id(task, model):
    return task["id"] + "_" + model["selector"].replace("/", "__")


def is_agy_selector(selector: str) -> bool:
    return selector.startswith(("agy/", "antigravity-acp/"))


def is_agy_stream_interrupted(text: str) -> bool:
    return "The stream was interrupted." in text and "API error" in text


def agy_model_id(selector: str, thinking: str) -> str:
    """Map benchmark selector + thinking level to the Antigravity ACP model ID."""
    base = selector.partition("/")[2]
    if base in AGY_PASSTHROUGH_MODELS or base.endswith(AGY_EFFORT_SUFFIXES):
        return base
    return f"{base}-{thinking}"


def agy_cost_usd(selector: str, usage: dict) -> dict:
    """Compute API-equivalent token cost record for an Antigravity ACP prompt turn."""
    base = selector.partition("/")[2]
    rate_key = next((k for k in AGY_TOKEN_RATES_USD if base == k or base.startswith(k + "-")), "gemini-4-carbon")
    in_rate, out_rate, cache_rate = AGY_TOKEN_RATES_USD[rate_key]

    def tok(name: str) -> int:
        v = usage.get(name, 0) if isinstance(usage, dict) else 0
        return v if type(v) is int and v >= 0 else 0

    inp = tok("inputTokens")
    out = tok("outputTokens")
    cached = tok("cachedReadTokens")
    c_in = in_rate * inp / 1_000_000
    c_out = out_rate * out / 1_000_000
    c_cache = cache_rate * cached / 1_000_000
    return {
        "input": c_in,
        "output": c_out,
        "cacheRead": c_cache,
        "cacheWrite": 0.0,
        "total": c_in + c_out + c_cache,
    }


def build_system_prompt(base: str, primary_skill_md: str, secondary_skill_md: str | None = None,
                        *, primary_name: str = "aps-operator") -> str:
    """The frozen system prompt: the base prompt followed by the SKILL.md text(s)."""
    if secondary_skill_md is None:
        return base + f"\n\n# Skill: {primary_name}\n\n" + primary_skill_md
    return (base + "\n\n# Skill: ads-operator\n\n" + primary_skill_md
            + "\n\n# Skill: vacuum-operations\n\n" + secondary_skill_md)


def load_trial(campaign, task, model, fingerprint):
    try:
        manifest = load_manifest(campaign)
    except (OSError, ValueError, UnicodeError):
        raise TrialFailure("campaign_manifest_invalid") from None
    if manifest.fingerprint != fingerprint:
        raise TrialFailure("campaign_fingerprint_mismatch")
    if not REQUIRED_INPUTS.issubset(manifest.input_sha256):
        raise TrialFailure("missing_input_fingerprints")
    own_paths = {Path(__file__).resolve(), GUARD.resolve(), (HARNESS / "guard_policy.ts").resolve(),
                 (HARNESS / "campaign_core.py").resolve(), (HARNESS / "manifest.py").resolve(),
                 (HARNESS / "local_tools.py").resolve(), (HARNESS / "bridge.py").resolve()}
    if not own_paths.issubset({Path(absolute).resolve() for absolute in manifest.runner_sha256}):
        raise TrialFailure("missing_runner_fingerprints")
    if model not in manifest.models:
        raise TrialFailure("selected_model_mismatch")
    try:
        tasks = load_suite(manifest)
    except (OSError, ValueError, UnicodeError):
        raise TrialFailure("frozen_suite_invalid") from None
    if tasks.get(task["id"]) != task:
        raise TrialFailure("frozen_task_mismatch")
    declarations = parse_json((campaign / "inputs/tools.json").read_text(encoding="utf-8"))
    if [d["name"] for d in declarations] != [tool.name for tool in manifest.tools]:
        raise TrialFailure("tool_declarations_mismatch")
    if any(d.get("loadMode") != "essential" for d in declarations):
        raise TrialFailure("tool_load_mode_mismatch")
    local_tools = LocalTools(skill_snapshot_path(campaign, manifest.skill_snapshot), SOURCE_ROOTS)
    local_names = {tool.name for tool in manifest.tools if tool.cls == "local"}
    frozen_local = {d["name"]: canonical(d) for d in declarations if d["name"] in local_names}
    if frozen_local != {d["name"]: canonical(d) for d in local_tools.declarations()}:
        raise TrialFailure("local_tool_declaration_drift")
    prefix = manifest.tool_prefix
    task_wire = task_declarations(declarations, task, prefix)
    guard = ToolGuard([d["name"] for d in task_wire], task_wire)
    # The bridge executable is pinned by the manifest; the task's key selects its config (and tier).
    if Path(manifest.mcp_server["command"]) != BRIDGE:
        raise TrialFailure("mcp_server_mismatch")
    try:
        server = bridge_params(task["key"], default_secrets_dir())
    except RuntimeError:
        raise TrialFailure("bridge_secrets_dir_unset") from None
    return manifest, declarations, guard, server, local_tools


def task_declarations(declarations, task, prefix):
    """Wire declarations for the task's own tools, in manifest order; the dispatch boundary admits only these."""
    allowed = set(task["tools"])
    return portable_declarations([d for d in declarations if d["name"] in allowed], prefix)


class Journal:
    def __init__(self, path, mode):
        self.stream = path.open(mode, buffering=0)

    def emit(self, event):
        self.stream.write((json_text({"at": now(), **event}) + "\n").encode("utf-8"))

    def close(self):
        self.stream.close()


def native_message(message):
    # Native errorMessage can embed HTTP headers. Preserve completed model content
    # and usage, never provider diagnostics or unbounded protocol side channels.
    fields = ("role", "content", "api", "provider", "model", "usage", "stopReason",
              "timestamp", "responseId", "duration", "ttft", "completedAt", "toolCallId", "toolName", "isError")
    result = {key: message[key] for key in fields if key in message}
    if "errorMessage" in message:
        result["provider_error_present"] = True
    return result


async def kill_tree(process):
    if process is None:
        return True
    killed = False
    if os.name == "nt":
        killer = None
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(process.pid), "/T", "/F",
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
            killed = killer.returncode == 0 or process.returncode is not None
        except (OSError, TimeoutError):
            if killer is not None and killer.returncode is None:
                killer.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
        except ProcessLookupError:
            killed = True
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        return False
    return killed


async def safe_stderr(stream, journal=None):
    # Consume all bytes, including oversized/non-UTF8 lines, without retaining raw
    # errors. Only the guard's narrow, structured policy vocabulary is persisted.
    buffer = b""
    oversized = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return
        for segment in chunk.splitlines(keepends=True):
            if not oversized:
                buffer += segment
                if len(buffer) > 8192:
                    oversized = True
                    buffer = b""
            if not segment.endswith(b"\n"):
                continue
            if not oversized and journal is not None:
                # Markers written by campaign_guard.ts; renamed together with the guard.
                if buffer.startswith(DENIED_PREFIX):
                    category = buffer[len(DENIED_PREFIX):].rstrip(b"\r\n").decode("utf-8", "replace")
                    journal.emit({"type": "guard_denied",
                                  "category": category if category in GUARD_DENIALS else "unrecognized"})
                prefix = b"BENCH_CAMPAIGN_POLICY "
                if buffer.startswith(prefix):
                    try:
                        record = parse_json(buffer[len(prefix):].decode("utf-8"))
                        keys = {"selector", "model", "provider", "api", "output_token_cap", "request"}
                        if (isinstance(record, dict) and set(record) == keys
                                and all(isinstance(record[k], str) and len(record[k]) < 256
                                        for k in ("selector", "model", "provider", "api"))
                                and type(record["request"]) is int and record["request"] > 0
                                and (record["output_token_cap"] is None
                                     or type(record["output_token_cap"]) is int)):
                            journal.emit({"type": "provider_policy", **record})
                    except (ValueError, UnicodeError, TypeError):
                        pass
            buffer = b""
            oversized = False


class Rpc:
    def __init__(self, process):
        self.process = process
        self.counter = 0

    async def send(self, frame):
        line = (json_text(frame) + "\n").encode("utf-8")
        if len(line) > FRAME_LIMIT:
            raise TrialFailure("rpc_inbound_frame_limit")
        self.process.stdin.write(line)
        await self.process.stdin.drain()

    async def physical(self):
        line = await self.process.stdout.readline()
        if not line:
            raise TrialFailure("rpc_unexpected_eof")
        if len(line) > FRAME_LIMIT or not line.endswith(b"\n"):
            raise TrialFailure("rpc_physical_frame_limit")
        frame = parse_json(line.decode("utf-8"))
        if not isinstance(frame, dict):
            raise TrialFailure("rpc_invalid_frame")
        return frame

    async def receive(self):
        frame = await self.physical()
        if frame.get("type") != "rpc_chunk":
            return frame
        chunk_id, count, size = frame.get("chunkId"), frame.get("count"), frame.get("byteLength")
        if (not isinstance(chunk_id, str) or not chunk_id or type(count) is not int
                or count < 1 or type(size) is not int or not 0 < size <= LOGICAL_LIMIT
                or count > size):
            raise TrialFailure("rpc_invalid_chunk_metadata")
        parts, total = [], 0
        for index in range(count):
            if index:
                frame = await self.physical()
            if (frame.get("type") != "rpc_chunk" or frame.get("chunkId") != chunk_id
                    or type(frame.get("index")) is not int or frame["index"] != index
                    or type(frame.get("count")) is not int or frame["count"] != count
                    or type(frame.get("byteLength")) is not int or frame["byteLength"] != size):
                raise TrialFailure("rpc_interrupted_chunk_sequence")
            try:
                part = base64.b64decode(frame["data"], validate=True)
            except (KeyError, ValueError, TypeError):
                raise TrialFailure("rpc_invalid_chunk_data") from None
            if not part:
                raise TrialFailure("rpc_empty_chunk")
            total += len(part)
            if total > size:
                raise TrialFailure("rpc_chunk_size_overflow")
            parts.append(part)
        if total != size:
            raise TrialFailure("rpc_chunk_size_mismatch")
        value = parse_json(b"".join(parts).decode("utf-8"))
        if not isinstance(value, dict) or value.get("type") == "rpc_chunk":
            raise TrialFailure("rpc_invalid_reassembled_frame")
        return value

    async def command(self, kind, handler, **arguments):
        self.counter += 1
        key = "command-" + str(self.counter)
        await self.send({"id": key, "type": kind, **arguments})
        while True:
            frame = await self.receive()
            if frame.get("type") == "response" and frame.get("id") == key:
                if frame.get("success") is not True or frame.get("command") != kind:
                    raise TrialFailure("rpc_command_failed_" + kind)
                return frame.get("data")
            await handler(frame)


def args_without_intent(arguments):
    if isinstance(arguments, str):
        arguments = parse_json(arguments)
    if not isinstance(arguments, dict):
        raise PermissionError("tool_arguments_not_object")
    return {key: value for key, value in arguments.items()
            if key not in ("i", "toolSummary", "waitForPreviousTools")}


class Trial:
    def __init__(self, task, model, guard, journal, prefix, local_tools=None):
        self.task, self.model, self.guard, self.journal, self.prefix = task, model, guard, journal, prefix
        self.local_tools = local_tools
        self.identifier = trial_id(task, model)
        self.prepared = {}
        self.reserved = set()
        self.called = set()
        self.host_requests = set()
        self.usage = {}
        self.message_hashes = {}
        self.denied = []
        self.answer = None
        self.last_stop = None
        self.prompt_completed = False
        self.settled = False
        self.tool_count = 0
        self.session = None
        self.rpc = None
        self.prompt_id = "trial-prompt"
        self.continuations = 0
    def deny(self, category, name=None, call_id=None):
        entry = {"category": category, "name": name, "call_id": call_id}
        self.denied.append(entry)
        self.journal.emit({"type": "denied", **entry})
        raise TrialFailure(category)

    def check_identity(self, message):
        if message.get("provider") + "/" + message.get("model") != self.model["selector"]:
            raise TrialFailure("native_model_identity_mismatch")

    def record_usage(self, frame, message):
        usage = message.get("usage")
        cost = usage.get("cost", {}).get("total") if isinstance(usage, dict) else None
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise TrialFailure("invalid_native_cost")
        identity = frame.get("messageId") or message.get("responseId")
        if not isinstance(identity, str) or not identity:
            if message.get("timestamp") is None:
                raise TrialFailure("missing_native_message_identity")
            identity = "event-sha256-" + digest(canonical(native_message(message)).encode("utf-8"))
        key = self.identifier + "/" + identity
        observed = digest(canonical(native_message(message)).encode("utf-8"))
        if key in self.usage:
            if self.usage[key]["cost"] != cost or self.message_hashes[key] != observed:
                raise TrialFailure("conflicting_native_message_duplicate")
            return False
        entry = {"key": key, "cost": cost}
        self.usage[key] = entry
        self.message_hashes[key] = observed
        self.journal.emit({"type": "usage", **entry})
        print(json_text({"type": "cost", **entry}), flush=True)
        return True

    async def handle(self, frame):
        kind = frame.get("type")
        if kind == "message_end":
            message = frame.get("message")
            if not isinstance(message, dict):
                raise TrialFailure("invalid_native_message")
            self.journal.emit({"type": "message_end", "messageId": frame.get("messageId"),
                               "message": native_message(message)})
            if message.get("role") != "assistant":
                return
            if not self.record_usage(frame, message):
                return
            self.check_identity(message)
            content = message.get("content")
            if not isinstance(content, list):
                raise TrialFailure("invalid_assistant_content")
            calls = []
            for block in content:
                if not isinstance(block, dict):
                    raise TrialFailure("invalid_assistant_content_block")
                if block.get("type") == "toolCall":
                    arguments = block.get("arguments")
                    if "partialArgs" in block:
                        try:
                            original = parse_json(block["partialArgs"])
                            parsed = parse_json(arguments) if isinstance(arguments, str) else arguments
                            if canonical(_normalize_js_numbers(original)) != canonical(_normalize_js_numbers(parsed)):
                                self.deny("native_raw_arguments_mismatch", block.get("name"), block.get("id"))
                        except (ValueError, TypeError):
                            self.deny("native_raw_arguments_invalid", block.get("name"), block.get("id"))
                    calls.append({"id": block.get("id"), "name": block.get("name"), "arguments": arguments})
                elif block.get("type") not in ("text", "thinking"):
                    self.deny("unsupported_assistant_content", block.get("type"))
            self.last_stop = message.get("stopReason")
            self.answer = None
            if calls:
                if self.last_stop != "toolUse":
                    self.deny("tool_batch_stop_reason_mismatch")
                try:
                    normalized = [{**call, "arguments": args_without_intent(call["arguments"])} for call in calls]
                    prepared = self.guard.prepare(normalized, self.reserved)
                except (PermissionError, ValueError, TypeError):
                    forbidden = next((call for call in calls if not isinstance(call["name"], str)
                                      or call["name"] not in self.guard.names), {})
                    self.deny("assistant_batch_denied", forbidden.get("name"), forbidden.get("id"))
                if self.prepared:
                    self.deny("undispatched_prior_batch")
                self.prepared = prepared
                self.reserved.update(prepared)
                self.journal.emit({"type": "tool_batch_validated", "call_ids": list(prepared)})
            elif self.last_stop == "stop":
                text = "\n".join(block["text"] for block in content
                                 if block.get("type") == "text" and isinstance(block.get("text"), str))
                if text.strip():
                    self.answer = text
            elif self.last_stop == "error" and message.get("errorMessage") and self.continuations < 3:
                pass
            else:
                raise TrialFailure("native_assistant_not_completed")
        elif kind == "host_tool_call":
            request_id, call_id, name = frame.get("id"), frame.get("toolCallId"), frame.get("toolName")
            if (not isinstance(request_id, str) or not request_id or request_id in self.host_requests
                    or not isinstance(call_id, str) or call_id in self.called or call_id not in self.prepared):
                self.deny("unprepared_or_repeated_host_call", name, call_id)
            if name not in self.guard.names or not name.startswith(self.prefix):
                self.deny("host_tool_name_denied", name, call_id)
            try:
                arguments = args_without_intent(frame.get("arguments"))
                confirmed = self.guard.prepare([{"id": call_id, "name": name, "arguments": arguments}], set())
            except (PermissionError, ValueError, TypeError):
                self.deny("host_tool_arguments_denied", name, call_id)
            prepared = self.prepared[call_id]
            if canonical(_normalize_js_numbers(confirmed[call_id])) != canonical(_normalize_js_numbers(prepared)):
                self.deny("host_call_differs_from_native_batch", name, call_id)
            self.host_requests.add(request_id)
            self.called.add(call_id)
            self.prepared.pop(call_id)
            self.tool_count += 1
            backend_name = name[len(self.prefix):]
            self.journal.emit({"type": "tool_call", "host_request_id": request_id, "call_id": call_id,
                               "name": backend_name, "wire_name": name, "arguments": prepared["arguments"], "host_arguments": frame["arguments"]})
            # This is the only dispatch boundary. Every call in this assistant
            # batch has already passed ToolGuard before this point.
            if self.local_tools is not None and backend_name in self.local_tools.names:
                raw = await self.local_tools.call(backend_name, prepared["arguments"])
            else:
                result = await self.session.call_tool(backend_name, prepared["arguments"])
                raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                raw.setdefault("isError", bool(result.isError))
            self.journal.emit({"type": "tool_result", "host_request_id": request_id, "call_id": call_id,
                               "name": backend_name, "wire_name": name, "result": raw})
            content = [{"type": "text", "text": json_text(raw)}]
            await self.rpc.send({"type": "host_tool_result", "id": request_id,
                                 "result": {"content": content}, "isError": bool(raw["isError"])})
        elif kind == "tool_execution_start":
            name, call_id = frame.get("toolName"), frame.get("toolCallId")
            if name not in self.guard.names or call_id not in self.reserved:
                self.deny("non_host_tool_execution_denied", name, call_id)
            self.journal.emit({"type": kind, "toolName": name, "toolCallId": call_id,
                               "args": frame.get("args")})
        elif kind == "tool_execution_end":
            if frame.get("toolName") not in self.guard.names or frame.get("toolCallId") not in self.called:
                self.deny("unexpected_tool_execution_end", frame.get("toolName"), frame.get("toolCallId"))
            self.journal.emit({"type": kind, "toolName": frame.get("toolName"),
                               "toolCallId": frame.get("toolCallId"), "isError": frame.get("isError")})
        elif kind == "prompt_result":
            self.journal.emit({"type": kind, "id": frame.get("id"), "status": frame.get("status"),
                               "agentInvoked": frame.get("agentInvoked"), "sessionSettled": frame.get("sessionSettled")})
            if (frame.get("id") != self.prompt_id or frame.get("status") != "completed"
                    or frame.get("agentInvoked") is not True):
                raise TrialFailure("native_prompt_not_completed")
            self.prompt_completed = True
            self.settled = frame.get("sessionSettled") is True
        elif kind == "session_settled":
            self.journal.emit({"type": kind})
            self.settled = True
        elif kind in ("agent_start", "turn_start"):
            self.settled = False
            self.journal.emit({"type": kind})
        elif kind in ("agent_end", "turn_end"):
            # message_end is the sole full-message copy; lifecycle events often
            # duplicate all messages and results, so retain only their metadata.
            self.journal.emit({"type": kind, **{key: frame[key] for key in
                               ("isTerminal", "yielded", "awaitingAsyncWork", "messageCount") if key in frame}})
        elif kind in ("auto_compaction_start", "auto_compaction_end", "auto_retry_start", "auto_retry_end",
                      "cache_warming_start", "cache_warming_end", "retry_fallback_applied", "retry_fallback_succeeded"):
            raise TrialFailure("unexpected_native_" + kind)
        elif kind in ("extension_error", "rpc_frame_error", "host_tool_cancel", "host_uri_request", "host_uri_cancel",
                      "extension_ui_request", "subagent_lifecycle", "subagent_event", "subagent_progress"):
            raise TrialFailure("unexpected_native_" + kind)
        elif kind == "model_changed":
            changed = frame.get("model", {})
            if changed.get("provider") + "/" + changed.get("id") != self.model["selector"]:
                raise TrialFailure("native_model_changed")
        elif kind == "thinking_level_changed":
            if frame.get("level", frame.get("thinkingLevel")) != self.model["thinking"]:
                raise TrialFailure("native_thinking_changed")
        elif kind == "response":
            if frame.get("id") != self.prompt_id or frame.get("success") is not True:
                raise TrialFailure("unexpected_rpc_response")
            if frame.get("data", {}).get("agentInvoked") is False:
                raise TrialFailure("prompt_completed_without_native_agent")
        elif kind in ("available_commands_update", "config_update", "config_warnings_changed", "session_info_update",
                      "message_start", "message_update", "tool_execution_update", "tool_stream_update"):
            return
        elif kind == "notice":
            raise TrialFailure("native_notice")
        else:
            raise TrialFailure("unexpected_rpc_frame")

    def result(self, status, started, category=None):
        return {"trial_id": self.identifier, "task": self.task["id"], "selector": self.model["selector"],
                "thinking": self.model["thinking"], "status": status,
                "answer": self.answer if status == "completed" else None,
                "duration_s": round(time.monotonic() - started, 3),
                "catalog_estimated_cost_usd": math.fsum(entry["cost"] for entry in self.usage.values()),
                "usage_records": list(self.usage.values()), "tool_call_count": self.tool_count,
                "denied_tool_calls": self.denied, "error_category": category}
def remote_declarations(declarations, tools):
    """Declarations served by the MCP server; local host tools never appear in its live list."""
    local = {tool.name for tool in tools if tool.cls == "local"}
    return [declaration for declaration in declarations if declaration["name"] not in local]


async def discover(session, declarations, tools):
    found, cursors, cursor = {}, set(), None
    while True:
        page = await session.list_tools(cursor=cursor)
        for tool in page.tools:
            if tool.name in found:
                raise TrialFailure("duplicate_live_mcp_tool")
            found[tool.name] = tool
        cursor = page.nextCursor
        if not cursor:
            break
        if cursor in cursors:
            raise TrialFailure("repeated_mcp_discovery_cursor")
        cursors.add(cursor)
    for declaration in remote_declarations(declarations, tools):
        tool = found.get(declaration["name"])
        if (tool is None or canonical(tool.inputSchema) != canonical(declaration["parameters"])
                or (tool.description or "") != declaration["description"]):
            raise TrialFailure("live_mcp_declaration_drift")


async def gate_bridge(session, expected_profile: str | None = None):
    """describe_machine on the trial's own bridge session; anything but the emulator profile is refused."""
    target_profile = expected_profile or PROFILE
    result = await session.call_tool("describe_machine", {})
    text = "".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    try:
        machine = json.loads(text) if not result.isError else None
    except ValueError:
        machine = None
    if not isinstance(machine, dict) or machine.get("profile") != target_profile or machine.get("isEmulator") is False:
        raise TrialFailure("bridge_not_emulator")
    return machine


def _prepare_agy_gemini_dir() -> Path:
    """Create an isolated Gemini customization directory without global plugins or skills."""
    iso = default_secrets_dir() / "agy-gemini"
    acp_dir = iso / "antigravity-acp"
    cfg_dir = iso / "config"
    acp_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    home_gemini = Path.home() / ".gemini"
    for name in ("jetski_state.pbtxt", "installation_id"):
        for candidate in (home_gemini / "antigravity-acp" / name, home_gemini / "antigravity-cli" / name):
            if candidate.is_file():
                shutil.copy2(candidate, acp_dir / name)
                break
    (cfg_dir / "config.json").write_text("{}", encoding="utf-8")
    (cfg_dir / "skills.json").write_text('{"entries":[]}', encoding="utf-8")
    (cfg_dir / "mcp_config.json").write_text('{"mcpServers":{}}', encoding="utf-8")
    return iso


class _AcpClient:
    """Async JSON-RPC stdio client for agy.exe acp."""

    def __init__(self, process, manifest, trial, on_abort):
        self.process = process
        self.manifest = manifest
        self.trial = trial
        self.on_abort = on_abort
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.thought_chunks: list[str] = []
        self.text_chunks: list[str] = []
        self._expected_prefix = f"mcp_{AGY_MCP_SERVER_NAME}_"

    def _allowed_agy_tool(self, name: str | None) -> bool:
        if not isinstance(name, str) or not name.startswith(self._expected_prefix):
            return False
        wire_name = self.manifest.tool_prefix + name[len(self._expected_prefix):]
        return wire_name in self.trial.guard.names

    async def send(self, payload: dict) -> None:
        raw = (json_text(payload) + "\n").encode("utf-8")
        if len(raw) > FRAME_LIMIT:
            raise TrialFailure("rpc_inbound_frame_limit")
        self.process.stdin.write(raw)
        await self.process.stdin.drain()

    async def request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self.send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    def fail_all(self, error: BaseException) -> None:
        self.on_abort(error)
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(error)

    async def read_loop(self) -> None:
        while True:
            line = await self.process.stdout.readline()
            if not line:
                self.fail_all(TrialFailure("acp_unexpected_eof"))
                return
            if len(line) > FRAME_LIMIT or not line.endswith(b"\n"):
                self.fail_all(TrialFailure("rpc_physical_frame_limit"))
                return
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("{"):
                continue
            try:
                msg = parse_json(text)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            if "id" in msg and "method" not in msg:
                fut = self._pending.get(msg["id"])
                if fut is not None and not fut.done():
                    if "error" in msg and msg["error"] is not None:
                        fut.set_exception(TrialFailure("acp_request_failed"))
                    else:
                        fut.set_result(msg.get("result") or {})
            elif msg.get("method") == "session/update":
                params = msg.get("params") or {}
                update = params.get("update") or {}
                kind = update.get("sessionUpdate")
                content = update.get("content") or {}
                if kind == "agent_thought_chunk" and content.get("type") == "text":
                    self.thought_chunks.append(content.get("text") or "")
                elif kind == "agent_message_chunk" and content.get("type") == "text":
                    self.text_chunks.append(content.get("text") or "")
                elif kind == "tool_call":
                    tool_name = update.get("name")
                    if not self._allowed_agy_tool(tool_name):
                        try:
                            self.trial.deny("non_host_tool_execution_denied", str(tool_name), update.get("toolCallId"))
                        except TrialFailure as exc:
                            self.fail_all(exc)
                            return
            elif msg.get("method") == "session/request_permission":
                params = msg.get("params") or {}
                tc = params.get("toolCall") or {}
                tool_name = tc.get("name") or tc.get("title")
                opts = params.get("options") or []
                if self._allowed_agy_tool(tool_name) and opts:
                    allow_opt = next((o["optionId"] for o in opts if "allow" in str(o.get("optionId", "")).lower()), opts[0].get("optionId"))
                    outcome = {"outcome": "selected", "optionId": allow_opt}
                else:
                    outcome = {"outcome": "cancelled"}
                await self.send({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})


async def _run_agy_trial(campaign, manifest, declarations, task, model, journal, trial):
    from aiohttp import web

    if not AGY.is_file():
        raise TrialFailure("agy_binary_missing")
    iso_dir = _prepare_agy_gemini_dir()
    wire_declarations = task_declarations(declarations, task, manifest.tool_prefix)
    prefix = manifest.tool_prefix
    backend_to_wire = {d["name"][len(prefix):]: d["name"] for d in wire_declarations}
    token = secrets.token_hex(24)
    abort_holder: list[BaseException] = []
    call_seq = 0

    def record_abort(exc: BaseException) -> None:
        if not abort_holder:
            abort_holder.append(exc)

    acp_holder: list[_AcpClient] = []

    async def handle_mcp(request: web.Request) -> web.StreamResponse:
        nonlocal call_seq
        if request.headers.get("Authorization") != f"Bearer {token}":
            return web.Response(status=401)
        try:
            body = parse_json(await request.text())
        except ValueError:
            return web.json_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}})
        method = body.get("method")
        rid = body.get("id")
        if method == "initialize":
            return web.json_response({
                "jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                           "serverInfo": {"name": AGY_MCP_SERVER_NAME, "version": "1.0.0"}},
            })
        if method == "notifications/initialized":
            return web.Response(status=202)
        if method == "tools/list":
            tools_payload = [
                {"name": d["name"][len(prefix):], "description": d["description"], "inputSchema": d["parameters"]}
                for d in wire_declarations
            ]
            return web.json_response({"jsonrpc": "2.0", "id": rid, "result": {"tools": tools_payload}})
        if method == "tools/call":
            params = body.get("params") or {}
            backend_name = params.get("name")
            raw_args = params.get("arguments") or {}
            call_seq += 1
            call_id = f"agy-call-{call_seq}"
            wire_name = backend_to_wire.get(backend_name)
            try:
                if wire_name is None:
                    trial.deny("host_tool_name_denied", str(backend_name), call_id)
                clean_args = args_without_intent(raw_args)
                confirmed = trial.guard.prepare([{"id": call_id, "name": wire_name, "arguments": clean_args}], trial.called)
            except TrialFailure as exc:
                record_abort(exc)
                if acp_holder:
                    acp_holder[0].fail_all(exc)
                return web.json_response({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True}})
            except (PermissionError, ValueError, TypeError):
                try:
                    trial.deny("host_tool_arguments_denied", wire_name, call_id)
                except TrialFailure as exc:
                    record_abort(exc)
                    if acp_holder:
                        acp_holder[0].fail_all(exc)
                return web.json_response({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "denied"}], "isError": True}})

            prepared = confirmed[call_id]
            trial.reserved.add(call_id)
            trial.called.add(call_id)
            trial.tool_count += 1
            journal.emit({"type": "tool_batch_validated", "call_ids": [call_id]})
            journal.emit({"type": "tool_call", "host_request_id": call_id, "call_id": call_id,
                          "name": backend_name, "wire_name": wire_name,
                          "arguments": prepared["arguments"], "host_arguments": raw_args})
            call_args = dict(prepared["arguments"])
            # Prevent agy's HTTP MCP client from hitting per-request timeouts on long status waits.
            if backend_name == "recipe_status" and isinstance(call_args.get("waitSeconds"), (int, float)) and call_args["waitSeconds"] > AGY_MAX_RECIPE_WAIT_SECONDS:
                call_args["waitSeconds"] = AGY_MAX_RECIPE_WAIT_SECONDS
            if trial.local_tools is not None and backend_name in trial.local_tools.names:
                raw = await trial.local_tools.call(backend_name, call_args)
            else:
                res = await trial.session.call_tool(backend_name, call_args)
                raw = res.model_dump(mode="json", by_alias=True, exclude_none=True)
                raw.setdefault("isError", bool(res.isError))
            journal.emit({"type": "tool_result", "host_request_id": call_id, "call_id": call_id,
                          "name": backend_name, "wire_name": wire_name, "result": raw})
            return web.json_response({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": json_text(raw)}], "isError": bool(raw["isError"])}})
        return web.json_response({"jsonrpc": "2.0", "id": rid, "result": {}})

    app = web.Application()
    app.router.add_post("/mcp", handle_mcp)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    agy_process, stderr_task, reader_task, session_id = None, None, None, None
    try:
        sys_prompt = (campaign / "inputs/system_prompt.md").read_text(encoding="utf-8")
        sys_prompt += (
            f"\n\nUse ONLY the mcp_{AGY_MCP_SERVER_NAME}_* tools exposed by the {AGY_MCP_SERVER_NAME} MCP server "
            "for this task. Never call read_file, write_file, edit_file, run_command, or any other tool."
        )
        agy_process = await asyncio.create_subprocess_exec(
            str(AGY), "acp", "--stdio", "--dangerously-skip-permissions", "--include-default-customizations=false",
            f"--gemini-dir={iso_dir}", "--app_data_dir=antigravity-acp",
            cwd=str(campaign), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=FRAME_LIMIT + 1, start_new_session=os.name != "nt",
        )
        stderr_task = asyncio.create_task(safe_stderr(agy_process.stderr))
        acp = _AcpClient(agy_process, manifest, trial, record_abort)
        acp_holder.append(acp)
        reader_task = asyncio.create_task(acp.read_loop())

        try:
            await acp.request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
                "clientInfo": {"name": "ads-benchmark", "version": "1.0"},
            }, timeout=AGY_ACP_STARTUP_TIMEOUT_SECONDS)
            new_sess = await acp.request("session/new", {
                "cwd": str(campaign),
                "mcpServers": [{"type": "http", "name": AGY_MCP_SERVER_NAME,
                                "url": f"http://127.0.0.1:{port}/mcp",
                                "headers": [{"name": "Authorization", "value": f"Bearer {token}"}]}],
                "_meta": {"antigravity": {"systemPrompt": {"append": sys_prompt},
                                          "eagerMcpServers": [AGY_MCP_SERVER_NAME], "enabledTools": []}},
            }, timeout=AGY_ACP_STARTUP_TIMEOUT_SECONDS)
            session_id = new_sess.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                raise TrialFailure("acp_missing_session_id")
            target_model = agy_model_id(model["selector"], model["thinking"])
            await acp.request("session/set_model", {"sessionId": session_id, "modelId": target_model}, timeout=AGY_ACP_STARTUP_TIMEOUT_SECONDS)
            journal.emit({"type": "native_startup_verified", "selector": model["selector"],
                          "thinking": model["thinking"], "host_tools": [d["name"] for d in wire_declarations]})

            provider, _, model_name = model["selector"].partition("/")
            prompt_text = task["ask"]
            prompt_timeout = float(task_timeout(manifest, task))
            turn_idx = 0
            while True:
                acp.thought_chunks.clear()
                acp.text_chunks.clear()
                res = await acp.request(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt_text}]},
                    timeout=prompt_timeout,
                )
                if abort_holder:
                    raise abort_holder[0]
                acp_stop = res.get("stopReason")
                answer_text = "".join(acp.text_chunks)
                thought_text = "".join(acp.thought_chunks)
                usage = res.get("usage") or {}
                cost_obj = agy_cost_usd(model["selector"], usage)
                is_limit = acp_stop in ("max_tokens", "max_turn_requests")
                mapped_stop = "stop" if acp_stop == "end_turn" else "length" if is_limit else "error"
                blocks = []
                if thought_text:
                    blocks.append({"type": "thinking", "thinking": thought_text})
                blocks.append({"type": "text", "text": answer_text})
                msg_id = f"agy-turn-{turn_idx}"
                message = {
                    "role": "assistant", "content": blocks, "api": "antigravity-acp",
                    "provider": provider, "model": model_name,
                    "usage": {**usage, "cost": cost_obj}, "stopReason": mapped_stop,
                    "timestamp": int(time.time() * 1000),
                }
                frame = {"messageId": msg_id, "message": message}
                journal.emit({"type": "message_end", "messageId": msg_id, "message": native_message(message)})
                trial.record_usage(frame, message)
                turn_idx += 1
                if mapped_stop == "length" and trial.continuations < 3:
                    trial.continuations += 1
                    prompt_text = "The previous turn reached an output or step limit without finishing. Continue from where you left off."
                    journal.emit({"type": "continuation", "index": trial.continuations,
                                  "prompt_id": f"trial-continuation-{trial.continuations}", "message": prompt_text})
                    continue
                if is_agy_stream_interrupted(answer_text):
                    raise TrialFailure("acp_stream_interrupted")
                if mapped_stop != "stop" or not answer_text.strip():
                    raise TrialFailure("native_assistant_not_completed")
                trial.answer = answer_text
                trial.last_stop = "stop"
                break
            journal.emit({"type": "native_session_verified", "isSettled": True, "selector": model["selector"]})
        except TimeoutError:
            raise TrialFailure("acp_timeout") from None
    finally:
        if session_id and agy_process is not None and agy_process.returncode is None:
            try:
                await asyncio.wait_for(acp_holder[0].request("session/delete", {"sessionId": session_id}, timeout=2.0), timeout=2.0)
            except Exception:
                pass
        if reader_task is not None:
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)
        if agy_process is not None:
            await kill_tree(agy_process)
        if stderr_task is not None:
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except (TimeoutError, Exception):
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
        await runner.cleanup()


async def child_trial(campaign, task, model, fingerprint, journal_path):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    started = time.monotonic()
    manifest, declarations, guard, server, local_tools = load_trial(campaign, task, model, fingerprint)
    journal = Journal(journal_path, "ab")
    trial = Trial(task, model, guard, journal, manifest.tool_prefix, local_tools)
    omp_process, stderr_task = None, None
    status, category = "failed", "worker_error"
    try:
        # The config's host/port must still be the frozen local destination before the bridge starts.
        try:
            check_endpoint(task["key"], default_secrets_dir(), manifest.bridge_endpoints)
        except (EndpointError, RuntimeError):
            raise TrialFailure("bridge_endpoint_mismatch") from None
        # The exported wrapper owns this entire process tree. Its deadline and
        # final taskkill bound both SDK context teardown and native OMP shutdown.
        with open(os.devnull, "w", encoding="utf-8") as errlog:
            async with stdio_client(server, errlog=errlog) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    initialized = await session.initialize()
                    journal.emit({"type": "mcp_initialized", "protocolVersion": initialized.protocolVersion,
                                  "serverInfo": initialized.serverInfo.model_dump(mode="json", exclude_none=True)})
                    await discover(session, declarations, manifest.tools)
                    journal.emit({"type": "mcp_declarations_verified",
                                  "names": [d["name"] for d in remote_declarations(declarations, manifest.tools)]})
                    # This session, not only the stack's Read probe, must reach the emulator before any tool
                    # is exposed to the model.
                    machine = await gate_bridge(session, (parse_json((Path(campaign) / "campaign.json").read_text(encoding="utf-8")).get("machine") or {}).get("profile"))
                    journal.emit({"type": "bridge_gate_passed", "key": task["key"], "profile": machine["profile"]})
                    trial.session = session
                    if is_agy_selector(model["selector"]):
                        await _run_agy_trial(campaign, manifest, declarations, task, model, journal, trial)
                        status, category = "completed", None
                    else:
                        env = os.environ.copy()
                        env.update({"BENCH_CAMPAIGN_MODEL": model["selector"], "BENCH_CAMPAIGN_FINGERPRINT": fingerprint,
                                    "BENCH_CAMPAIGN_THINKING": model["thinking"], "BENCH_CAMPAIGN_DIR": str(campaign),
                                    "BENCH_CAMPAIGN_TASK": task["id"], "PI_NO_TITLE": "1"})
                        omp_process = await asyncio.create_subprocess_exec(
                            str(OMP), "--mode", "rpc", "--no-ui", "--no-session", "--no-tools", "--no-lsp",
                            "--no-pty", "--no-skills", "--no-rules", "--trusted-extension", str(GUARD),
                            "--config", str(campaign / "inputs/runner-config.yml"),
                            "--model", model["selector"], "--thinking", model["thinking"],
                            cwd=str(campaign), env=env, stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                            limit=FRAME_LIMIT + 1, start_new_session=os.name != "nt",
                        )
                        stderr_task = asyncio.create_task(safe_stderr(omp_process.stderr, journal))
                        rpc = Rpc(omp_process)
                        trial.rpc = rpc
                        ready = await rpc.receive()
                        if ready.get("type") != "ready" or 2 not in ready.get("supportedProtocolVersions", []):
                            raise TrialFailure("rpc_lossless_protocol_unavailable")
                        await rpc.command("negotiate_protocol", trial.handle, protocolVersion=2)
                        await rpc.command("set_event_filter", trial.handle, events=EVENTS, messageUpdates="delta")
                        await rpc.command("set_auto_retry", trial.handle, enabled=False)
                        await rpc.command("set_auto_compaction", trial.handle, enabled=False)
                        await rpc.command("set_cache_warming", trial.handle, mode="off")
                        wire_declarations = task_declarations(declarations, task, manifest.tool_prefix)
                        registered = await rpc.command("set_host_tools", trial.handle, tools=wire_declarations)
                        if (not isinstance(registered, dict)
                                or registered.get("toolNames") != [d["name"] for d in wire_declarations]):
                            raise TrialFailure("host_tools_not_acknowledged")
                        state = await rpc.command("get_state", trial.handle)
                        if (state.get("model", {}).get("provider") + "/" + state.get("model", {}).get("id")
                                != model["selector"] or state.get("thinkingLevel") != model["thinking"]
                                or state.get("autoCompactionEnabled") is not False):
                            raise TrialFailure("native_startup_state_mismatch")
                        journal.emit({"type": "native_startup_verified", "selector": model["selector"],
                                      "thinking": model["thinking"], "host_tools": registered["toolNames"]})
                        prompt_msg = task["ask"]
                        while True:
                            await rpc.send({"id": trial.prompt_id, "type": "prompt", "message": prompt_msg})
                            while not (trial.prompt_completed and trial.settled):
                                await trial.handle(await rpc.receive())
                            if trial.last_stop == "error" and not trial.answer and trial.continuations < 3:
                                trial.continuations += 1
                                trial.prompt_id = "trial-continuation-%d" % trial.continuations
                                trial.prompt_completed = trial.settled = False
                                prompt_msg = "The previous turn was interrupted by a transient stream error. Please continue from where you left off."
                                journal.emit({"type": "continuation", "index": trial.continuations,
                                              "prompt_id": trial.prompt_id, "message": prompt_msg})
                            else:
                                break
                        state = await rpc.command("get_state", trial.handle)
                        if (state.get("isSettled") is not True or state.get("isStreaming") is not False
                                or state.get("isCompacting") is not False or state.get("hasPendingAsyncWork") is not False
                                or state.get("queuedMessageCount") != 0 or trial.prepared or trial.denied
                                or not trial.answer or trial.last_stop != "stop"):
                            raise TrialFailure("native_session_not_settled")
                        if (state.get("model", {}).get("provider") + "/" + state.get("model", {}).get("id")
                                != model["selector"] or state.get("thinkingLevel") != model["thinking"]):
                            raise TrialFailure("native_final_identity_mismatch")
                        journal.emit({"type": "native_session_verified", "isSettled": True, "selector": model["selector"]})
                        status, category = "completed", None
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        category = failure_category(exc)
    finally:
        if omp_process is not None:
            stopped = await kill_tree(omp_process)
            if not stopped:
                status, category = "failed", "omp_teardown_failed"
        if stderr_task is not None:
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except (TimeoutError, Exception):
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
    result = trial.result(status, started, category)
    journal.emit({"type": "worker_result", **result})
    journal.close()
    print(json_text({"type": "worker_result", "result": result}), flush=True)
    # Stay alive until the exporting wrapper kills the owned tree. This keeps
    # Windows taskkill attached to a live root even after SDK context cleanup.
    await asyncio.Event().wait()


async def run_trial(*, campaign: Path, task: dict, model: dict, fingerprint: str,
                    on_cost: Callable[[str, float], None],
                    stack: Callable[[], Awaitable[dict]] | None = None) -> dict:
    """Run one exclusive trial; forward costs live and always kill its owned tree.

    stack() readies and gates the emulator and returns the stack_verified record, which is
    journalled before the prompt; a StackError fails the trial before any model process starts.
    """
    campaign = Path(campaign)
    manifest = load_trial(campaign, task, model, fingerprint)[0]
    hard_timeout = task_timeout(manifest, task)
    identifier = trial_id(task, model)
    journal_path = campaign / "results" / (identifier + ".jsonl")
    result_path = campaign / "results" / (identifier + ".json")
    if result_path.exists():
        raise FileExistsError(str(result_path))
    journal = Journal(journal_path, "xb")
    journal.emit({"type": "trial_start", "trial_id": identifier, "task": task, "ask": task["ask"],
                  "selector": model["selector"], "thinking": model["thinking"], "fingerprint": fingerprint,
                  "transport": ("AGY ACP stdio + loopback MCP ClientSession" if is_agy_selector(model["selector"]) else "OMP RPC v2 + installed MCP stdio ClientSession"),
                  "hard_timeout_seconds": hard_timeout, "max_output_tokens": manifest.max_output_tokens,
                  "codex_output_cap": None, "stderr_policy": "raw diagnostics discarded; structured guard policy only"})
    journal.close()
    started = time.monotonic()
    process, stderr_task = None, None
    records = {}
    result = None
    status, category, cancelled = "failed", "worker_error", False
    try:
        if stack is not None:
            try:
                verified = await stack()
            except StackError as exc:
                raise TrialFailure(exc.code) from None
            stack_journal = Journal(journal_path, "ab")
            stack_journal.emit(verified)
            stack_journal.close()
        async with asyncio.timeout(hard_timeout):
            env = os.environ.copy()
            env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--internal-worker", str(campaign),
                task["id"], model["selector"], model["thinking"], fingerprint,
                env=env,
                cwd=str(campaign), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=FRAME_LIMIT + 1, start_new_session=os.name != "nt",
            )
            stderr_task = asyncio.create_task(safe_stderr(process.stderr))
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                if len(line) > FRAME_LIMIT or not line.endswith(b"\n"):
                    raise TrialFailure("worker_frame_limit")
                event = parse_json(line.decode("utf-8"))
                if event.get("type") == "cost":
                    key, cost = event.get("key"), event.get("cost")
                    if (not isinstance(key, str) or not key.startswith(identifier + "/")
                            or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                        raise TrialFailure("invalid_worker_cost")
                    if key in records:
                        if records[key] != cost:
                            raise TrialFailure("conflicting_worker_cost")
                        continue
                    records[key] = cost
                    on_cost(key, cost)
                elif event.get("type") == "worker_result" and result is None:
                    result = event["result"]
                    break
                else:
                    raise TrialFailure("unexpected_worker_protocol")
            if result is None:
                raise TrialFailure("worker_exited_without_result")
            if (result.get("trial_id") != identifier or result.get("selector") != model["selector"]
                    or result.get("thinking") != model["thinking"] or result.get("task") != task["id"]
                    or result.get("usage_records") != [{"key": key, "cost": cost} for key, cost in records.items()]):
                raise TrialFailure("worker_result_identity_or_usage_mismatch")
            status, category = result["status"], result["error_category"]
    except asyncio.CancelledError:
        status, category, cancelled = "cancelled", "parent_cancelled", True
    except TimeoutError:
        status, category = "timeout", "trial_deadline_exceeded"
    except Exception as exc:
        status, category = "failed", str(exc) if isinstance(exc, TrialFailure) else type(exc).__name__
    finally:
        # Cleanup is a separate task so caller cancellation cannot strand the
        # owned Python/OMP/MCP tree. The total cleanup wait is bounded.
        async def cleanup():
            stopped = await kill_tree(process)
            if stderr_task is not None:
                try:
                    await asyncio.wait_for(stderr_task, timeout=2)
                except (TimeoutError, Exception):
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)
            return stopped
        cleanup_task = asyncio.create_task(cleanup())
        try:
            stopped = await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=TEARDOWN_TIMEOUT)
        except asyncio.CancelledError:
            cancelled = True
            status, category = "cancelled", "parent_cancelled"
            try:
                stopped = await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=TEARDOWN_TIMEOUT)
            except (TimeoutError, asyncio.CancelledError):
                stopped = False
        except TimeoutError:
            stopped = False
        if not stopped:
            status, category = "failed", "worker_tree_teardown_failed"
    # A deadline can interrupt IPC after the native cost was durably journaled.
    # Reconcile those observed usage records before returning, including failures.
    observed_calls, observed_denials = 0, []
    with journal_path.open("rb") as source:
        for line in source:
            try:
                event = parse_json(line.decode("utf-8"))
            except (ValueError, UnicodeError):
                status, category = "failed", "interrupted_journal_record"
                break
            if event.get("type") == "tool_call":
                observed_calls += 1
            elif event.get("type") == "denied":
                observed_denials.append({key: event.get(key) for key in ("category", "name", "call_id")})
            elif event.get("type") == "usage":
                key, cost = event.get("key"), event.get("cost")
                if (not isinstance(key, str) or not key.startswith(identifier + "/")
                        or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                    status, category = "failed", "invalid_journal_cost"
                    continue
                if key in records:
                    if records[key] != cost:
                        status, category = "failed", "conflicting_journal_cost"
                else:
                    records[key] = cost
                    try:
                        on_cost(key, cost)
                    except Exception:
                        status, category = "failed", "cost_callback_failed"
    recovered_usage = []
    try:
        for key, cost in iter_native_usage(journal_path, identifier):
            if key in records:
                if records[key] != cost:
                    status, category = "failed", "conflicting_journal_cost"
                continue
            records[key] = cost
            recovered_usage.append({"key": key, "cost": cost})
            try:
                on_cost(key, cost)
            except Exception:
                status, category = "failed", "cost_callback_failed"
    except (ValueError, UnicodeError, TypeError):
        status, category = "failed", "invalid_native_journal_cost"
    if recovered_usage:
        recovery_journal = Journal(journal_path, "ab")
        try:
            for usage in recovered_usage:
                recovery_journal.emit({"type": "usage", "source": "recovered_native_message", **usage})
        finally:
            recovery_journal.close()
    if result is None or status != result.get("status") or category != result.get("error_category"):
        # Interrupted journals stay intact. They cannot become completed merely
        # because a partial assistant answer or an earlier final record exists.
        result = {"trial_id": identifier, "task": task["id"], "selector": model["selector"],
                  "thinking": model["thinking"], "status": status, "answer": None,
                  "tool_call_count": observed_calls, "denied_tool_calls": observed_denials, "error_category": category}
    result.update({"duration_s": round(time.monotonic() - started, 3), "fingerprint": fingerprint,
                   "journal_file": str(journal_path), "usage_records": [{"key": key, "cost": cost} for key, cost in records.items()],
                   "catalog_estimated_cost_usd": math.fsum(records.values())})
    terminal = Journal(journal_path, "ab")
    terminal.emit({"type": "result", **result})
    terminal.close()
    with journal_path.open("rb") as source:
        result["journal_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    with result_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json_text(result) + "\n")
    if cancelled:
        raise asyncio.CancelledError
    return result


def internal_main():
    if len(sys.argv) != 7 or sys.argv[1] != "--internal-worker":
        raise SystemExit(2)
    campaign, task_id, selector, thinking, fingerprint = sys.argv[2:]
    root = Path(campaign)
    task = load_suite(load_manifest(root))[task_id]
    model = {"selector": selector, "thinking": thinking}
    path = root / "results" / (trial_id(task, model) + ".jsonl")
    asyncio.run(child_trial(root, task, model, fingerprint, path))


if __name__ == "__main__":
    try:
        internal_main()
    except BaseException:
        # Never print a traceback: provider/SDK exceptions may contain secrets.
        raise SystemExit(1) from None
