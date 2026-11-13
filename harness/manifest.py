"""Frozen campaign configuration: campaign.json, its input and runner fingerprints, and the suite."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal

from bridge import KEY_TOOLS, LOCAL_HOSTS, check_task_tools
from campaign_core import parse_json

HARNESS = Path(__file__).resolve().parent
SUITE = "inputs/suite.json"
TOOL_CLASSES = ("read", "write", "local")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    cls: Literal["read", "write", "local"]


@dataclass(frozen=True)
class Manifest:
    campaign: Path
    fingerprint: str
    models: list[dict]
    task_ids: list[str]
    task_pattern: str
    repetitions: int
    parallelism: int
    cost_threshold_usd: float
    timeouts: dict[str, int]
    max_output_tokens: int
    tool_prefix: str
    tools: list[ToolSpec]
    mcp_server: dict
    bridge_endpoints: dict[str, dict]
    skill_snapshot: str
    grader_model: str
    input_sha256: dict[str, str]
    runner_sha256: dict[str, str]


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _positive_int(value):
    return type(value) is int and value > 0


def _nonempty_str(value):
    return isinstance(value, str) and bool(value)


def _field(settings, key, valid):
    if key not in settings:
        raise ValueError("campaign.json is missing field: " + key)
    if not valid(settings[key]):
        raise ValueError("campaign.json field is invalid: " + key)
    return settings[key]


def _verify_inputs(campaign, inputs):
    root = (campaign / "inputs").resolve()
    for relative, expected in inputs.items():
        if PureWindowsPath(relative).anchor or Path(relative).anchor:
            raise ValueError("Campaign input path is not relative: " + relative)
        path = (campaign / relative).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Campaign input is outside frozen inputs: " + relative)
        try:
            data = path.read_bytes()
        except OSError:
            raise ValueError("Campaign input is missing: " + relative) from None
        if _digest(data) != expected:
            raise ValueError("Campaign input changed: " + relative)


def _verify_runners(runners):
    root = HARNESS.resolve()
    for absolute, expected in runners.items():
        path = Path(absolute)
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("Campaign runner is outside the harness: " + absolute)
        try:
            data = path.read_bytes()
        except OSError:
            raise ValueError("Campaign runner is missing: " + absolute) from None
        if _digest(data) != expected:
            raise ValueError("Campaign runner source changed: " + absolute)


def skill_snapshot_path(campaign: Path, skill_snapshot: str) -> Path:
    """The resolved snapshot directory; a relative skill_snapshot is relative to the campaign."""
    return (Path(campaign) / skill_snapshot).resolve()


def load_manifest(campaign: Path) -> Manifest:
    campaign = Path(campaign)
    content = (campaign / "campaign.json").read_bytes()
    settings = parse_json(content.decode("utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("campaign.json is not an object")

    models = _field(settings, "models", lambda v: isinstance(v, list) and bool(v) and all(
        isinstance(m, dict) and set(m) == {"selector", "thinking"}
        and _nonempty_str(m["selector"]) and _nonempty_str(m["thinking"]) for m in v))
    if len({m["selector"] for m in models}) != len(models):
        raise ValueError("campaign.json field is invalid: models")
    task_pattern = _field(settings, "task_pattern", _nonempty_str)
    task_ids = _field(settings, "task_ids", lambda v: isinstance(v, list) and bool(v)
                      and all(_nonempty_str(i) for i in v) and len(set(v)) == len(v))
    for task_id in task_ids:
        if not re.fullmatch(task_pattern, task_id):
            raise ValueError("Task id does not match task_pattern: " + task_id)
    # Trial identity has no repetition index, so one repetition is all the runner supports.
    repetitions = _field(settings, "repetitions", lambda v: v == 1 and type(v) is int)
    # One emulator stack serves every trial, so trials cannot overlap.
    parallelism = _field(settings, "parallelism", lambda v: v == 1 and type(v) is int)
    cost = _field(settings, "cost_threshold_usd", lambda v: type(v) in (int, float) and math.isfinite(v) and v > 0)
    timeouts = _field(settings, "timeouts", lambda v: isinstance(v, dict) and bool(v)
                      and all(_positive_int(t) for t in v.values()))
    max_output_tokens = _field(settings, "max_output_tokens", _positive_int)
    tool_prefix = _field(settings, "tool_prefix", _nonempty_str)
    raw_tools = _field(settings, "tools", lambda v: isinstance(v, list) and bool(v) and all(
        isinstance(t, dict) and set(t) == {"name", "cls"} and _nonempty_str(t["name"])
        and t["cls"] in TOOL_CLASSES for t in v))
    tools = [ToolSpec(t["name"], t["cls"]) for t in raw_tools]
    if len({t.name for t in tools}) != len(tools):
        raise ValueError("campaign.json field is invalid: tools")
    # Per-trial server arguments are built by the worker; the manifest pins the executable only.
    mcp_server = _field(settings, "mcp_server", lambda v: isinstance(v, dict) and set(v) == {"command"}
                        and _nonempty_str(v["command"]) and Path(v["command"]).is_absolute())
    # host/port of each key's bridge config at freeze; every connection must still use them.
    bridge_endpoints = _field(settings, "bridge_endpoints", lambda v: isinstance(v, dict) and set(v) == set(KEY_TOOLS)
                              and all(isinstance(e, dict) and set(e) == {"host", "port"} and e["host"] in LOCAL_HOSTS
                                      and type(e["port"]) is int and 0 < e["port"] < 65536 for e in v.values()))
    skill_snapshot = _field(settings, "skill_snapshot", _nonempty_str)
    grader_model = _field(settings, "grader_model", _nonempty_str)
    is_hash_map = lambda v: isinstance(v, dict) and all(_nonempty_str(k) and _nonempty_str(h) for k, h in v.items())
    input_sha256 = _field(settings, "input_sha256", lambda v: is_hash_map(v) and bool(v))
    runner_sha256 = _field(settings, "runner_sha256", lambda v: is_hash_map(v) and bool(v))
    _verify_inputs(campaign, input_sha256)
    _verify_runners(runner_sha256)
    return Manifest(
        campaign=campaign, fingerprint=_digest(content), models=models, task_ids=task_ids,
        task_pattern=task_pattern, repetitions=repetitions, parallelism=parallelism,
        cost_threshold_usd=cost, timeouts=timeouts, max_output_tokens=max_output_tokens,
        tool_prefix=tool_prefix, tools=tools, mcp_server=mcp_server, bridge_endpoints=bridge_endpoints,
        skill_snapshot=skill_snapshot,
        grader_model=grader_model, input_sha256=input_sha256, runner_sha256=runner_sha256,
    )


def load_suite(manifest: Manifest) -> dict[str, dict]:
    if SUITE not in manifest.input_sha256:
        raise ValueError("Suite is not a frozen input: " + SUITE)
    content = (manifest.campaign / SUITE).read_bytes()
    if _digest(content) != manifest.input_sha256[SUITE]:
        raise ValueError("Campaign input changed: " + SUITE)
    suite = parse_json(content.decode("utf-8"))
    if not isinstance(suite, dict) or not isinstance(suite.get("tasks"), list):
        raise ValueError("Suite has no task list")
    tasks = {}
    tool_names = {tool.name for tool in manifest.tools}
    local_names = {tool.name for tool in manifest.tools if tool.cls == "local"}
    for task in suite["tasks"]:
        if not isinstance(task, dict) or not _nonempty_str(task.get("id")):
            raise ValueError("Suite task has no id")
        task_id = task["id"]
        if task_id in tasks:
            raise ValueError("Duplicate suite task: " + task_id)
        if not re.fullmatch(manifest.task_pattern, task_id):
            raise ValueError("Task id does not match task_pattern: " + task_id)
        if task.get("type") not in manifest.timeouts:
            raise ValueError("Task type has no timeout: " + task_id)
        if "timeout_s" in task and not _positive_int(task["timeout_s"]):
            raise ValueError("Task timeout is invalid: " + task_id)
        if not _nonempty_str(task.get("ask")):
            raise ValueError("Task has no ask: " + task_id)
        if not isinstance(task.get("tools"), list) or not task["tools"]:
            raise ValueError("Task has no tools: " + task_id)
        for name in task["tools"]:
            if name not in tool_names:
                raise ValueError("Task " + task_id + " uses a tool outside the manifest: " + str(name))
        check_task_tools(task, local_names)
        tasks[task_id] = task
    if list(tasks) != manifest.task_ids:
        raise ValueError("Suite task ids differ from campaign task_ids")
    return tasks


def task_timeout(manifest: Manifest, task: dict) -> int:
    return task.get("timeout_s", manifest.timeouts[task["type"]])
