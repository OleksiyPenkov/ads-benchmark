"""Freeze an ADS campaign: copy and hash every input a trial reads, and write campaign.json.

    python harness/freeze.py --campaign <dir> --models <selector[:thinking]>... --grader <selector>

The tool declarations come from one live tools/list through the Manual bridge key, on an emulator stack
that the freeze starts from the reset snapshot and stops again. A Control or emulator process that is
already running is refused and left alone; a bridge whose describe_machine.profile is not ADS is
refused. An existing campaign directory is refused; a failed freeze removes the directory it created.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

from ads_stack import (PROFILE, StackPaths, WindowsRunner, agent_port_open, bridge_probe, bridge_session,
                       freeze_baseline, profile_hashes, stack_processes, start_stack, stop_stack, wait_ready)
from bridge import BRIDGE, bridge_endpoints, default_secrets_dir
from campaign_core import parse_json
from campaign_worker import build_system_prompt
from local_tools import SOURCE_ROOTS, LocalTools
from manifest import HARNESS, load_manifest, load_suite

ROOT = HARNESS.parent
SUITE_SOURCE = ROOT / "suite" / "ads-suite.json"
PROMPT_SOURCE = ROOT / "suite" / "system_prompt.md"
REFERENCE_SOURCE = ROOT / "suite" / "scoring-reference.json"
GRADER_BRIEF_SOURCE = ROOT / "suite" / "grader-brief.md"
RUBRIC_SOURCE = ROOT / "suite" / "ADS-Benchmark.md"
AC1_BRIEF_SOURCE = ROOT / "suite" / "AC1-brief.md"
FIXTURE_SOURCE = ROOT / "fixtures"
RUNNER_CONFIG = ROOT / "suite" / "runner-config.yml"
USER_MCP = Path(os.environ.get("ADS_BENCH_USER_MCP") or (Path.home() / ".omp" / "agent" / "mcp.json"))
STACK_PATHS = StackPaths()
# The Manual key grants the full tool set, so its tools/list declares every tool a task can name.
LIST_KEY = "manual"
# describe_machine fields recorded in campaign.json; the rest of the reply is not needed.
MACHINE_FIELDS = ("profile", "appVersion", "controlMode")
# Every harness file the runtime imports or loads.
RUNNERS = ("campaign_core.py", "campaign_worker.py", "run_campaign.py", "manifest.py", "local_tools.py",
           "ads_stack.py", "bridge.py", "grading.py", "score_ac1.py", "score_sc1.py", "score_cc1.py", "make_report.py", "campaign_guard.ts",
           "guard_policy.ts")
WRITE_TOOLS = frozenset({"validate_recipe", "upload_recipe", "start_recipe", "abort_recipe", "resolve_pause",
                         "device_command"})
SELECTOR = re.compile(r"[a-z0-9][a-z0-9-]*/\S+")
DEFAULT_THINKING = "high"
TRACKS = {
    "sputter": {
        "profile": "Sputter",
        "suite": "sputter-suite.json",
        "brief": ("SC1", "SC1-brief.md"),
        "task_pattern": "^S(0[1-9]|10)$|^SC[12]$",
    },
    "ads": {
        "profile": "ADS",
        "suite": "ads-suite.json",
        "brief": ("AC1", "AC1-brief.md"),
        "task_pattern": "^A(0[1-9]|1[0-9]|20)$|^AC[12]$",
    },
    "cluster": {
        "profile": "Cluster",
        "suite": "cluster-suite.json",
        "brief": ("CC1", "CC1-brief.md"),
        "task_pattern": "^C(0[1-9]|10)$|^CC[12]$",
    },
}
SETTINGS = {
    "task_pattern": "^A(0[1-9]|1[0-9]|20)$|^AC[12]$",
    "repetitions": 1,
    "parallelism": 1,
    "cost_threshold_usd": 50,
    "timeouts": {"short": 900, "capstone": 5400},
    "max_output_tokens": 32768,
    "tool_prefix": "aps_",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def query_bridge(paths: StackPaths, *, runner=None, session=None, port_open=agent_port_open,
                 sleep=asyncio.sleep, sync_sleep=time.sleep, clock=time.monotonic) -> tuple[dict, list[dict], dict]:
    """Start an own stack, read describe_machine and the full tools/list through the Manual key, stop the stack.

    Returns the describe_machine reply, the tools and the profile hashes measured while the stack ran.
    A running Control or emulator is refused and never touched; the started PIDs are always stopped.
    """
    runner = WindowsRunner() if runner is None else runner
    session = functools.partial(bridge_session, LIST_KEY, default_secrets_dir()) if session is None else session
    running = stack_processes(runner)
    if running:
        raise RuntimeError("A Control or emulator process is already running (" + ", ".join(sorted(
            {ident[0] for ident in running.values()})) + "); the freeze starts and stops its own stack, so close it first")
    pids: dict = {}
    try:
        pids = start_stack(paths, runner, os.environ.copy(), sleep=sync_sleep)
        return asyncio.run(_query(paths, session, port_open, sleep, clock))
    finally:
        stop_stack(runner, pids, sleep=sync_sleep, clock=clock)


async def _query(paths, session, port_open, sleep, clock):
    ready = await wait_ready(bridge_probe(session), port_open=port_open, sleep=sleep, clock=clock)
    machine = ready["machine"]
    expected_profile = paths.profile
    if not isinstance(machine, dict) or machine.get("profile") != expected_profile or machine.get("isEmulator") is False:
        profile = machine.get("profile") if isinstance(machine, dict) else None
        raise ValueError("describe_machine.profile is " + repr(profile) + ", not " + expected_profile)
    async with session() as bridge:
        tools, cursors, cursor = [], set(), None
        while True:
            page = await bridge.session.list_tools(cursor=cursor)
            tools += [{"name": tool.name, "description": tool.description or "",
                       "inputSchema": tool.inputSchema} for tool in page.tools]
            cursor = page.nextCursor
            if not cursor:
                break
            if cursor in cursors:
                raise RuntimeError("Bridge tools/list repeated a cursor")
            cursors.add(cursor)
    return machine, tools, profile_hashes(paths)


def _models(models):
    if not isinstance(models, list) or not models:
        raise ValueError("At least one model is required")
    for model in models:
        if (not isinstance(model, dict) or set(model) != {"selector", "thinking"}
                or not isinstance(model["selector"], str) or not SELECTOR.fullmatch(model["selector"])
                or not isinstance(model["thinking"], str) or not model["thinking"]):
            raise ValueError("Invalid model: " + repr(model))
    if len({model["selector"] for model in models}) != len(models):
        raise ValueError("A model is listed twice")
    return [dict(model) for model in models]


def parse_model(value: str) -> dict:
    """selector[:thinking]; thinking defaults to high."""
    selector, _, thinking = value.rpartition(":") if ":" in value else (value, "", DEFAULT_THINKING)
    return {"selector": selector, "thinking": thinking}


def _fenced(text):
    fence = "```"
    while fence in text:
        fence += "`"
    return fence + "\n" + text + ("" if text.endswith("\n") else "\n") + fence


def _source_file(name, expected):
    """fixtures/<name>, checked against its recorded SHA-256."""
    path = FIXTURE_SOURCE / name
    data = path.read_bytes()
    if Path(name).name != name or sha256(data) != expected:
        raise ValueError("Fixture source does not match fixtures.json: " + name)
    return data


def _copy_skills(target):
    """Each skill source root copied file by file into its own snapshot folder."""
    for source_root, folder in SOURCE_ROOTS:
        source_root = Path(source_root)
        for source in sorted(source_root.rglob("*")):
            relative = source.relative_to(source_root)
            if "__pycache__" in relative.parts or not source.is_file():
                continue
            destination = target / folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if not (target / folder / "SKILL.md").is_file():
            raise ValueError("Skill source has no SKILL.md: " + str(source_root))


def _embed_attachments(tasks):
    """Attachment text appended to each task's ask under `Attached file: <name>`."""
    index = parse_json((FIXTURE_SOURCE / "fixtures.json").read_bytes().decode("utf-8")).get("attachments", {})
    for task in tasks:
        for attachment_id in task.get("attachments", []):
            record = index.get(attachment_id)
            if not isinstance(record, dict) or set(record) != {"file", "sha256"}:
                raise ValueError("Unknown attachment: " + repr(attachment_id))
            text = _source_file(record["file"], record["sha256"]).decode("utf-8")
            task["ask"] += "\n\nAttached file: " + record["file"] + "\n\n" + _fenced(text)


def _check_capstone_brief(tasks, capstone_id="AC1", brief_source=None):
    """The execution capstone ask is the frozen brief, byte for byte."""
    source = brief_source or AC1_BRIEF_SOURCE
    brief = source.read_bytes().decode("utf-8")
    for task in tasks:
        if task.get("id") == capstone_id and task.get("ask") != brief:
            raise ValueError(f"The {capstone_id} ask differs from {source.as_posix()}; fix the suite first")


def _freeze_tools(tasks, live, local_declarations):
    """tools.json declarations and campaign.json tool classes: live bridge tools the tasks name, then local tools."""
    wanted = {name for task in tasks for name in task["tools"]}
    local_names = [d["name"] for d in local_declarations]
    found = {}
    for tool in live:
        if tool["name"] in found:
            raise ValueError("Bridge lists a tool twice: " + tool["name"])
        found[tool["name"]] = tool
    missing = sorted(wanted - set(found) - set(local_names))
    if missing:
        raise ValueError("Tasks name tools the bridge does not list: " + ", ".join(missing))
    declarations = [{"name": tool["name"], "description": tool["description"], "parameters": tool["inputSchema"],
                     "loadMode": "essential"} for tool in live if tool["name"] in wanted and tool["name"] not in local_names]
    declarations += local_declarations
    tools = [{"name": d["name"], "cls": "local" if d["name"] in local_names
              else "write" if d["name"] in WRITE_TOOLS else "read"} for d in declarations]
    return declarations, tools


def frozen_endpoints() -> dict[str, dict]:
    """host/port of the three bridge configs (never their keys); all must point at this workstation."""
    return bridge_endpoints(default_secrets_dir())


def _write_campaign(campaign, models, grader_model, machine, live, running_hashes, endpoints, *,
                    track="ads", stack_paths=None):
    spec = TRACKS[track]
    paths = stack_paths or (STACK_PATHS if track == "ads" else StackPaths.for_profile(spec["profile"]))
    suite_source = SUITE_SOURCE if track == "ads" else (ROOT / "suite" / spec["suite"])
    cap_id, brief_file = spec["brief"]
    brief_source = AC1_BRIEF_SOURCE if track == "ads" else (ROOT / "suite" / brief_file)
    inputs = campaign / "inputs"
    snapshot = inputs / "skill"
    _copy_skills(snapshot)
    suite = parse_json(suite_source.read_bytes().decode("utf-8"))
    tasks = suite["tasks"]
    _check_capstone_brief(tasks, cap_id, brief_source)
    _embed_attachments(tasks)
    (inputs / "suite.json").write_bytes(json_bytes(suite))
    skill_mds = [(folder, (snapshot / folder / "SKILL.md").read_bytes().decode("utf-8"))
                 for _, folder in SOURCE_ROOTS]
    if len(skill_mds) == 1:
        prompt = build_system_prompt(PROMPT_SOURCE.read_bytes().decode("utf-8"),
                                     skill_mds[0][1], primary_name=skill_mds[0][0])
    else:
        prompt = build_system_prompt(PROMPT_SOURCE.read_bytes().decode("utf-8"),
                                     skill_mds[0][1], skill_mds[1][1])
    (inputs / "system_prompt.md").write_bytes(prompt.encode("utf-8"))
    local = LocalTools(snapshot, SOURCE_ROOTS).declarations()
    declarations, tools = _freeze_tools(tasks, live, local)
    (inputs / "tools.json").write_bytes(json_bytes(declarations))
    shutil.copyfile(RUNNER_CONFIG, inputs / "runner-config.yml")
    shutil.copyfile(REFERENCE_SOURCE, inputs / "scoring-reference.json")
    shutil.copyfile(GRADER_BRIEF_SOURCE, inputs / "grader-brief.md")
    shutil.copyfile(RUBRIC_SOURCE, inputs / "ADS-Benchmark.md")
    # inputs/stack-baseline.json and inputs/recipes-baseline/, taken with the stack stopped.
    profile = freeze_baseline(paths, campaign)
    if profile != running_hashes:
        raise ValueError(f"The {paths.profile} profile changed between the live query and the baseline; "
                         "a trial gate would refuse it")
    (campaign / "results").mkdir()

    servers = parse_json(USER_MCP.read_bytes().decode("utf-8")).get("mcpServers", {}) if USER_MCP.is_file() else {}
    (campaign / ".omp").mkdir()
    (campaign / ".omp" / "mcp.json").write_bytes(
        json_bytes({"mcpServers": {name: {"enabled": False} for name in sorted(servers)}}))

    settings = {
        "campaign_id": campaign.name,
        "created_on": date.today().isoformat(),
        "models": models,
        "task_ids": [task["id"] for task in tasks],
        **{**SETTINGS, "task_pattern": spec["task_pattern"]},
        "track": track,
        "tools": tools,
        "keys": {task["id"]: task["key"] for task in tasks},
        "mcp_server": {"command": BRIDGE.as_posix()},
        "bridge_endpoints": endpoints,
        "machine": {field: machine.get(field) for field in MACHINE_FIELDS},
        "snapshot": {"path": paths.snapshot.as_posix(), "sha256": sha256(paths.snapshot.read_bytes())},
        "profile_sha256": profile,
        "skill_snapshot": "inputs/skill",
        "grader_model": grader_model,
        "input_sha256": {path.relative_to(campaign).as_posix(): sha256(path.read_bytes())
                         for path in sorted(inputs.rglob("*")) if path.is_file()},
        "runner_sha256": {(HARNESS / name).as_posix(): sha256((HARNESS / name).read_bytes()) for name in RUNNERS},
    }
    (campaign / "campaign.json").write_bytes(json_bytes(settings))


def freeze(campaign: Path, models: list[dict], grader_model: str, *, track: str = "ads",
           stack_paths: StackPaths | None = None) -> Path:
    """Create the campaign directory with frozen inputs, .omp/mcp.json and campaign.json; return its path."""
    campaign = Path(campaign).resolve()
    if track == "all":
        if campaign.exists():
            raise FileExistsError("Campaign directory already exists: " + str(campaign))
        campaign.mkdir(parents=True)
        try:
            for sub in ("sputter", "ads", "cluster"):
                freeze(campaign / sub, models, grader_model, track=sub)
            (campaign / "tracks.json").write_bytes(json_bytes({"tracks": ["sputter", "ads", "cluster"]}))
        except BaseException:
            shutil.rmtree(campaign, ignore_errors=True)
            raise
        return campaign
    if track not in TRACKS:
        raise ValueError("Unknown track: " + repr(track))
    models = _models(models)
    if not isinstance(grader_model, str) or not grader_model.strip() or grader_model == "unset":
        raise ValueError("A grader model is required (got " + repr(grader_model) + ")")
    if grader_model in {model["selector"] for model in models}:
        raise ValueError("The grader model must not be in the model roster: " + grader_model)
    if campaign.exists():
        raise FileExistsError("Campaign directory already exists: " + str(campaign))
    spec = TRACKS[track]
    paths = stack_paths or (STACK_PATHS if track == "ads" else StackPaths.for_profile(spec["profile"]))
    endpoints = frozen_endpoints()
    machine, live, running_hashes = query_bridge(paths)
    campaign.mkdir(parents=True)
    try:
        _write_campaign(campaign, models, grader_model, machine, live, running_hashes, endpoints,
                        track=track, stack_paths=paths)
        load_suite(load_manifest(campaign))
    except BaseException:
        shutil.rmtree(campaign, ignore_errors=True)
        raise
    return campaign


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze an ADS-Bench v2.0 campaign.")
    parser.add_argument("--campaign", required=True, type=Path, help="campaign directory to create")
    parser.add_argument("--track", choices=("sputter", "ads", "cluster", "all"), default="ads",
                        help="benchmark track to freeze (default: ads)")
    parser.add_argument("--models", required=True, nargs="+", help="selector[:thinking], thinking defaults to high")
    parser.add_argument("--grader", required=True, help="grader model selector")
    args = parser.parse_args(argv)
    campaign = freeze(args.campaign, [parse_model(value) for value in args.models], args.grader, track=args.track)
    print("CAMPAIGN_FROZEN " + str(campaign))
    return 0


if __name__ == "__main__":
    sys.exit(main())
