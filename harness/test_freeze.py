import contextlib
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import freeze
from ads_stack import CONTROL_IMAGE, EMULATOR_IMAGE, StackError, check_gate, load_baseline, profile_hashes
from bridge import BRIDGE, KEY_TOOLS
from local_tools import SOURCE_ROOTS
from manifest import HARNESS, load_manifest, load_suite
from test_ads_stack import IDLE, MACHINE, FakeRunner, make_paths
from test_manifest import write_bridge_configs

ROOT = HARNESS.parent
GRADER = "anthropic/claude-opus-5-5"
MODELS = [{"selector": "anthropic/claude-sonnet-5-5", "thinking": "high"},
          {"selector": "deepseek/deepseek-flash", "thinking": "off"}]
RUNNERS = ("campaign_core.py", "campaign_worker.py", "run_campaign.py", "manifest.py", "local_tools.py",
           "ads_stack.py", "bridge.py", "grading.py", "score_ac1.py", "score_sc1.py", "score_cc1.py", "make_report.py", "campaign_guard.ts",
           "guard_policy.ts")
WRITE = {"validate_recipe", "upload_recipe", "start_recipe", "abort_recipe", "resolve_pause", "device_command"}


def live_tools():
    """A tools/list as the Manual key returns it, plus one tool no task names."""
    return [{"name": name, "description": "Live " + name + ".",
             "inputSchema": {"type": "object", "properties": {"n": {"type": "integer"}}, "additionalProperties": False}}
            for name in KEY_TOOLS["manual"] + ["unused_tool"]]


class FakeBridge:
    """Bridge sessions on the fake stack; tools/list answers in two pages."""

    def __init__(self, machine=MACHINE):
        self.machine, self.keys, self.calls = dict(machine), [], []

    def __call__(self, key, secrets_dir):
        self.keys.append(key)
        return self.open()

    @contextlib.asynccontextmanager
    async def open(self):
        yield SimpleNamespace(call=self.call, session=SimpleNamespace(list_tools=self.list_tools))

    async def call(self, name, arguments=None):
        self.calls.append(name)
        return {"describe_machine": self.machine, "system_status": IDLE}[name]

    async def list_tools(self, cursor=None):
        self.calls.append("tools/list:" + str(cursor))
        tools = [SimpleNamespace(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                 for t in live_tools()]
        return (SimpleNamespace(tools=tools[:5], nextCursor="page-2") if cursor is None
                else SimpleNamespace(tools=tools[5:], nextCursor=None))


async def no_sleep(_seconds):
    return None


async def port_open():
    return True


def fake_stack(monkeypatch, tmp_path, *, running=(), machine=MACHINE):
    """Stack paths in tmp_path, a fake process table and fake Manual bridge sessions."""
    paths = make_paths(tmp_path / "aps")
    runner, bridge = FakeRunner(running), FakeBridge(machine)
    monkeypatch.setattr(freeze, "STACK_PATHS", paths)
    monkeypatch.setattr(freeze, "bridge_session", bridge)
    monkeypatch.setattr(freeze, "query_bridge", functools.partial(
        freeze.query_bridge, runner=runner, port_open=port_open, sleep=no_sleep, sync_sleep=lambda _s: None))
    secrets = tmp_path / "secrets"
    write_bridge_configs(secrets)
    monkeypatch.setattr(freeze, "default_secrets_dir", lambda: secrets)
    return SimpleNamespace(paths=paths, runner=runner, bridge=bridge, secrets=secrets)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suite_source():
    return json.loads((ROOT / "suite" / "ads-suite.json").read_text(encoding="utf-8"))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    campaign = freeze.freeze(tmp_path / "ads-test", MODELS, GRADER)
    return campaign, stack


def test_frozen_campaign_loads_and_profile_change_fails_gate(frozen):
    campaign, stack = frozen
    manifest = load_manifest(campaign)
    assert list(load_suite(manifest)) == manifest.task_ids
    assert manifest.models == MODELS and manifest.grader_model == GRADER
    hashes, _baseline = load_baseline(campaign, manifest.input_sha256)
    check_gate(MACHINE, IDLE, hashes, stack.paths)
    (stack.paths.profile_dir / "Sources.xml").write_text("<Sources edited='yes'/>", encoding="utf-8")
    with pytest.raises(StackError) as error:
        check_gate(MACHINE, IDLE, hashes, stack.paths)
    assert error.value.code == "profile_hash_mismatch"


def test_snapshot_tamper_fails_manifest(frozen):
    campaign, _ = frozen
    target = campaign / "inputs" / "skill" / "aps-operator" / "SKILL.md"
    data = bytearray(target.read_bytes())
    data[100] ^= 1
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="inputs/skill/aps-operator/SKILL.md"):
        load_manifest(campaign)


def test_live_query_owns_its_stack_and_uses_the_manual_key(frozen):
    _, stack = frozen
    started = [entry[1][0] for entry in stack.runner.log if entry[0] == "start"]
    assert [Path(argv).name for argv in started] == [EMULATOR_IMAGE, CONTROL_IMAGE]
    assert sorted(entry[1] for entry in stack.runner.log if entry[0] == "kill") == sorted([CONTROL_IMAGE, EMULATOR_IMAGE])
    assert stack.runner.procs == {}
    assert set(stack.bridge.keys) == {"manual"}
    assert stack.bridge.calls[:2] == ["describe_machine", "system_status"]
    assert "tools/list:None" in stack.bridge.calls and "tools/list:page-2" in stack.bridge.calls


def test_fixed_campaign_values(frozen):
    campaign, stack = frozen
    settings = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    assert settings["parallelism"] == 1 and settings["repetitions"] == 1
    assert settings["cost_threshold_usd"] == 50
    assert settings["timeouts"] == {"short": 900, "capstone": 5400}
    assert settings["max_output_tokens"] == 32768
    assert settings["task_pattern"] == "^A(0[1-9]|1[0-9]|20)$|^AC[12]$"
    assert settings["tool_prefix"] == "aps_"
    assert settings["mcp_server"] == {"command": freeze.BRIDGE.as_posix()}
    assert Path(settings["mcp_server"]["command"]) == BRIDGE
    assert settings["skill_snapshot"] == "inputs/skill"
    assert "validator_argv" not in settings and "engine" not in settings
    assert settings["machine"] == {key: MACHINE[key] for key in ("profile", "appVersion", "controlMode")}
    classes = {tool["name"]: tool["cls"] for tool in settings["tools"]}
    assert classes["skill_read"] == "local"
    assert {name for name, cls in classes.items() if cls == "write"} == WRITE
    assert {cls for name, cls in classes.items() if name not in WRITE | {"skill_read"}} == {"read"}
    tasks = suite_source()["tasks"]
    assert set(classes) == {name for task in tasks for name in task["tools"]}
    assert settings["keys"] == {task["id"]: task["key"] for task in tasks}
    assert settings["snapshot"] == {"path": stack.paths.snapshot.as_posix(), "sha256": sha(stack.paths.snapshot)}
    assert settings["profile_sha256"] == profile_hashes(stack.paths)
    assert settings["profile_sha256"] == json.loads(
        (campaign / "inputs" / "stack-baseline.json").read_text(encoding="utf-8"))
    assert "AuditLog/2026-10-09.jsonl" not in settings["profile_sha256"]
    assert "bin/AgentServer.xml" in settings["profile_sha256"]


def test_real_snapshot_path_is_the_emulator_reset_state():
    assert freeze.STACK_PATHS.snapshot == (ROOT / "snapshots" / "ADS_Idle.emstate")


def test_hashes_cover_every_input_and_runner(frozen):
    campaign, _ = frozen
    settings = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    files = {path.relative_to(campaign).as_posix(): sha(path)
             for path in (campaign / "inputs").rglob("*") if path.is_file()}
    assert settings["input_sha256"] == files
    for required in ("inputs/suite.json", "inputs/system_prompt.md", "inputs/tools.json", "inputs/runner-config.yml",
                     "inputs/scoring-reference.json", "inputs/ADS-Benchmark.md", "inputs/grader-brief.md",
                     "inputs/skill/aps-operator/SKILL.md",
                     "inputs/stack-baseline.json", "inputs/recipes-baseline/Cleaning.rcpx",
                     "inputs/recipes-baseline/Tests/Deposition_test.rcpx"):
        assert required in files
    assert settings["runner_sha256"] == {(HARNESS / name).as_posix(): sha(HARNESS / name) for name in RUNNERS}


def test_frozen_inputs_match_sources(frozen):
    campaign, stack = frozen
    inputs = campaign / "inputs"
    for source, folder in SOURCE_ROOTS:
        source_files = {p.relative_to(source).as_posix(): sha(p) for p in Path(source).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts}
        frozen_files = {p.relative_to(inputs / "skill" / folder).as_posix(): sha(p)
                        for p in (inputs / "skill" / folder).rglob("*") if p.is_file()}
        assert frozen_files == source_files and "SKILL.md" in frozen_files
    assert (inputs / "runner-config.yml").read_bytes() == (ROOT / "suite" / "runner-config.yml").read_bytes()
    for name in ("scoring-reference.json", "grader-brief.md", "ADS-Benchmark.md"):
        assert (inputs / name).read_bytes() == (ROOT / "suite" / name).read_bytes()
    from campaign_worker import build_system_prompt
    expected = build_system_prompt(
        (ROOT / "suite" / "system_prompt.md").read_bytes().decode("utf-8"),
        (Path(SOURCE_ROOTS[0][0]) / "SKILL.md").read_bytes().decode("utf-8"),
        primary_name=SOURCE_ROOTS[0][1])
    assert (inputs / "system_prompt.md").read_bytes() == expected.encode("utf-8")
    recipes = {p.relative_to(stack.paths.recipes_dir).as_posix(): sha(p)
               for p in stack.paths.recipes_dir.rglob("*") if p.is_file()}
    assert {p.relative_to(inputs / "recipes-baseline").as_posix(): sha(p)
            for p in (inputs / "recipes-baseline").rglob("*") if p.is_file()} == recipes
    assert not (inputs / "fixtures").exists() and not (inputs / "fixtures.json").exists()


def test_tools_json_is_live_declarations_plus_local(frozen):
    campaign, _ = frozen
    declarations = json.loads((campaign / "inputs" / "tools.json").read_text(encoding="utf-8"))
    settings = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    assert [d["name"] for d in declarations] == [t["name"] for t in settings["tools"]]
    live = {t["name"]: t for t in live_tools()}
    for declaration in declarations:
        assert set(declaration) == {"name", "description", "parameters", "loadMode"}
        assert declaration["loadMode"] == "essential"
        if declaration["name"] != "skill_read":
            assert declaration["parameters"] == live[declaration["name"]]["inputSchema"]
            assert declaration["description"] == live[declaration["name"]]["description"]
    assert declarations[-1]["name"] == "skill_read"
    assert "unused_tool" not in {d["name"] for d in declarations}


def test_every_task_passes_trial_admission(frozen):
    from campaign_worker import load_trial
    campaign, _ = frozen
    manifest = load_manifest(campaign)
    for task in load_suite(manifest).values():
        load_trial(campaign, task, MODELS[0], manifest.fingerprint)


def test_attachments_are_embedded_in_the_ask(frozen):
    campaign, _ = frozen
    tasks = {t["id"]: t for t in json.loads((campaign / "inputs" / "suite.json").read_text(encoding="utf-8"))["tasks"]}
    source = {t["id"]: t for t in suite_source()["tasks"]}
    index = json.loads((ROOT / "fixtures" / "fixtures.json").read_text(encoding="utf-8"))["attachments"]
    for task_id, task in source.items():
        ask = tasks[task_id]["ask"]
        if not task["attachments"]:
            assert ask == task["ask"]
            continue
        (attachment,) = task["attachments"]
        name = index[attachment]["file"]
        text = (ROOT / "fixtures" / name).read_bytes().decode("utf-8")
        assert ask.startswith(task["ask"] + "\n\nAttached file: " + name + "\n\n```")
        assert text in ask and ask.endswith("```")
    assert tasks["AC1"]["ask"] == (ROOT / "suite" / "AC1-brief.md").read_text(encoding="utf-8")


def test_ac1_ask_must_be_the_frozen_brief(tmp_path, monkeypatch):
    fake_stack(monkeypatch, tmp_path)
    suite = suite_source()
    next(t for t in suite["tasks"] if t["id"] == "AC1")["ask"] += "\nEdited."
    edited = tmp_path / "ads-suite.json"
    edited.write_text(json.dumps(suite), encoding="utf-8")
    monkeypatch.setattr(freeze, "SUITE_SOURCE", edited)
    with pytest.raises(ValueError, match="AC1-brief.md"):
        freeze.freeze(tmp_path / "edited-ac1", MODELS, GRADER)
    assert not (tmp_path / "edited-ac1").exists()


def test_user_global_servers_disabled(frozen):
    campaign, _ = frozen
    servers = json.loads(freeze.USER_MCP.read_text(encoding="utf-8"))["mcpServers"] if freeze.USER_MCP.is_file() else {}
    local = json.loads((campaign / ".omp" / "mcp.json").read_text(encoding="utf-8"))
    assert local == {"mcpServers": {name: {"enabled": False} for name in sorted(servers)}}


def test_refusals(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        freeze.freeze(existing, MODELS, GRADER)
    for grader in ("", "unset", None):
        with pytest.raises(ValueError, match="grader"):
            freeze.freeze(tmp_path / "no-grader", MODELS, grader)
    assert stack.runner.log == [] and stack.bridge.keys == []
    assert not (tmp_path / "no-grader").exists()


def test_non_emulator_profile_refused_and_own_stack_stopped(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path, machine={**MACHINE, "profile": "ADS_MACHINE"})
    with pytest.raises(ValueError, match="ADS_MACHINE"):
        freeze.freeze(tmp_path / "machine", MODELS, GRADER)
    assert "tools/list:None" not in stack.bridge.calls
    assert stack.runner.procs == {} and not (tmp_path / "machine").exists()


def test_foreign_stack_refused_untouched(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path, running=(CONTROL_IMAGE,))
    with pytest.raises(RuntimeError, match="already running"):
        freeze.freeze(tmp_path / "foreign", MODELS, GRADER)
    assert [entry[0] for entry in stack.runner.log] == [] and set(stack.runner.procs.values()) == {CONTROL_IMAGE}
    assert stack.bridge.keys == [] and not (tmp_path / "foreign").exists()


def test_stack_not_ready_stops_own_stack(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    clock = iter(range(0, 1000, 10))

    async def closed():
        return False
    monkeypatch.setattr(freeze, "query_bridge", functools.partial(
        freeze.query_bridge.func, runner=stack.runner, port_open=closed, sleep=no_sleep,
        sync_sleep=lambda _s: None, clock=lambda: next(clock)))
    with pytest.raises(StackError) as error:
        freeze.freeze(tmp_path / "not-ready", MODELS, GRADER)
    assert error.value.code == "stack_not_ready"
    assert stack.runner.procs == {} and stack.bridge.keys == [] and not (tmp_path / "not-ready").exists()


def test_profile_changed_by_the_running_stack_refused(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    kill = stack.runner.kill

    def kill_and_write(pid):
        (stack.paths.profile_dir / "Sources.xml").write_text("<Sources written='on exit'/>", encoding="utf-8")
        kill(pid)
    monkeypatch.setattr(stack.runner, "kill", kill_and_write)
    with pytest.raises(ValueError, match="profile"):
        freeze.freeze(tmp_path / "drift", MODELS, GRADER)
    assert not (tmp_path / "drift").exists()


def test_missing_live_tool_removes_campaign(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    original = stack.bridge.list_tools

    async def without_tail_log(cursor=None):
        page = await original(cursor)
        return SimpleNamespace(tools=[t for t in page.tools if t.name != "tail_log"], nextCursor=page.nextCursor)
    stack.bridge.list_tools = without_tail_log
    with pytest.raises(ValueError, match="tail_log"):
        freeze.freeze(tmp_path / "missing-tool", MODELS, GRADER)
    assert not (tmp_path / "missing-tool").exists() and stack.runner.procs == {}


def test_cli_models_and_grader(tmp_path, monkeypatch):
    fake_stack(monkeypatch, tmp_path)
    campaign = tmp_path / "cli"
    assert freeze.main(["--campaign", str(campaign), "--models", "anthropic/claude-sonnet-5-5",
                        "strata/qwen3.8-flash-next-iq2_xs:off", "--grader", GRADER]) == 0
    assert load_manifest(campaign).models == [{"selector": "anthropic/claude-sonnet-5-5", "thinking": "high"},
                                              {"selector": "strata/qwen3.8-flash-next-iq2_xs", "thinking": "off"}]
    with pytest.raises(SystemExit):
        freeze.main(["--campaign", str(tmp_path / "cli2"), "--models", "a/b"])
    with pytest.raises(ValueError, match="grader"):
        freeze.main(["--campaign", str(tmp_path / "cli3"), "--models", "a/b", "--grader", "unset"])


def test_grader_in_roster_refused_before_stack_or_directory(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="grader"):
        freeze.freeze(tmp_path / "self-graded", MODELS, MODELS[1]["selector"])
    assert stack.runner.log == [] and stack.bridge.keys == [] and not (tmp_path / "self-graded").exists()


def test_freeze_creates_results_dir(frozen):
    campaign, _ = frozen
    assert (campaign / "results").is_dir()


def test_brief_and_rubric_frozen_as_inputs(frozen):
    campaign, _ = frozen
    settings = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    for name in ("grader-brief.md", "ADS-Benchmark.md"):
        assert (campaign / "inputs" / name).read_bytes() == (ROOT / "suite" / name).read_bytes()
        assert settings["input_sha256"]["inputs/" + name] == sha(ROOT / "suite" / name)
    brief = (campaign / "inputs" / "grader-brief.md").read_text(encoding="utf-8")
    assert "`<campaign>\\inputs\\ADS-Benchmark.md`" in brief and "LLM-Benchmarks\\ADS\\suite" not in brief


def test_no_secret_reaches_campaign_json(frozen):
    campaign, _ = frozen
    text = (campaign / "campaign.json").read_text(encoding="utf-8")
    assert "certSha256" not in text and "apiKey" not in text and "APS_DEV_AUTOLOGIN" not in text


def test_launch_smoke_runs_every_task_for_every_model(frozen, monkeypatch):
    import asyncio
    import run_campaign
    campaign, _ = frozen
    calls = []

    class Stack:
        async def prepare(self, previous, task):
            return {"type": "stack_verified"}

        async def finish(self, dest, started_utc):
            return []

        async def close(self):
            return None

    async def fake_trial(*, campaign, task, model, fingerprint, on_cost, stack):
        assert (campaign / "results").is_dir()
        calls.append((task["id"], model["selector"]))
        return {"trial_id": run_campaign.trial_id(task, model), "fingerprint": fingerprint, "task": task["id"],
                "selector": model["selector"], "status": "completed", "duration_s": 0.0,
                "catalog_estimated_cost_usd": 0.0, "error_category": None}
    monkeypatch.setattr(run_campaign, "run_trial", fake_trial)
    assert asyncio.run(run_campaign.launch(campaign, smoke=True, stack=Stack())) == 0
    manifest = load_manifest(campaign)
    assert sorted(calls) == sorted((t, m["selector"]) for t in manifest.task_ids for m in manifest.models)
    assert len(manifest.task_ids) > 1
    assert (campaign / "smoke-summary.json").is_file() and (campaign / "smoke-progress.json").is_file()


# Final review: freeze records each key config's destination (host/port, never the key).
def test_freeze_records_bridge_endpoints_without_keys(frozen):
    campaign, _ = frozen
    settings = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    assert settings["bridge_endpoints"] == {k: {"host": "127.0.0.1", "port": 8026} for k in ("read", "recipe", "manual")}
    assert "dummy" not in (campaign / "campaign.json").read_text(encoding="utf-8")


def test_freeze_refuses_a_bridge_config_that_is_not_local(tmp_path, monkeypatch):
    stack = fake_stack(monkeypatch, tmp_path)
    write_bridge_configs(stack.secrets, {"host": "10.0.0.5", "port": 8026})
    with pytest.raises(ValueError, match="does not point at this workstation"):
        freeze.freeze(tmp_path / "ads-test", MODELS, GRADER)
    assert not (tmp_path / "ads-test").exists() and stack.runner.log == []


@pytest.mark.parametrize("where", ["worker", "finish"])
def test_fatal_bridge_refusal_after_a_verified_stack_stops_the_campaign(frozen, monkeypatch, where):
    import asyncio
    import run_campaign
    from ads_stack import StackError
    campaign, _ = frozen
    calls, closed = [], []

    class Stack:
        async def prepare(self, previous, task):
            return {"type": "stack_verified"}

        async def finish(self, dest, started_utc):
            if where == "finish":
                raise StackError("profile_not_emulator")
            return []

        async def close(self):
            closed.append(True)

    async def fake_trial(*, campaign, task, model, fingerprint, on_cost, stack):
        await stack()
        calls.append(task["id"])
        return {"trial_id": run_campaign.trial_id(task, model), "fingerprint": fingerprint, "task": task["id"],
                "selector": model["selector"], "status": "failed" if where == "worker" else "completed",
                "duration_s": 0.0, "catalog_estimated_cost_usd": 0.0,
                "error_category": "bridge_not_emulator" if where == "worker" else None}
    monkeypatch.setattr(run_campaign, "run_trial", fake_trial)
    with pytest.raises(StackError):
        asyncio.run(run_campaign.launch(campaign, smoke=True, stack=Stack()))
    assert len(calls) == 1 and closed


def test_freeze_sputter_and_cluster_tracks_without_recipes_dir(tmp_path, monkeypatch):
    from test_ads_stack import StackPaths, write_root_config
    secrets = tmp_path / "secrets"
    write_bridge_configs(secrets)
    monkeypatch.setattr(freeze, "default_secrets_dir", lambda: secrets)
    for track, prof, expected_count in (("sputter", "Sputter", 12), ("cluster", "Cluster", 12)):
        prof_dir = tmp_path / "configs" / prof
        prof_dir.mkdir(parents=True)
        (prof_dir / "Sources.xml").write_text("<Sources/>", encoding="utf-8")
        snap = tmp_path / "snapshots" / f"{prof}_Idle.emstate"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text("{}", encoding="utf-8")
        bin_dir = tmp_path / f"bin-{track}"
        (bin_dir / "Data").mkdir(parents=True)
        (bin_dir / "Logs").mkdir()
        (bin_dir / "AgentServer.xml").write_text('<AgentServer enabled="true" port="8026"/>', encoding="utf-8")
        write_root_config(bin_dir / "RootConfig.xml", prof_dir.parent, prof)
        paths = StackPaths(
            emulator=bin_dir / EMULATOR_IMAGE,
            control=bin_dir / CONTROL_IMAGE,
            profile_dir=prof_dir,
            agent_server_xml=bin_dir / "AgentServer.xml",
            root_config=bin_dir / "RootConfig.xml",
            snapshot=snap,
            data_dir=bin_dir / "Data",
            logs_dir=bin_dir / "Logs",
        )
        machine = {"profile": prof, "appVersion": "1.7.1", "controlMode": "Remote", "isEmulator": True}
        runner, bridge = FakeRunner(), FakeBridge(machine)
        monkeypatch.setattr(freeze, "bridge_session", bridge)
        monkeypatch.setattr(freeze, "query_bridge", functools.partial(
            freeze.query_bridge, runner=runner, port_open=port_open, sleep=no_sleep, sync_sleep=lambda _s: None))
        out = freeze.freeze(tmp_path / f"camp-{track}", MODELS, GRADER, track=track, stack_paths=paths)
        manifest = load_manifest(out)
        assert len(manifest.task_ids) == expected_count
        assert (out / "inputs" / "recipes-baseline").is_dir()
