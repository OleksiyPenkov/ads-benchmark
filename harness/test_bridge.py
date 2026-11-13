"""Bridge per key: tool sets per tier, task key checks, per-key configs and the secret-leak rule."""
import asyncio
import contextlib
import functools
import hashlib
import json
from pathlib import Path

import pytest

import bridge
from bridge import BRIDGE, KEY_TOOLS, bridge_params, check_task_tools, default_secrets_dir

READ = ["describe_machine", "describe_hardware", "describe_recipes", "describe_dsl", "describe_interlocks",
        "describe_alarms", "system_status", "device_snapshot", "tail_log", "recipe_status", "list_runs",
        "fetch_run_files", "compute_deposition_plan"]
RECIPE_EXTRA = ["validate_recipe", "upload_recipe", "start_recipe", "abort_recipe", "resolve_pause"]
SENTINEL = "SENTINEL-apiKey-7c1e9b44-do-not-journal"
MODEL = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}


def test_key_tool_lists_match_spec_6_2():
    assert KEY_TOOLS["read"] == READ
    assert KEY_TOOLS["recipe"] == READ + RECIPE_EXTRA
    assert KEY_TOOLS["manual"] == READ + RECIPE_EXTRA + ["device_command"]
    assert set(KEY_TOOLS) == {"read", "recipe", "manual"}


def test_read_key_task_with_start_recipe_refused():
    task = {"id": "A01", "type": "short", "key": "read", "tools": ["describe_machine", "start_recipe"]}
    with pytest.raises(ValueError, match="start_recipe"):
        check_task_tools(task)


def test_recipe_key_short_task_with_upload_recipe_refused():
    task = {"id": "A10", "type": "short", "key": "recipe", "tools": ["describe_dsl", "validate_recipe", "upload_recipe"]}
    with pytest.raises(ValueError, match="upload_recipe"):
        check_task_tools(task)


def test_recipe_key_short_task_with_validate_recipe_accepted():
    check_task_tools({"id": "A10", "type": "short", "key": "recipe", "tools": ["describe_dsl", "validate_recipe"]})


def test_recipe_key_capstone_gets_mutating_tools():
    check_task_tools({"id": "AC1", "type": "capstone", "key": "recipe", "tools": READ + RECIPE_EXTRA})


def test_manual_task_with_device_command_accepted():
    # A17/A18 give the Manual key on purpose; a mutating call is scored from the trace, not blocked here.
    check_task_tools({"id": "A17", "type": "short", "key": "manual",
                      "tools": ["describe_machine", "system_status", "start_recipe", "device_command"]})


@pytest.mark.parametrize("key", [None, "", "Read", "lab-read", "admin"])
def test_task_without_valid_key_refused(key):
    task = {"id": "A01", "type": "short", "tools": ["describe_machine"]}
    if key is not None:
        task["key"] = key
    with pytest.raises(ValueError, match="key"):
        check_task_tools(task)


def test_registered_local_tools_are_exempt():
    check_task_tools({"id": "A07", "type": "short", "key": "read", "tools": ["describe_machine", "skill_read"]},
                     local_tools={"skill_read"})


@pytest.mark.parametrize("name", ["skill_read", "list_measurements", "fit_xrr"])
def test_unregistered_tool_outside_key_refused(name):
    # Only manifest-registered local tools are exempt; any other name must be in the key's set.
    with pytest.raises(ValueError, match=name):
        check_task_tools({"id": "A07", "type": "short", "key": "manual", "tools": ["describe_machine", name]})


def test_load_suite_refuses_remote_tool_outside_key(tmp_path):
    from manifest import load_manifest, load_suite
    from test_manifest import make_campaign
    declarations = [{"name": name, "description": name, "loadMode": "essential",
                     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
                    for name in ("describe_machine", "list_measurements")]
    task = {"id": "X01", "tier": 1, "type": "short", "key": "read", "ask": "List.", "tools": ["list_measurements"]}
    campaign = make_campaign(tmp_path, tasks=[task], declarations=declarations,
                             tools=[{"name": "describe_machine", "cls": "read"},
                                    {"name": "list_measurements", "cls": "read"}])
    with pytest.raises(ValueError, match="list_measurements"):
        load_suite(load_manifest(campaign))


def test_bridge_params_per_key(tmp_path):
    for key in ("read", "recipe", "manual"):
        params = bridge_params(key, tmp_path)
        assert params.command == str(BRIDGE) and params.command.endswith("APS.Mcp.exe")
        assert params.args == ["--config", str(tmp_path / f"apsmcp.{key}.json")]
    with pytest.raises(ValueError):
        bridge_params("admin", tmp_path)


def test_default_secrets_dir_resolved_at_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_secrets_dir() == tmp_path / "llm-bench-ads"
    monkeypatch.delenv("LOCALAPPDATA")
    with pytest.raises(RuntimeError):
        default_secrets_dir()


# Ruling 4: every suite task declares a key, checked when the suite is loaded.
def test_load_suite_requires_task_key(tmp_path):
    from manifest import load_manifest, load_suite
    from test_manifest import SHORT, make_campaign
    task = {k: v for k, v in SHORT.items() if k != "key"}
    campaign = make_campaign(tmp_path, tasks=[task])
    with pytest.raises(ValueError, match="key"):
        load_suite(load_manifest(campaign))


def test_load_suite_refuses_tool_above_key(tmp_path):
    from manifest import load_manifest, load_suite
    from test_manifest import make_campaign
    declarations = [{"name": name, "description": name, "loadMode": "essential",
                     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
                    for name in ("describe_machine", "start_recipe")]
    task = {"id": "X01", "tier": 1, "type": "short", "key": "read", "ask": "Run it.",
            "tools": ["describe_machine", "start_recipe"]}
    campaign = make_campaign(tmp_path, tasks=[task], declarations=declarations,
                             tools=[{"name": "describe_machine", "cls": "read"}, {"name": "start_recipe", "cls": "write"}])
    with pytest.raises(ValueError, match="start_recipe"):
        load_suite(load_manifest(campaign))


# Ruling 3: the worker opens the bridge with the task's key config.
def test_load_trial_server_is_task_key_bridge(tmp_path, monkeypatch):
    from campaign_worker import load_trial
    from test_manifest import CAPSTONE, SHORT, make_campaign, sha
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    campaign = make_campaign(tmp_path)
    secrets = tmp_path / "local" / "llm-bench-ads"
    fingerprint = sha(campaign / "campaign.json")
    assert load_trial(campaign, SHORT, MODEL, fingerprint)[3] == bridge_params("read", secrets)
    assert load_trial(campaign, CAPSTONE, MODEL, fingerprint)[3] == bridge_params("recipe", secrets)


def test_load_trial_refuses_other_mcp_server(tmp_path, monkeypatch):
    from campaign_worker import TrialFailure, load_trial
    from test_manifest import SHORT, make_campaign, sha
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    campaign = make_campaign(tmp_path, mcp_server={"command": str((tmp_path / "other" / "server.exe").resolve())})
    with pytest.raises(TrialFailure, match="mcp_server_mismatch"):
        load_trial(campaign, SHORT, MODEL, sha(campaign / "campaign.json"))


def test_child_trial_opens_bridge_with_task_key(tmp_path, monkeypatch):
    import mcp
    import mcp.client.stdio

    import campaign_worker
    from test_manifest import SHORT, FakeClientSession, make_campaign, sha, write_bridge_configs

    opened = []

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        opened.append(params)
        raise campaign_worker.TrialFailure("bridge_captured")
        yield None, None

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    write_bridge_configs(tmp_path / "local" / "llm-bench-ads")
    campaign = make_campaign(tmp_path)

    async def scenario():
        trial = asyncio.create_task(campaign_worker.child_trial(
            campaign, SHORT, MODEL, sha(campaign / "campaign.json"), tmp_path / "trial.jsonl"))
        for _ in range(500):
            if opened:
                break
            await asyncio.sleep(0.01)
        trial.cancel()
        await asyncio.gather(trial, return_exceptions=True)

    asyncio.run(scenario())
    assert opened == [bridge_params("read", tmp_path / "local" / "llm-bench-ads")]


def test_campaign_stack_uses_read_probe_and_recipe_abort(tmp_path, monkeypatch):
    import ads_stack
    import run_campaign
    captured = {}

    def fake_for_campaign(campaign, input_sha256, *, read_session, recipe_session):
        captured.update(read=read_session, recipe=recipe_session)
        return "stack"

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(ads_stack.StackController, "for_campaign", fake_for_campaign)
    endpoints = {k: {"host": "127.0.0.1", "port": 8026} for k in ("read", "recipe", "manual")}
    manifest = type("M", (), {"input_sha256": {}, "bridge_endpoints": endpoints})()
    assert run_campaign.campaign_stack(tmp_path / "campaign", manifest) == "stack"
    secrets = tmp_path / "llm-bench-ads"
    for role, key in (("read", "read"), ("recipe", "recipe")):
        session = captured[role]
        assert isinstance(session, functools.partial) and session.func is ads_stack.bridge_session
        assert session.args == (key, secrets, endpoints) and not session.keywords


def test_bridge_session_opens_key_config(tmp_path, monkeypatch):
    import mcp
    import mcp.client.stdio

    import ads_stack
    from test_manifest import FakeClientSession
    opened = []

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        opened.append(params)
        yield None, None

    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)

    async def scenario():
        async with ads_stack.bridge_session("recipe", tmp_path):
            pass

    asyncio.run(scenario())
    assert opened == [bridge_params("recipe", tmp_path)]


# Ruling 1-2: config generation.
def _key_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _key_fixture(tmp_path, tiers=None):
    secrets = {"lab-read": "secret-read-1", "paper-arm-emulator": "secret-recipe-2", "behavior-validation": "secret-manual-3"}
    tiers = tiers or {"lab-read": "Read", "paper-arm-emulator": "Recipe", "behavior-validation": "Manual"}
    agent_server = tmp_path / "AgentServer.xml"
    agent_server.write_text('<AgentServer enabled="true" port="8026">\n' + "".join(
        f'  <Key name="{name}" hash="{_key_hash(value)}" tier="{tiers[name]}"/>\n' for name, value in secrets.items())
        + "</AgentServer>\n", encoding="utf-8")
    template = tmp_path / "apsmcp.json"
    template.write_text(json.dumps({"host": "127.0.0.1", "port": 8026, "apiKey": secrets["behavior-validation"],
                                    "certSha256": "CERT", "downloadDir": "/tmp/runs"}), encoding="utf-8-sig")
    lab = tmp_path / "apsmqtt.lab.json"
    lab.write_text(json.dumps({"control": {"apiKey": secrets["lab-read"]}}), encoding="utf-8")
    arm = tmp_path / "apsmcp.emulator.json"
    arm.write_text(json.dumps({"apiKey": secrets["paper-arm-emulator"]}), encoding="utf-8")
    return secrets, dict(agent_server=agent_server, template=template, sources=(template, lab, arm))


def test_make_configs_replaces_only_api_key(tmp_path):
    from make_bridge_configs import make_configs
    secrets, paths = _key_fixture(tmp_path)
    out = tmp_path / "secrets"
    written = make_configs(out, **paths)
    assert written == [(out / "apsmcp.read.json", "lab-read"), (out / "apsmcp.recipe.json", "paper-arm-emulator"),
                       (out / "apsmcp.manual.json", "behavior-validation")]
    base = json.loads(paths["template"].read_text(encoding="utf-8-sig"))
    for path, name in written:
        assert json.loads(path.read_text(encoding="utf-8")) == {**base, "apiKey": secrets[name]}


@pytest.mark.parametrize("name,tier", [("lab-read", "Recipe"), ("paper-arm-emulator", "Manual"),
                                       ("behavior-validation", "Read")])
def test_make_configs_refuses_wrong_tier(tmp_path, name, tier):
    from make_bridge_configs import make_configs
    tiers = {"lab-read": "Read", "paper-arm-emulator": "Recipe", "behavior-validation": "Manual", name: tier}
    _, paths = _key_fixture(tmp_path, tiers)
    out = tmp_path / "secrets"
    with pytest.raises(ValueError, match=name):
        make_configs(out, **paths)
    assert not out.exists()


def test_make_configs_refuses_unknown_key(tmp_path):
    from make_bridge_configs import make_configs
    _, paths = _key_fixture(tmp_path)
    paths["sources"] = paths["sources"][:2]
    out = tmp_path / "secrets"
    with pytest.raises(ValueError, match="paper-arm-emulator"):
        make_configs(out, **paths)
    assert not out.exists()


def test_make_configs_main_prints_paths_and_names_only(tmp_path, monkeypatch, capsys):
    import make_bridge_configs
    secrets, paths = _key_fixture(tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(make_bridge_configs, "make_configs",
                        functools.partial(make_bridge_configs.make_configs, **paths))
    assert make_bridge_configs.main() == 0
    out = capsys.readouterr()
    for value in secrets.values():
        assert value not in out.out + out.err and _key_hash(value) not in out.out + out.err
    assert "lab-read" in out.out and str(tmp_path / "local" / "llm-bench-ads" / "apsmcp.read.json") in out.out


# Ruling 5: a trial whose bridge config holds a sentinel apiKey never writes it into the campaign.
SITECUSTOMIZE = '''
import asyncio, contextlib, json, os
from types import SimpleNamespace
import mcp, mcp.client.stdio

@contextlib.asynccontextmanager
async def fake_stdio_client(params, errlog=None):
    # Stands in for APS.Mcp: it reads its --config, as the real bridge does.
    config = params.args[params.args.index("--config") + 1]
    with open(config, encoding="utf-8") as source:
        key = json.load(source)["apiKey"]
    with open(os.environ["BENCH_TEST_BRIDGE_SEEN"], "w", encoding="utf-8") as seen:
        seen.write(key)
    yield None, None

class FakeClientSession:
    def __init__(self, reader, writer):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False
    async def initialize(self):
        info = SimpleNamespace(model_dump=lambda **kwargs: {"name": "APS.Mcp", "version": "fake"})
        return SimpleNamespace(protocolVersion="test", serverInfo=info)
    async def list_tools(self, cursor=None):
        with open(os.path.join(os.environ["BENCH_CAMPAIGN_TEST_DIR"], "inputs", "tools.json"), encoding="utf-8") as f:
            declarations = json.load(f)
        tools = [SimpleNamespace(name=d["name"], inputSchema=d["parameters"], description=d["description"])
                 for d in declarations if d["name"] != "skill_read"]
        return SimpleNamespace(tools=tools, nextCursor=None)

async def no_model(*args, **kwargs):
    raise RuntimeError("model process not started in this test")

mcp.ClientSession = FakeClientSession
mcp.client.stdio.stdio_client = fake_stdio_client
asyncio.create_subprocess_exec = no_model
'''


def test_trial_never_journals_the_api_key(tmp_path, monkeypatch):
    import campaign_worker
    from test_manifest import SHORT, make_campaign, sha
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    secrets = tmp_path / "local" / "llm-bench-ads"
    secrets.mkdir(parents=True)
    (secrets / "apsmcp.read.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": 8026, "apiKey": SENTINEL, "certSha256": "CERT"}), encoding="utf-8")
    seen = tmp_path / "bridge-seen.txt"
    campaign = make_campaign(tmp_path)
    (campaign / "results").mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setenv("BENCH_TEST_BRIDGE_SEEN", str(seen))
    monkeypatch.setenv("BENCH_CAMPAIGN_TEST_DIR", str(campaign))
    result = asyncio.run(campaign_worker.run_trial(campaign=campaign, task=SHORT, model=MODEL,
                                                   fingerprint=sha(campaign / "campaign.json"),
                                                   on_cost=lambda *_: None))
    # Positive control: the fake bridge did receive the sentinel key through the task's config.
    assert seen.read_text(encoding="utf-8") == SENTINEL
    journal = Path(result["journal_file"])
    types = [json.loads(line)["type"] for line in journal.read_text(encoding="utf-8").splitlines()]
    assert "mcp_initialized" in types and "mcp_declarations_verified" in types and types[-1] == "result"
    sidecar = journal.with_suffix(".json")
    assert sidecar.is_file()
    for path in campaign.rglob("*"):
        if path.is_file():
            assert SENTINEL.encode("utf-8") not in path.read_bytes(), path
