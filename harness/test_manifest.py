import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bridge import BRIDGE
from local_tools import SKILL_READ

HARNESS = Path(__file__).resolve().parent
RUNNERS = ("campaign_worker.py", "campaign_core.py", "manifest.py", "campaign_guard.ts", "guard_policy.ts",
           "run_campaign.py", "local_tools.py", "ads_stack.py", "bridge.py")

SHORT = {"id": "X01", "tier": 1, "type": "short", "key": "read", "ask": "Describe the machine.",
         "tools": ["describe_machine"]}
CAPSTONE = {"id": "XC1", "tier": 5, "type": "capstone", "key": "recipe", "ask": "Run the recipe.",
            "tools": ["describe_machine", "start_recipe", "skill_read"]}
DECLARATIONS = [
    {"name": "describe_machine", "description": "Describe the machine.", "loadMode": "essential",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "start_recipe", "description": "Start a recipe.", "loadMode": "essential",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}},
    SKILL_READ,
]


ENDPOINT = {"host": "127.0.0.1", "port": 8026}


def write_bridge_configs(secrets_dir, endpoint=ENDPOINT):
    """apsmcp.<key>.json for the three keys with a dummy key; only host/port matter to the harness."""
    secrets_dir.mkdir(parents=True, exist_ok=True)
    for key in ("read", "recipe", "manual"):
        (secrets_dir / f"apsmcp.{key}.json").write_text(json.dumps({**endpoint, "apiKey": "dummy"}), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_campaign(tmp_path, *, tasks=None, declarations=None, **overrides):
    """A complete, self-consistent campaign; `overrides` replace campaign.json fields."""
    campaign = tmp_path / "campaign"
    inputs = campaign / "inputs"
    inputs.mkdir(parents=True)
    tasks = [SHORT, CAPSTONE] if tasks is None else tasks
    (inputs / "suite.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    (inputs / "system_prompt.md").write_text("You are an XRR analyst.\n", encoding="utf-8")
    (inputs / "tools.json").write_text(json.dumps(DECLARATIONS if declarations is None else declarations),
                                       encoding="utf-8")
    (inputs / "runner-config.yml").write_text("retry:\n  enabled: false\n", encoding="utf-8")
    settings = {
        "campaign_id": "test-campaign",
        "models": [{"selector": "anthropic/claude-opus-5-5", "thinking": "high"}],
        "task_ids": [task["id"] for task in tasks],
        "task_pattern": "^X(0[1-9]|1[0-6])$|^XC[12]$",
        "repetitions": 1,
        "parallelism": 1,
        "cost_threshold_usd": 50,
        "timeouts": {"short": 900, "capstone": 3600},
        "max_output_tokens": 32768,
        "tool_prefix": "aps_",
        "tools": [{"name": "describe_machine", "cls": "read"}, {"name": "start_recipe", "cls": "write"},
                  {"name": "skill_read", "cls": "local"}],
        "mcp_server": {"command": BRIDGE.as_posix()},
        "bridge_endpoints": {key: dict(ENDPOINT) for key in ("read", "recipe", "manual")},
        "skill_snapshot": "inputs/skill",
        "grader_model": "unset",
        "input_sha256": {"inputs/" + path.name: sha(path) for path in sorted(inputs.iterdir())},
        "runner_sha256": {(HARNESS / name).as_posix(): sha(HARNESS / name) for name in RUNNERS},
    }
    settings.update(overrides)
    (campaign / "campaign.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return campaign


def test_fixture_is_accepted(tmp_path):
    from manifest import load_manifest, load_suite
    campaign = make_campaign(tmp_path)
    manifest = load_manifest(campaign)
    assert manifest.fingerprint == sha(campaign / "campaign.json")
    assert [tool.cls for tool in manifest.tools] == ["read", "write", "local"]
    assert list(load_suite(manifest)) == ["X01", "XC1"]


def test_validator_argv_is_not_a_manifest_field(tmp_path):
    from manifest import Manifest, load_manifest
    assert "validator_argv" not in Manifest.__dataclass_fields__
    assert not hasattr(load_manifest(make_campaign(tmp_path)), "validator_argv")


def test_changed_input_rejected(tmp_path):
    from manifest import load_manifest
    campaign = make_campaign(tmp_path)
    path = campaign / "inputs/system_prompt.md"
    path.write_bytes(path.read_bytes().replace(b"XRR", b"XRF"))
    with pytest.raises(ValueError, match="inputs/system_prompt.md"):
        load_manifest(campaign)


@pytest.mark.parametrize("parallelism", [0, 2, 3, True, 1.0])
def test_parallelism_other_than_one_rejected(tmp_path, parallelism):
    from manifest import load_manifest
    campaign = make_campaign(tmp_path, parallelism=parallelism)
    with pytest.raises(ValueError, match="parallelism"):
        load_manifest(campaign)


def test_input_outside_inputs_rejected(tmp_path):
    from manifest import load_manifest
    outside = tmp_path / "x.json"
    outside.write_text("{}", encoding="utf-8")
    probe = make_campaign(tmp_path / "probe")
    inputs = json.loads((probe / "campaign.json").read_text(encoding="utf-8"))["input_sha256"]
    campaign = make_campaign(tmp_path, input_sha256={**inputs, "../x.json": sha(outside)})
    with pytest.raises(ValueError, match=r"\.\./x\.json"):
        load_manifest(campaign)


def test_runner_outside_harness_rejected(tmp_path):
    from manifest import load_manifest
    outside = tmp_path / "runner.py"
    outside.write_text("pass\n", encoding="utf-8")
    probe = make_campaign(tmp_path / "probe")
    runners = json.loads((probe / "campaign.json").read_text(encoding="utf-8"))["runner_sha256"]
    campaign = make_campaign(tmp_path, runner_sha256={**runners, outside.as_posix(): sha(outside)})
    with pytest.raises(ValueError, match="runner.py"):
        load_manifest(campaign)


def test_task_id_pattern(tmp_path):
    from manifest import load_manifest, load_suite
    task = {**SHORT, "id": "X17"}
    campaign = make_campaign(tmp_path, tasks=[task])
    with pytest.raises(ValueError, match="X17"):
        load_suite(load_manifest(campaign))


def test_unknown_task_tool(tmp_path):
    from manifest import load_manifest, load_suite
    campaign = make_campaign(tmp_path, tasks=[{**SHORT, "tools": ["bash"]}])
    with pytest.raises(ValueError, match="bash"):
        load_suite(load_manifest(campaign))


def test_timeout_by_type(tmp_path):
    from manifest import load_manifest, task_timeout
    manifest = load_manifest(make_campaign(tmp_path))
    assert task_timeout(manifest, SHORT) == 900
    assert task_timeout(manifest, CAPSTONE) == 3600
    assert task_timeout(manifest, {**SHORT, "timeout_s": 1200}) == 1200


def test_load_trial_uses_manifest(tmp_path):
    from campaign_worker import load_trial
    campaign = make_campaign(tmp_path)
    fingerprint = sha(campaign / "campaign.json")
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    manifest, declarations, guard, server, local = load_trial(campaign, SHORT, model, fingerprint)
    assert manifest.timeouts == {"short": 900, "capstone": 3600}
    assert sorted(guard.names) == ["aps_describe_machine"]
    capstone_guard = load_trial(campaign, CAPSTONE, model, fingerprint)[2]
    assert sorted(capstone_guard.names) == ["aps_describe_machine", "aps_skill_read", "aps_start_recipe"]
    assert server.command == str(BRIDGE) and server.args[0] == "--config"
    assert [d["name"] for d in declarations] == ["describe_machine", "start_recipe", "skill_read"]
    assert [d["name"] for d in local.declarations()] == ["skill_read"]
    assert local.snapshot == (campaign / "inputs/skill").resolve()


def test_run_campaign_loads_manifest(tmp_path):
    import run_campaign
    campaign = make_campaign(tmp_path)
    manifest, settings, tasks = run_campaign.load_campaign(campaign)
    assert settings["campaign_id"] == "test-campaign"
    assert list(tasks) == manifest.task_ids == ["X01", "XC1"]


class FakeSession:
    def __init__(self, declarations):
        self.tools = [SimpleNamespace(name=d["name"], inputSchema=d["parameters"], description=d["description"])
                      for d in declarations]

    async def list_tools(self, cursor=None):
        return SimpleNamespace(tools=self.tools, nextCursor=None)


def test_discover_ignores_local_tools():
    from campaign_worker import TrialFailure, discover
    from manifest import ToolSpec
    tools = [ToolSpec("describe_machine", "read"), ToolSpec("start_recipe", "write"), ToolSpec("skill_read", "local")]
    live = FakeSession(DECLARATIONS[:2])
    asyncio.run(discover(live, DECLARATIONS, tools))
    missing_remote = FakeSession(DECLARATIONS[:1])
    with pytest.raises(TrialFailure, match="live_mcp_declaration_drift"):
        asyncio.run(discover(missing_remote, DECLARATIONS, tools))


class FakeClientSession:
    profile = "ADS"

    def __init__(self, reader, writer):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return SimpleNamespace(protocolVersion="test", serverInfo=SimpleNamespace(model_dump=lambda **kwargs: {}))

    async def call_tool(self, name, arguments=None):
        self.calls.append(name)
        if name != "describe_machine":
            raise AssertionError("unexpected bridge call " + name)
        text = json.dumps({"profile": self.profile, "appVersion": "test", "controlMode": "Remote"})
        return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text=text)])


def test_child_trial_passes_task_to_guard(tmp_path, monkeypatch):
    import contextlib

    import mcp
    import mcp.client.stdio

    import campaign_worker

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        yield None, None

    async def fake_discover(session, declarations, tools):
        return None

    captured = {}
    spawned = asyncio.Event()

    async def fake_spawn(*args, **kwargs):
        captured.update(kwargs["env"])
        spawned.set()
        raise campaign_worker.TrialFailure("spawn_captured")

    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(campaign_worker, "discover", fake_discover)
    monkeypatch.setattr(campaign_worker.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    write_bridge_configs(tmp_path / "local" / "llm-bench-ads")
    campaign = make_campaign(tmp_path)
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}

    async def scenario():
        # child_trial stays alive after its result until the wrapper kills it.
        trial = asyncio.create_task(campaign_worker.child_trial(
            campaign, SHORT, model, sha(campaign / "campaign.json"), tmp_path / "trial.jsonl"))
        await asyncio.wait_for(spawned.wait(), timeout=10)
        trial.cancel()
        await asyncio.gather(trial, return_exceptions=True)

    asyncio.run(scenario())
    assert captured["BENCH_CAMPAIGN_TASK"] == "X01"
    assert captured["BENCH_CAMPAIGN_DIR"] == str(campaign)


def test_safe_stderr_keeps_bench_policy_marker():
    from campaign_worker import safe_stderr
    record = {"selector": "a/b", "model": "a/b", "provider": "a", "api": "anthropic-messages",
              "output_token_cap": 32768, "request": 1}
    data = ("noise\nOTHER_POLICY " + json.dumps({**record, "request": 2}) + "\nBENCH_CAMPAIGN_POLICY "
            + json.dumps(record) + "\n").encode("utf-8")

    async def scenario():
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        events = []
        await safe_stderr(stream, SimpleNamespace(emit=events.append))
        return events

    assert asyncio.run(scenario()) == [{"type": "provider_policy", **record}]


def test_safe_stderr_journals_guard_denials():
    from campaign_worker import safe_stderr
    data = (b"BENCH_CAMPAIGN_DENIED tool_admission_failed\n"
            b"BENCH_CAMPAIGN_DENIED abort_failed\n"
            b"BENCH_CAMPAIGN_DENIED secret provider text\n"
            b"ELN_CAMPAIGN_DENIED tool_admission_failed\n")

    async def scenario():
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        events = []
        await safe_stderr(stream, SimpleNamespace(emit=events.append))
        return events

    assert asyncio.run(scenario()) == [
        {"type": "guard_denied", "category": "tool_admission_failed"},
        {"type": "guard_denied", "category": "abort_failed"},
        {"type": "guard_denied", "category": "unrecognized"},
    ]


def test_task_forbidden_call_is_denied_and_journaled(tmp_path):
    from campaign_worker import Trial, TrialFailure, load_trial, task_declarations
    campaign = make_campaign(tmp_path)
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    manifest, declarations, guard, _, _ = load_trial(campaign, SHORT, model, sha(campaign / "campaign.json"))
    assert [d["name"] for d in task_declarations(declarations, SHORT, manifest.tool_prefix)] == [
        "aps_describe_machine"]
    events = []
    trial = Trial(SHORT, model, guard, SimpleNamespace(emit=events.append), manifest.tool_prefix)
    # start_recipe is in the manifest, so OMP would offer it if registered, but X01 does not allow it.
    frame = {"type": "message_end", "messageId": "m1", "message": {
        "role": "assistant", "provider": "anthropic", "model": "claude-opus-5-5", "stopReason": "toolUse",
        "usage": {"cost": {"total": 0.01}},
        "content": [{"type": "toolCall", "id": "c1", "name": "aps_start_recipe", "arguments": {"id": "a"}}]}}
    with pytest.raises(TrialFailure, match="assistant_batch_denied"):
        asyncio.run(trial.handle(frame))
    denied = [event for event in events if event["type"] == "denied"]
    assert denied == [{"type": "denied", "category": "assistant_batch_denied", "name": "aps_start_recipe",
                       "call_id": "c1"}]
    assert trial.denied == [{"category": "assistant_batch_denied", "name": "aps_start_recipe", "call_id": "c1"}]


def run_child_until_result(tmp_path, monkeypatch, *, profile="ADS", endpoint=ENDPOINT):
    """child_trial with a fake bridge; returns (journal records, stdio opened?, OMP spawned?)."""
    import contextlib

    import mcp
    import mcp.client.stdio

    import campaign_worker
    opened, spawned = [], []

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        opened.append(params)
        yield None, None

    async def fake_discover(session, declarations, tools):
        return None

    async def fake_spawn(*args, **kwargs):
        spawned.append(args)
        raise campaign_worker.TrialFailure("spawn_captured")

    class Session(FakeClientSession):
        pass
    Session.profile = profile
    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(campaign_worker, "discover", fake_discover)
    monkeypatch.setattr(campaign_worker.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    write_bridge_configs(tmp_path / "local" / "llm-bench-ads", endpoint)
    campaign = make_campaign(tmp_path)
    journal = tmp_path / "trial.jsonl"
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}

    async def scenario():
        trial = asyncio.create_task(campaign_worker.child_trial(
            campaign, CAPSTONE, model, sha(campaign / "campaign.json"), journal))
        for _ in range(500):
            if journal.exists() and b"worker_result" in journal.read_bytes():
                break
            await asyncio.sleep(0.01)
        trial.cancel()
        await asyncio.gather(trial, return_exceptions=True)

    asyncio.run(scenario())
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    return records, opened, spawned


# Final review: the trial's own bridge session is gated, not only the stack's Read probe.
def test_child_trial_refuses_a_non_emulator_bridge_before_any_tool_is_exposed(tmp_path, monkeypatch):
    records, opened, spawned = run_child_until_result(tmp_path, monkeypatch, profile="ADS_MACHINE")
    assert opened and spawned == []
    assert records[-1]["type"] == "worker_result" and records[-1]["error_category"] == "bridge_not_emulator"
    assert "bridge_gate_passed" not in [r["type"] for r in records]


def test_child_trial_refuses_a_config_that_left_the_frozen_destination(tmp_path, monkeypatch):
    records, opened, spawned = run_child_until_result(tmp_path, monkeypatch, endpoint={"host": "10.0.0.5", "port": 8026})
    assert opened == [] and spawned == []
    assert records[-1]["error_category"] == "bridge_endpoint_mismatch"


def test_child_trial_gate_passes_on_the_emulator(tmp_path, monkeypatch):
    records, opened, spawned = run_child_until_result(tmp_path, monkeypatch)
    gates = [(r["key"], r["profile"]) for r in records if r["type"] == "bridge_gate_passed"]
    assert spawned and gates == [("recipe", "ADS")]


def test_bridge_endpoints_must_be_frozen_and_local(tmp_path):
    from manifest import load_manifest
    with pytest.raises(ValueError, match="bridge_endpoints"):
        load_manifest(make_campaign(tmp_path / "a", bridge_endpoints={k: {"host": "10.0.0.5", "port": 8026}
                                                                     for k in ("read", "recipe", "manual")}))
    with pytest.raises(ValueError, match="bridge_endpoints"):
        load_manifest(make_campaign(tmp_path / "b", bridge_endpoints={"read": dict(ENDPOINT)}))
