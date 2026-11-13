"""ADS emulator stack controller: reset, readiness, gate, recipe library and trial artifacts."""
import asyncio
import contextlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ads_stack
from ads_stack import (AGENT_SERVER_KEY, CONTROL_IMAGE, ROOT_CONFIG_KEY, EMULATOR_IMAGE, StackController, StackError, StackPaths,
                       check_gate, finish_trial, freeze_baseline, load_baseline, needs_reset, profile_hashes,
                       start_stack, stop_stack, wait_ready)

TASK = {"id": "A01", "type": "short", "key": "read", "ask": "Describe the machine.", "tools": ["describe_machine"]}
IDLE = {"chambers": [{"id": "Main", "state": "Idle"}, {"id": "LL", "state": "Idle"}],
        "activeAlarms": [], "isRecipeRunning": False, "isDepositionActive": False, "controlMode": "Remote"}
MACHINE = {"profile": "ADS", "appVersion": "test", "controlMode": "Remote"}


def write_root_config(path, config_dir, prefix):
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<AppConfig>
    <ConfigDir>{config_dir}</ConfigDir>
	<Prefix>{prefix}</Prefix>
	<HWDir>HW_LIB</HWDir>
</AppConfig>
""", encoding="utf-8")


def make_paths(tmp_path):
    profile = tmp_path / "Configs" / "ADS"
    (profile / "Snapshots").mkdir(parents=True)
    (profile / "Recipes" / "Tests").mkdir(parents=True)
    (profile / "AuditLog").mkdir()
    (profile / "Sources.xml").write_text("<Sources/>", encoding="utf-8")
    (profile / "Snapshots" / "All_Pumped.emstate").write_text("{}", encoding="utf-8")
    (profile / "Recipes" / "Cleaning.rcpx").write_text("<Recipe>clean</Recipe>", encoding="utf-8")
    (profile / "Recipes" / "Tests" / "Deposition_test.rcpx").write_text("<Recipe>test</Recipe>", encoding="utf-8")
    (profile / "AuditLog" / "2026-10-09.jsonl").write_text("{}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    (bin_dir / "Data").mkdir(parents=True)
    (bin_dir / "Logs").mkdir()
    (bin_dir / "AgentServer.xml").write_text('<AgentServer enabled="true" port="8026"/>', encoding="utf-8")
    write_root_config(bin_dir / "RootConfig.xml", profile.parent, "ADS")
    return StackPaths(emulator=bin_dir / EMULATOR_IMAGE, control=bin_dir / CONTROL_IMAGE, profile_dir=profile,
                      agent_server_xml=bin_dir / "AgentServer.xml", root_config=bin_dir / "RootConfig.xml",
                      snapshot=profile / "Snapshots" / "All_Pumped.emstate",
                      data_dir=bin_dir / "Data", logs_dir=bin_dir / "Logs")


class FakeRunner:
    """PID table of the fake machine; starting the emulator restores the snapshot (Idle, no recipe)."""

    def __init__(self, running=(), machine=None):
        self.procs, self.next_pid, self.log = {}, 1000, []
        self.created, self.clock = {}, 0
        self.machine = machine
        for image in running:
            self.spawn(image)

    def spawn(self, image):
        self.next_pid += 4
        self.clock += 1
        self.procs[self.next_pid] = image
        self.created[self.next_pid] = f"t{self.clock}"
        return self.next_pid

    @property
    def images(self):
        return set(self.procs.values())

    def processes(self):
        return {pid: (image, self.created.get(pid)) for pid, image in self.procs.items()}

    def kill(self, pid):
        self.log.append(("kill", self.procs.get(pid), pid))
        self.procs.pop(pid, None)

    def start(self, argv, env, cwd):
        self.log.append(("start", list(argv), dict(env), Path(cwd)))
        pid = self.spawn(Path(argv[0]).name)
        if self.machine is not None and Path(argv[0]).name == EMULATOR_IMAGE:
            self.machine.restore()
        return pid


def kills(runner):
    return [entry[1] for entry in runner.log if entry[0] == "kill"]


class FakeMachine:
    """Bridge view of the emulator: read tools plus abort_recipe."""

    def __init__(self, profile="ADS", running=False, abort_polls=2):
        self.profile = profile
        self.running = running
        self.abort_polls = abort_polls
        self.aborting = None
        self.calls = []

    def restore(self):
        self.running, self.aborting = False, None

    def status(self):
        if self.aborting is not None:
            self.aborting -= 1
            if self.aborting <= 0:
                self.running, self.aborting = False, None
        return {**IDLE, "isRecipeRunning": self.running,
                "chambers": [{"id": "Main", "state": "Processing" if self.running else "Idle"}]}

    def call(self, name, arguments=None):
        self.calls.append(name)
        if name == "describe_machine":
            return {**MACHINE, "profile": self.profile}
        if name == "system_status":
            return self.status()
        if name == "recipe_status":
            return {"isRecipeRunning": self.status()["isRecipeRunning"], "failureProtocolActive": self.aborting is not None}
        if name == "abort_recipe":
            self.aborting = self.abort_polls
            return {"accepted": True}
        raise AssertionError("unexpected bridge call " + name)

    def session_factory(self):
        machine = self

        class Session:
            async def call(self, name, arguments=None):
                return machine.call(name, arguments)

        @contextlib.asynccontextmanager
        async def factory():
            yield Session()
        return factory


async def no_sleep(_seconds):
    return None


def controller(paths, runner, machine, campaign, port_open=None):
    async def always_open():
        return True
    expected, baseline = load_baseline_for(paths, campaign)
    return StackController(paths=paths, runner=runner, read_session=machine.session_factory(),
                           recipe_session=machine.session_factory(), expected_hashes=expected,
                           recipes_baseline=baseline, env={"APS_DEV_AUTOLOGIN": "x"},
                           port_open=port_open or always_open, sleep=no_sleep, sync_sleep=lambda _s: None)


def fingerprint_of(campaign):
    import hashlib
    return hashlib.sha256((campaign / "campaign.json").read_bytes()).hexdigest()


def load_baseline_for(paths, campaign):
    freeze_baseline(paths, campaign)
    return json.loads((campaign / ads_stack.STACK_BASELINE).read_text(encoding="utf-8")), campaign / ads_stack.RECIPES_BASELINE


def journal(tmp_path, names, status="completed"):
    path = tmp_path / "previous.jsonl"
    rows = [{"type": "trial_start"}] + [{"type": "tool_call", "name": name, "wire_name": "aps_" + name} for name in names]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"status": status, "journal_file": str(path)}


# Review Focus 1 (controller ruling: a stack the harness did not start is never touched)
@pytest.mark.parametrize("running", [{CONTROL_IMAGE, EMULATOR_IMAGE}, {CONTROL_IMAGE}, {EMULATOR_IMAGE}])
def test_foreign_stack_refused(tmp_path, running):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(running=running | {"com.aps.control.exe"}, machine=machine)
    before = runner.processes()
    stack = controller(paths, runner, machine, tmp_path / "campaign")

    with pytest.raises(StackError, match="foreign_stack_running"):
        asyncio.run(stack.prepare(None, TASK))

    assert runner.log == [] and runner.processes() == before
    assert machine.calls == []
    assert stack.pids == {}


def test_fresh_start_owns_started_pids(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(running={"com.aps.control.exe"}, machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")

    record = asyncio.run(stack.prepare(None, TASK))

    starts = [entry for entry in runner.log if entry[0] == "start"]
    assert kills(runner) == []
    assert starts[0][1] == [str(paths.emulator), "--project", "ADS", "--state", str(paths.snapshot)]
    assert starts[1][1] == [str(paths.control)]
    assert starts[1][2]["APS_DEV_CONTROL_MODE"] == "Remote"
    assert starts[1][2]["APS_DEV_AUTOLOGIN"] == "x"
    assert stack.pids == {pid: ident for pid, ident in runner.processes().items() if ident[0] in ads_stack.STACK_IMAGES}
    assert record == {"type": "stack_verified", "profile": "ADS", "reset": True,
                      "profile_files": len(profile_hashes(paths))}


def test_reset_kills_only_owned_pids(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    owned = dict(stack.pids)
    control_pid = next(pid for pid, ident in owned.items() if ident[0] == CONTROL_IMAGE)
    emulator_pid = next(pid for pid, ident in owned.items() if ident[0] == EMULATOR_IMAGE)

    runner.procs[control_pid] = "notepad.exe"  # owned Control exited, its PID was reused
    record = asyncio.run(stack.prepare(journal(tmp_path, ["device_command"]), TASK))

    assert record["reset"] is True
    assert [(entry[1], entry[2]) for entry in runner.log if entry[0] == "kill"] == [(EMULATOR_IMAGE, emulator_pid)]
    assert runner.procs[control_pid] == "notepad.exe"
    assert {ident[0] for ident in stack.pids.values()} == {CONTROL_IMAGE, EMULATOR_IMAGE}
    assert not set(stack.pids) & set(owned)


def test_foreign_process_beside_owned_stack_refused(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    emulator_pid = next(pid for pid, ident in stack.pids.items() if ident[0] == EMULATOR_IMAGE)
    del runner.procs[emulator_pid]
    runner.spawn(EMULATOR_IMAGE)  # someone else started an emulator
    runner.log.clear()

    with pytest.raises(StackError, match="foreign_stack_running"):
        asyncio.run(stack.prepare(journal(tmp_path, ["describe_machine"]), TASK))
    assert runner.log == []


def test_stop_stack_kills_owned_pids_only_and_times_out():
    class Stuck(FakeRunner):
        def kill(self, pid):
            self.log.append(("kill", self.procs.get(pid), pid))

    runner = Stuck(running={"com.aps.control.exe", CONTROL_IMAGE})
    foreign = next(pid for pid, image in runner.procs.items() if image == CONTROL_IMAGE)
    owned_pid = runner.spawn(CONTROL_IMAGE)
    clock = iter(range(0, 1000))
    with pytest.raises(StackError, match="stack_stop_failed"):
        stop_stack(runner, {owned_pid: runner.processes()[owned_pid]}, sleep=lambda _s: None, clock=lambda: next(clock))
    assert [entry[2] for entry in runner.log] == [owned_pid]
    assert foreign in runner.procs


def test_start_stack_waits_between_emulator_and_control(tmp_path):
    paths = make_paths(tmp_path)
    runner, slept = FakeRunner(), []
    pids = start_stack(paths, runner, {"PATH": "p"}, sleep=slept.append)
    assert [entry[1][0] for entry in runner.log] == [str(paths.emulator), str(paths.control)]
    assert slept == [5]
    assert "APS_DEV_CONTROL_MODE" not in runner.log[0][2]
    assert pids == runner.processes() and all(created for _image, created in pids.values())


def test_start_stack_stops_emulator_when_control_fails(tmp_path):
    class Runner(FakeRunner):
        def start(self, argv, env, cwd):
            if Path(argv[0]).name == CONTROL_IMAGE:
                raise OSError("Control did not start")
            return super().start(argv, env, cwd)

    paths = make_paths(tmp_path)
    runner = Runner()
    with pytest.raises(OSError):
        start_stack(paths, runner, {}, sleep=lambda _s: None)
    assert kills(runner) == [EMULATOR_IMAGE] and runner.procs == {}


# Review Focus 2
def test_port_never_opens():
    probed = []

    async def probe():
        probed.append(1)
        return {}

    async def closed():
        return False

    ticks = iter(range(0, 10000))
    with pytest.raises(StackError, match="stack_not_ready"):
        asyncio.run(wait_ready(probe, deadline_s=90, port_open=closed, sleep=no_sleep, clock=lambda: next(ticks)))
    assert probed == []


def test_stack_not_ready_fails_trial_without_model_call(tmp_path, monkeypatch):
    import campaign_worker
    from test_manifest import SHORT, make_campaign

    spawned = []

    async def no_spawn(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("no worker or model process may start")

    async def prepare():
        raise StackError("stack_not_ready")

    monkeypatch.setattr(campaign_worker.asyncio, "create_subprocess_exec", no_spawn)
    campaign = make_campaign(tmp_path)
    (campaign / "results").mkdir()
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    costs = []
    result = asyncio.run(campaign_worker.run_trial(campaign=campaign, task=SHORT,
                                                   model=model, fingerprint=fingerprint_of(campaign),
                                                   on_cost=lambda key, cost: costs.append(cost), stack=prepare))
    assert result["status"] == "failed"
    assert result["error_category"] == "stack_not_ready"
    assert result["usage_records"] == [] and costs == [] and spawned == []
    records = [json.loads(line) for line in Path(result["journal_file"]).read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records] == ["trial_start", "result"]


def test_stack_verified_journaled_before_worker(tmp_path, monkeypatch):
    import campaign_worker
    from test_manifest import SHORT, make_campaign

    order = []

    async def no_spawn(*args, **kwargs):
        order.append([json.loads(line)["type"] for line in journal_path.read_text(encoding="utf-8").splitlines()])
        raise RuntimeError("stop here")

    async def prepare():
        return {"type": "stack_verified", "profile": "ADS", "reset": True, "profile_files": 3}

    monkeypatch.setattr(campaign_worker.asyncio, "create_subprocess_exec", no_spawn)
    campaign = make_campaign(tmp_path)
    (campaign / "results").mkdir()
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    journal_path = campaign / "results" / "X01_anthropic__claude-opus-5-5.jsonl"
    asyncio.run(campaign_worker.run_trial(campaign=campaign, task=SHORT,
                                          model=model, fingerprint=fingerprint_of(campaign), on_cost=lambda *_: None, stack=prepare))
    assert order == [["trial_start", "stack_verified"]]
    stack_record = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[1])
    assert {k: v for k, v in stack_record.items() if k != "at"} == {"type": "stack_verified", "profile": "ADS",
                                                                     "reset": True, "profile_files": 3}


# Review Focus 3
def test_profile_hash_change_refused(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))

    (paths.profile_dir / "AuditLog" / "2026-10-10.jsonl").write_text("{}\n", encoding="utf-8")
    asyncio.run(stack.prepare(None, TASK))  # the audit log is excluded

    (paths.profile_dir / "Sources.xml").write_text("<Sources edited='1'/>", encoding="utf-8")
    with pytest.raises(StackError, match="profile_hash_mismatch"):
        asyncio.run(stack.prepare(None, TASK))


def test_profile_hashes_scope(tmp_path):
    paths = make_paths(tmp_path)
    hashes = profile_hashes(paths)
    assert set(hashes) == {"Sources.xml", "Snapshots/All_Pumped.emstate", "Recipes/Cleaning.rcpx",
                           "Recipes/Tests/Deposition_test.rcpx", AGENT_SERVER_KEY, ROOT_CONFIG_KEY}
    paths.agent_server_xml.write_text('<AgentServer enabled="false"/>', encoding="utf-8")
    assert profile_hashes(paths)[AGENT_SERVER_KEY] != hashes[AGENT_SERVER_KEY]
    write_root_config(paths.root_config, paths.profile_dir.parent, "ADS_MACHINE")
    assert profile_hashes(paths)[ROOT_CONFIG_KEY] != hashes[ROOT_CONFIG_KEY]


def test_machine_profile_refused(tmp_path):
    paths = make_paths(tmp_path)
    hashes = profile_hashes(paths)
    with pytest.raises(StackError, match="profile_not_emulator"):
        check_gate({**MACHINE, "profile": "ADS_MACHINE"}, IDLE, hashes, paths)


def test_gate_requires_idle(tmp_path):
    paths = make_paths(tmp_path)
    hashes = profile_hashes(paths)
    check_gate(MACHINE, IDLE, hashes, paths)
    for status in ({**IDLE, "isRecipeRunning": True}, {**IDLE, "activeAlarms": [{"id": "A1"}]},
                   {**IDLE, "chambers": [{"id": "Main", "state": "Processing"}]}, {**IDLE, "chambers": []}):
        with pytest.raises(StackError, match="stack_not_idle"):
            check_gate(MACHINE, status, hashes, paths)


# Review Focus 4
def test_running_recipe_aborted_then_reset(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine(abort_polls=3)
    runner = FakeRunner(machine=machine)
    campaign = tmp_path / "campaign"
    stack = controller(paths, runner, machine, campaign)
    asyncio.run(stack.prepare(None, TASK))
    owned = dict(stack.pids)
    machine.running = True  # the agent left a recipe running
    started = datetime.now(timezone.utc) - timedelta(seconds=5)

    asyncio.run(stack.finish(tmp_path / "dest", started))

    assert machine.calls.index("abort_recipe") < len(machine.calls) - 1
    assert machine.running is False
    previous = journal(tmp_path, ["describe_machine", "start_recipe"], status="timeout")
    assert needs_reset(previous, TASK)
    machine.running = True  # even a leftover the abort missed is cleared by the reset
    record = asyncio.run(stack.prepare(previous, TASK))
    assert record["reset"] is True
    assert sorted(entry[2] for entry in runner.log if entry[0] == "kill") == sorted(owned)
    assert machine.running is False
    assert machine.calls[-2:] == ["describe_machine", "system_status"]


def test_abort_that_never_reaches_idle_raises(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine(running=True, abort_polls=10 ** 9)
    ticks = iter(range(0, 100000, 5))
    with pytest.raises(StackError, match="abort_not_idle"):
        asyncio.run(finish_trial(machine.session_factory(), tmp_path / "dest", datetime.now(timezone.utc),
                                 status_session=machine.session_factory(), paths=paths, recipes_baseline=tmp_path / "none", sleep=no_sleep,
                                 clock=lambda: next(ticks)))


def test_idle_trial_is_not_aborted(tmp_path):
    paths = make_paths(tmp_path)
    campaign = tmp_path / "campaign"
    freeze_baseline(paths, campaign)
    machine = FakeMachine()
    asyncio.run(finish_trial(machine.session_factory(), tmp_path / "dest", datetime.now(timezone.utc),
                             status_session=machine.session_factory(), paths=paths, recipes_baseline=campaign / ads_stack.RECIPES_BASELINE, sleep=no_sleep))
    assert "abort_recipe" not in machine.calls


def test_artifacts_copied_after_start(tmp_path):
    paths = make_paths(tmp_path)
    campaign = tmp_path / "campaign"
    freeze_baseline(paths, campaign)
    started = datetime.now(timezone.utc)
    old = (started - timedelta(hours=1)).timestamp()
    for name in ("old.tsv", "old.run.json"):
        (paths.data_dir / name).write_text("old", encoding="utf-8")
        os.utime(paths.data_dir / name, (old, old))
    (paths.logs_dir / "Process_old.log").write_text("old", encoding="utf-8")
    os.utime(paths.logs_dir / "Process_old.log", (old, old))
    (paths.data_dir / "run1.tsv").write_text("t\tp\n", encoding="utf-8")
    (paths.data_dir / "run1.run.json").write_text("{}", encoding="utf-8")
    (paths.data_dir / "run1.depo.json").write_text("{}", encoding="utf-8")
    (paths.logs_dir / "Process_2026-10-09.log").write_text("log", encoding="utf-8")
    (paths.logs_dir / "Other.log").write_text("log", encoding="utf-8")
    dest = tmp_path / "dest"

    machine = FakeMachine()
    copied = asyncio.run(finish_trial(machine.session_factory(), dest, started,
                                      status_session=machine.session_factory(), paths=paths,
                                      recipes_baseline=campaign / ads_stack.RECIPES_BASELINE, sleep=no_sleep))

    assert sorted(p.relative_to(dest).as_posix() for p in copied) == [
        "Data/run1.run.json", "Data/run1.tsv", "Logs/Process_2026-10-09.log", "final_status.json"]
    assert all(p.is_file() for p in copied)


def test_uploaded_recipe_moved_and_overwritten_recipe_restored(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    campaign = tmp_path / "campaign"
    stack = controller(paths, runner, machine, campaign)
    asyncio.run(stack.prepare(None, TASK))
    started = datetime.now(timezone.utc)

    recipes = paths.profile_dir / "Recipes"
    (recipes / "RuC_ML.rcpx").write_text("<Recipe>agent</Recipe>", encoding="utf-8")
    (recipes / "Cleaning.rcpx").write_text("<Recipe>overwritten</Recipe>", encoding="utf-8")
    (recipes / "Tests" / "Deposition_test.rcpx").unlink()
    with pytest.raises(StackError, match="profile_hash_mismatch"):
        asyncio.run(stack.prepare(None, TASK))

    dest = tmp_path / "dest"
    copied = asyncio.run(stack.finish(dest, started))

    assert (dest / "Recipes" / "RuC_ML.rcpx").read_text(encoding="utf-8") == "<Recipe>agent</Recipe>"
    assert (dest / "Recipes" / "Cleaning.rcpx").read_text(encoding="utf-8") == "<Recipe>overwritten</Recipe>"
    assert dest / "Recipes" / "RuC_ML.rcpx" in copied and dest / "Recipes" / "Cleaning.rcpx" in copied
    assert not (recipes / "RuC_ML.rcpx").exists()
    assert (recipes / "Cleaning.rcpx").read_text(encoding="utf-8") == "<Recipe>clean</Recipe>"
    assert (recipes / "Tests" / "Deposition_test.rcpx").read_text(encoding="utf-8") == "<Recipe>test</Recipe>"
    asyncio.run(stack.prepare(None, TASK))  # the gate passes again


def test_load_baseline_requires_frozen_input(tmp_path):
    paths = make_paths(tmp_path)
    campaign = tmp_path / "campaign"
    hashes = freeze_baseline(paths, campaign)
    import hashlib
    frozen = {ads_stack.STACK_BASELINE: hashlib.sha256((campaign / ads_stack.STACK_BASELINE).read_bytes()).hexdigest()}
    assert load_baseline(campaign, frozen) == (hashes, campaign / ads_stack.RECIPES_BASELINE)
    with pytest.raises(ValueError, match="frozen"):
        load_baseline(campaign, {})
    (campaign / ads_stack.RECIPES_BASELINE / "Cleaning.rcpx").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe baseline"):
        load_baseline(campaign, frozen)


# Review Focus 5
def test_first_probe_failure_retried_once():
    calls = []

    async def open_port():
        return True

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("first call after restart")
        return {"machine": MACHINE, "status": IDLE}

    assert asyncio.run(wait_ready(flaky, port_open=open_port, sleep=no_sleep)) == {"machine": MACHINE, "status": IDLE}
    assert len(calls) == 2

    calls.clear()

    async def broken():
        calls.append(1)
        raise ConnectionError("down")

    with pytest.raises(StackError, match="stack_not_ready"):
        asyncio.run(wait_ready(broken, port_open=open_port, sleep=no_sleep))
    assert len(calls) == 2


def test_probe_runs_before_every_trial_even_without_reset(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    runner.log.clear()
    machine.calls.clear()
    record = asyncio.run(stack.prepare(journal(tmp_path, ["describe_machine", "system_status"]), TASK))
    assert record["reset"] is False and runner.log == []
    assert machine.calls == ["describe_machine", "system_status"]


def test_missing_process_forces_reset(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    control_pid = next(pid for pid, ident in stack.pids.items() if ident[0] == CONTROL_IMAGE)
    del runner.procs[control_pid]
    record = asyncio.run(stack.prepare(journal(tmp_path, ["describe_machine"]), TASK))
    assert record["reset"] is True


def test_needs_reset_rules(tmp_path):
    assert needs_reset(None, TASK)
    assert not needs_reset(journal(tmp_path, ["describe_machine", "system_status", "recipe_status"]), TASK)
    assert needs_reset(journal(tmp_path, ["describe_machine", "device_command"]), TASK)
    for name in ("upload_recipe", "start_recipe", "abort_recipe", "resolve_pause"):
        assert needs_reset(journal(tmp_path, [name]), TASK)
    assert needs_reset(journal(tmp_path, ["describe_machine"], status="timeout"), TASK)
    assert needs_reset(journal(tmp_path, ["describe_machine"], status="failed"), TASK)
    assert needs_reset(journal(tmp_path, ["describe_machine"]), {**TASK, "fresh_state": True})
    assert needs_reset({"status": "completed", "journal_file": str(tmp_path / "missing.jsonl")}, TASK)


def test_launch_prepares_and_finishes_each_trial(tmp_path, monkeypatch):
    import run_campaign
    from test_manifest import make_campaign

    campaign = make_campaign(tmp_path, parallelism=1)
    (campaign / "results").mkdir()
    events = []

    class FakeStack:
        async def prepare(self, previous, task):
            events.append(("prepare", task["id"], previous and previous["trial_id"]))
            return {"type": "stack_verified", "profile": "ADS", "reset": previous is None, "profile_files": 1}

        async def finish(self, dest, started_utc):
            events.append(("finish", dest.name, started_utc.tzinfo is not None))
            return []

        async def close(self):
            events.append(("close",))

    async def fake_run_trial(*, campaign, task, model, fingerprint, on_cost, stack):
        record = await stack()
        events.append(("trial", task["id"], record["type"]))
        identifier = task["id"] + "_" + model["selector"].replace("/", "__")
        return {"trial_id": identifier, "fingerprint": fingerprint, "task": task["id"], "selector": model["selector"],
                "status": "completed", "error_category": None, "duration_s": 0, "catalog_estimated_cost_usd": 0}

    monkeypatch.setattr(run_campaign, "run_trial", fake_run_trial)
    assert asyncio.run(run_campaign.launch(campaign, stack=FakeStack())) == 0
    first = "X01_anthropic__claude-opus-5-5"
    assert events == [("prepare", "X01", None), ("trial", "X01", "stack_verified"), ("finish", first, True),
                      ("prepare", "XC1", first), ("trial", "XC1", "stack_verified"),
                      ("finish", "XC1_anthropic__claude-opus-5-5", True), ("close",)]


def launch_with_failing_prepare(tmp_path, monkeypatch, code, close_error=None):
    """Trial X01 fails its stack preparation with `code` (None: it passes); returns (launch outcome, events)."""
    import run_campaign
    from test_manifest import make_campaign

    campaign = make_campaign(tmp_path, parallelism=1)
    (campaign / "results").mkdir()
    events = []

    class FakeStack:
        async def prepare(self, previous, task):
            events.append(("prepare", task["id"], previous and previous["trial_id"]))
            if task["id"] == "X01" and code is not None:
                raise StackError(code)
            return {"type": "stack_verified", "profile": "ADS", "reset": True, "profile_files": 1}

        async def finish(self, dest, started_utc):
            events.append(("finish", dest.name))
            return []

        async def close(self):
            events.append(("close",))
            if close_error:
                raise StackError(close_error)

    async def fake_run_trial(*, campaign, task, model, fingerprint, on_cost, stack):
        identifier = task["id"] + "_" + model["selector"].replace("/", "__")
        status, category = "completed", None
        try:
            await stack()
        except StackError as exc:
            status, category = "failed", exc.code
        return {"trial_id": identifier, "fingerprint": fingerprint, "task": task["id"], "selector": model["selector"],
                "status": status, "error_category": category, "duration_s": 0, "catalog_estimated_cost_usd": 0}

    monkeypatch.setattr(run_campaign, "run_trial", fake_run_trial)
    try:
        outcome = asyncio.run(run_campaign.launch(campaign, stack=FakeStack()))
    except StackError as exc:
        outcome = exc
    return outcome, events


def test_launch_skips_bridge_cleanup_when_gate_failed(tmp_path, monkeypatch):
    outcome, events = launch_with_failing_prepare(tmp_path, monkeypatch, "stack_not_ready")
    assert outcome == 0
    assert events == [("prepare", "X01", None), ("prepare", "XC1", None),
                      ("finish", "XC1_anthropic__claude-opus-5-5"), ("close",)]


@pytest.mark.parametrize("code", ["profile_not_emulator", "foreign_stack_running", "stack_stop_failed"])
def test_launch_stops_campaign_on_fatal_stack_error(tmp_path, monkeypatch, code):
    outcome, events = launch_with_failing_prepare(tmp_path, monkeypatch, code)
    assert isinstance(outcome, StackError) and outcome.code == code
    assert events == [("prepare", "X01", None), ("close",)]
    progress = json.loads((tmp_path / "campaign" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "interrupted_or_error" and progress["attempted_trials"] == 1
    assert progress["stop_reason"] == code
    assert not (tmp_path / "campaign" / "campaign.lock").exists()


def test_close_stop_failure_fails_campaign(tmp_path, monkeypatch):
    outcome, events = launch_with_failing_prepare(tmp_path, monkeypatch, None, close_error="stack_stop_failed")
    assert isinstance(outcome, StackError) and outcome.code == "stack_stop_failed"
    assert events[-1] == ("close",) and events.count(("close",)) == 1
    campaign = tmp_path / "campaign"
    progress = json.loads((campaign / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "interrupted_or_error" and progress["stop_reason"] == "stack_stop_failed"
    assert progress["attempted_trials"] == 2
    assert not (campaign / "collection-summary.json").exists()
    assert not (campaign / "campaign.lock").exists()


def test_close_failure_after_fatal_error_is_recorded(tmp_path, monkeypatch):
    outcome, events = launch_with_failing_prepare(tmp_path, monkeypatch, "foreign_stack_running",
                                                  close_error="stack_stop_failed")
    assert isinstance(outcome, StackError) and outcome.code == "foreign_stack_running"
    progress = json.loads((tmp_path / "campaign" / "progress.json").read_text(encoding="utf-8"))
    assert progress["stop_reason"] == "foreign_stack_running"
    assert progress["stack_close_error"] == "stack_stop_failed"
    assert not (tmp_path / "campaign" / "campaign.lock").exists()


@pytest.mark.parametrize("failure", ["port", "hash", "idle", "start"])
def test_failed_preparation_stops_owned_stack(tmp_path, failure):
    paths = make_paths(tmp_path)
    machine = FakeMachine()

    class Runner(FakeRunner):
        def start(self, argv, env, cwd):
            if failure == "start" and Path(argv[0]).name == CONTROL_IMAGE:
                raise OSError("Control did not start")
            return super().start(argv, env, cwd)

    async def closed():
        return False

    runner = Runner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign", port_open=closed if failure == "port" else None)
    if failure == "hash":
        (paths.profile_dir / "Sources.xml").write_text("<Sources edited='1'/>", encoding="utf-8")
    if failure == "idle":
        machine.restore = lambda: None
        machine.running = True
    stack.clock = iter(range(0, 10 ** 6, 10)).__next__

    expected = {"port": "stack_not_ready", "hash": "profile_hash_mismatch", "idle": "stack_not_idle",
                "start": "stack_start_failed"}[failure]
    with pytest.raises(StackError, match=expected):
        asyncio.run(stack.prepare(None, TASK))

    assert runner.images == set()
    assert kills(runner) == ([EMULATOR_IMAGE] if failure == "start" else [CONTROL_IMAGE, EMULATOR_IMAGE])
    assert stack.pids == {}


def test_stop_failure_during_cleanup_is_reported(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()

    class Stuck(FakeRunner):
        def kill(self, pid):
            self.log.append(("kill", self.procs.get(pid), pid))

    async def closed():
        return False

    runner = Stuck(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign", port_open=closed)
    stack.clock = iter(range(0, 10 ** 6, 10)).__next__
    with pytest.raises(StackError, match="stack_stop_failed"):
        asyncio.run(stack.prepare(None, TASK))
    assert stack.pids == {}


def test_close_stops_only_owned_stack(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(running={"com.aps.control.exe"}, machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    asyncio.run(stack.close())
    assert runner.images == {"com.aps.control.exe"} and stack.pids == {}
    asyncio.run(stack.close())  # nothing owned: nothing touched
    assert len(kills(runner)) == 2


def test_machine_profile_leaves_processes_alone(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine(profile="ADS_MACHINE")
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    with pytest.raises(StackError, match="profile_not_emulator"):
        asyncio.run(stack.prepare(None, TASK))
    assert runner.log[-1][0] == "start"
    assert runner.images == {CONTROL_IMAGE, EMULATOR_IMAGE}
    assert stack.pids == {}
    assert machine.calls == ["describe_machine", "system_status"]


def frozen_recipes(tmp_path):
    paths = make_paths(tmp_path)
    campaign = tmp_path / "campaign"
    freeze_baseline(paths, campaign)
    return paths, campaign / ads_stack.RECIPES_BASELINE


def test_junctioned_recipe_folder_is_not_written_through(tmp_path):
    import _winapi
    paths, baseline = frozen_recipes(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Deposition_test.rcpx").write_text("precious", encoding="utf-8")
    tests_dir = paths.recipes_dir / "Tests"
    (tests_dir / "Deposition_test.rcpx").unlink()
    tests_dir.rmdir()
    _winapi.CreateJunction(str(outside), str(tests_dir))

    with pytest.raises(StackError, match="recipe_path_unsafe"):
        ads_stack.restore_recipes(paths.recipes_dir, baseline, tmp_path / "dest")
    assert (outside / "Deposition_test.rcpx").read_text(encoding="utf-8") == "precious"


def test_symlinked_recipe_is_not_overwritten_or_deleted(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("precious", encoding="utf-8")
    (paths.recipes_dir / "Cleaning.rcpx").unlink()
    os.symlink(outside, paths.recipes_dir / "Cleaning.rcpx")
    os.symlink(outside, paths.recipes_dir / "Extra.rcpx")
    (paths.recipes_dir / "Agent.rcpx").write_text("agent", encoding="utf-8")

    with pytest.raises(StackError, match="recipe_path_unsafe"):
        ads_stack.restore_recipes(paths.recipes_dir, baseline, tmp_path / "dest")
    assert outside.read_text(encoding="utf-8") == "precious"
    assert (paths.recipes_dir / "Agent.rcpx").exists()  # validation precedes every change
    assert not (tmp_path / "dest").exists()


def test_symlinked_baseline_recipe_rejected(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("<Recipe>clean</Recipe>", encoding="utf-8")
    (baseline / "Cleaning.rcpx").unlink()
    os.symlink(outside, baseline / "Cleaning.rcpx")
    with pytest.raises(StackError, match="recipe_path_unsafe"):
        ads_stack.restore_recipes(paths.recipes_dir, baseline, tmp_path / "dest")


def test_finish_still_restores_after_failed_session(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    (paths.recipes_dir / "Agent.rcpx").write_text("agent", encoding="utf-8")

    @contextlib.asynccontextmanager
    async def broken():
        raise ConnectionError("bridge down")
        yield

    with pytest.raises(StackError, match="finish_session_failed"):
        asyncio.run(finish_trial(broken, tmp_path / "dest", datetime.now(timezone.utc), status_session=FakeMachine().session_factory(), paths=paths,
                                 recipes_baseline=baseline, sleep=no_sleep))
    assert not (paths.recipes_dir / "Agent.rcpx").exists()
    assert (tmp_path / "dest" / "Recipes" / "Agent.rcpx").read_text(encoding="utf-8") == "agent"


def test_final_status_snapshot_is_taken_before_the_abort(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    machine = FakeMachine(running=True, abort_polls=2)
    reads = []

    @contextlib.asynccontextmanager
    async def read_session():
        class Session:
            async def call(self, name, arguments=None):
                reads.append(name)
                return machine.call(name, arguments)
        yield Session()

    dest = tmp_path / "dest"
    asyncio.run(finish_trial(machine.session_factory(), dest, datetime.now(timezone.utc), status_session=read_session,
                             paths=paths, recipes_baseline=baseline, sleep=no_sleep))
    assert reads == ["system_status"] and machine.calls[0] == "system_status"
    assert machine.calls.index("system_status") < machine.calls.index("abort_recipe")
    final = json.loads((dest / ads_stack.FINAL_STATUS).read_text(encoding="utf-8"))
    assert final["aborted"] is True
    assert final["system_status"]["isRecipeRunning"] is True  # the state the agent left, not the state after the abort
    assert machine.running is False


def test_idle_trial_snapshot_records_no_abort(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    machine = FakeMachine()
    dest = tmp_path / "dest"
    asyncio.run(finish_trial(machine.session_factory(), dest, datetime.now(timezone.utc),
                             status_session=machine.session_factory(), paths=paths, recipes_baseline=baseline,
                             sleep=no_sleep))
    final = json.loads((dest / ads_stack.FINAL_STATUS).read_text(encoding="utf-8"))
    assert final == {"system_status": machine.status(), "aborted": False}


def test_failed_status_read_writes_no_snapshot_but_still_aborts_and_restores(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    (paths.recipes_dir / "Agent.rcpx").write_text("agent", encoding="utf-8")
    machine = FakeMachine(running=True)

    @contextlib.asynccontextmanager
    async def broken():
        raise ConnectionError("read bridge down")
        yield

    dest = tmp_path / "dest"
    with pytest.raises(StackError, match="finish_status_failed"):
        asyncio.run(finish_trial(machine.session_factory(), dest, datetime.now(timezone.utc), status_session=broken,
                                 paths=paths, recipes_baseline=baseline, sleep=no_sleep))
    assert not (dest / ads_stack.FINAL_STATUS).exists()
    assert "abort_recipe" in machine.calls and machine.running is False
    assert not (paths.recipes_dir / "Agent.rcpx").exists()


def test_snapshot_is_on_disk_before_abort_recipe_is_called(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    dest = tmp_path / "dest"
    machine = FakeMachine(running=True, abort_polls=2)
    seen = {}
    call = machine.call

    def recording_call(name, arguments=None):
        if name == "abort_recipe":
            seen["at_abort"] = json.loads((dest / ads_stack.FINAL_STATUS).read_text(encoding="utf-8"))
        return call(name, arguments)
    machine.call = recording_call
    asyncio.run(finish_trial(machine.session_factory(), dest, datetime.now(timezone.utc),
                             status_session=machine.session_factory(), paths=paths, recipes_baseline=baseline,
                             sleep=no_sleep))
    assert seen["at_abort"]["aborted"] is None and seen["at_abort"]["system_status"]["isRecipeRunning"] is True
    final = json.loads((dest / ads_stack.FINAL_STATUS).read_text(encoding="utf-8"))
    assert final == {"system_status": seen["at_abort"]["system_status"], "aborted": True}


def test_cancelled_abort_leaves_the_snapshot(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    dest = tmp_path / "dest"
    machine = FakeMachine(running=True, abort_polls=10 ** 9)

    async def cancel(_seconds):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(finish_trial(machine.session_factory(), dest, datetime.now(timezone.utc),
                                 status_session=machine.session_factory(), paths=paths, recipes_baseline=baseline,
                                 sleep=cancel))
    final = json.loads((dest / ads_stack.FINAL_STATUS).read_text(encoding="utf-8"))
    assert final["aborted"] is None and final["system_status"]["isRecipeRunning"] is True


# Final review: process identity is PID + image + creation time.
def test_same_image_pid_reuse_is_not_owned_and_never_killed(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")
    asyncio.run(stack.prepare(None, TASK))
    control_pid = next(pid for pid, ident in stack.pids.items() if ident[0] == CONTROL_IMAGE)
    # Owned Control exited and an operator's new Control got the same PID.
    runner.created[control_pid] = "operator-restart"
    runner.log.clear()
    with pytest.raises(StackError, match="foreign_stack_running"):
        asyncio.run(stack.prepare(journal(tmp_path, ["describe_machine"]), TASK))
    assert kills(runner) == []
    stop_stack(runner, {control_pid: (CONTROL_IMAGE, "t-old")}, sleep=lambda _s: None)
    assert kills(runner) == [] and runner.procs[control_pid] == CONTROL_IMAGE


# Final review: Control's RootConfig.xml must select the emulator profile before anything starts.
@pytest.mark.parametrize("config_dir, prefix", [(None, "ADS_MACHINE"), ("/elsewhere/Configs", "ADS")])
def test_root_config_not_emulator_starts_nothing(tmp_path, config_dir, prefix):
    paths = make_paths(tmp_path)
    write_root_config(paths.root_config, config_dir or paths.profile_dir.parent, prefix)
    runner = FakeRunner()
    with pytest.raises(StackError, match="root_config_not_emulator"):
        start_stack(paths, runner, {}, sleep=lambda _s: None)
    assert runner.log == [] and runner.procs == {}
    assert "root_config_not_emulator" in ads_stack.CAMPAIGN_FATAL


def test_root_config_check_reads_the_real_layout(tmp_path):
    paths = make_paths(tmp_path)
    ads_stack.check_root_config(paths)  # the fixture's ConfigDir/Prefix select the emulator profile
    paths.root_config.unlink()
    with pytest.raises(StackError, match="root_config_not_emulator"):
        ads_stack.check_root_config(paths)


# Final review: a probe that never completes is bounded by the readiness deadline.
def test_hung_probe_is_bounded_by_the_deadline():
    async def hung():
        await asyncio.Event().wait()

    async def open_port():
        return True

    async def scenario():
        return await asyncio.wait_for(wait_ready(hung, 0.2, port_open=open_port, sleep=no_sleep), 5)

    with pytest.raises(StackError, match="stack_not_ready"):
        asyncio.run(scenario())


def test_hung_probe_stops_the_owned_stack(tmp_path):
    paths = make_paths(tmp_path)
    machine = FakeMachine()
    runner = FakeRunner(machine=machine)
    stack = controller(paths, runner, machine, tmp_path / "campaign")

    @contextlib.asynccontextmanager
    async def hung_session():
        await asyncio.Event().wait()
        yield None

    stack.probe = ads_stack.bridge_probe(hung_session)
    stack.ready_deadline_s = 0.2
    with pytest.raises(StackError, match="stack_not_ready"):
        asyncio.run(asyncio.wait_for(stack.prepare(None, TASK), 5))
    assert sorted(kills(runner)) == sorted([CONTROL_IMAGE, EMULATOR_IMAGE]) and stack.pids == {}


def test_fatal_probe_error_is_not_retried():
    calls = []

    async def wrong_destination():
        calls.append(1)
        raise StackError("bridge_endpoint_mismatch")

    async def open_port():
        return True

    with pytest.raises(StackError, match="bridge_endpoint_mismatch"):
        asyncio.run(wait_ready(wrong_destination, port_open=open_port, sleep=no_sleep))
    assert calls == [1]


# Final review: the cleanup Recipe session is gated before abort_recipe.
def test_cleanup_session_not_on_emulator_is_never_aborted(tmp_path):
    paths, baseline = frozen_recipes(tmp_path)
    machine = FakeMachine(profile="ADS_MACHINE", running=True)
    with pytest.raises(StackError, match="profile_not_emulator"):
        asyncio.run(finish_trial(machine.session_factory(), tmp_path / "dest", datetime.now(timezone.utc),
                                 status_session=FakeMachine().session_factory(), paths=paths,
                                 recipes_baseline=baseline, sleep=no_sleep))
    assert machine.calls == ["describe_machine"]


def test_bridge_session_refuses_a_config_that_left_the_frozen_destination(tmp_path):
    from test_manifest import ENDPOINT, write_bridge_configs
    frozen = {key: dict(ENDPOINT) for key in ("read", "recipe", "manual")}
    write_bridge_configs(tmp_path, {"host": "10.0.0.5", "port": 8026})

    async def scenario():
        async with ads_stack.bridge_session("recipe", tmp_path, frozen):
            raise AssertionError("bridge started")

    with pytest.raises(StackError, match="bridge_endpoint_mismatch"):
        asyncio.run(scenario())
    write_bridge_configs(tmp_path, {"host": "127.0.0.1", "port": 9000})
    with pytest.raises(StackError, match="bridge_endpoint_mismatch"):
        asyncio.run(scenario())
