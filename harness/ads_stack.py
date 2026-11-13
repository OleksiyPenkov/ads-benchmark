"""ADS emulator stack: reset, readiness, gate, recipe library and per-trial artifacts.

Agents only ever drive the emulator profile ADS; a trial whose bridge reports any other
profile (ADS_MACHINE in particular) is refused before the prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import functools
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from bridge import EndpointError, bridge_params, check_endpoint, default_secrets_dir

PROFILE = os.environ.get("ADS_BENCH_PROFILE", "ADS")
CONTROL_IMAGE = "APS.ControlCommunity.exe"
EMULATOR_IMAGE = "APS.EmulatorCommunity.exe"
STACK_IMAGES = (CONTROL_IMAGE, EMULATOR_IMAGE)
AGENT_HOST, AGENT_PORT = "127.0.0.1", 8026   # AgentServer.xml
# Written by Control at run time and never read back as configuration.
HASH_EXCLUDE = ("AuditLog/",)
AGENT_SERVER_KEY = "bin/AgentServer.xml"
ROOT_CONFIG_KEY = "bin/RootConfig.xml"
MUTATING_TOOLS = frozenset(("upload_recipe", "start_recipe", "abort_recipe", "resolve_pause", "device_command"))
# Preparation failures that stop the whole campaign: the machine setup is wrong or unknown.
CAMPAIGN_FATAL = frozenset(("profile_not_emulator", "foreign_stack_running", "stack_stop_failed",
                            "root_config_not_emulator", "bridge_endpoint_mismatch", "bridge_not_emulator"))
# Frozen campaign inputs: the expected profile hashes and a byte copy of the profile's Recipes\.
STACK_BASELINE = "inputs/stack-baseline.json"
RECIPES_BASELINE = "inputs/recipes-baseline"
STOP_TIMEOUT_S = 30
EMULATOR_HEAD_START_S = 5
READY_DEADLINE_S = 90
PROBE_RETRY_DELAY_S = 2
ABORT_IDLE_TIMEOUT_S = 120


class StackError(Exception):
    """A fixed failure category; it becomes the trial's error_category."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_APP_DIR = Path(
    os.environ.get("ADS_BENCH_APP_DIR")
    or (Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        / "Programs" / "APS Control Community" / "bin")
)
_DEFAULT_CONTROL_BIN = Path(os.environ.get("ADS_BENCH_CONTROL_BIN") or (_DEFAULT_APP_DIR / CONTROL_IMAGE))
_DEFAULT_EMULATOR_BIN = Path(os.environ.get("ADS_BENCH_EMULATOR_BIN") or (_DEFAULT_APP_DIR / EMULATOR_IMAGE))
_DEFAULT_CONFIGS_DIR = Path(os.environ.get("ADS_BENCH_CONFIGS_DIR") or (_REPO_ROOT / "configs"))
_DEFAULT_SNAPSHOT_DIR = Path(os.environ.get("ADS_BENCH_SNAPSHOT_DIR") or (_REPO_ROOT / "snapshots"))


@dataclass(frozen=True)
class StackPaths:
    emulator: Path = _DEFAULT_EMULATOR_BIN
    control: Path = _DEFAULT_CONTROL_BIN
    profile_dir: Path = _DEFAULT_CONFIGS_DIR / PROFILE
    agent_server_xml: Path = _DEFAULT_APP_DIR / "AgentServer.xml"
    # Control's own profile selector (ConfigDir + Prefix); checked before Control is started.
    root_config: Path = _DEFAULT_APP_DIR / "RootConfig.xml"
    snapshot: Path = _DEFAULT_SNAPSHOT_DIR / f"{PROFILE}_Idle.emstate"
    # AppPaths.cs takes DataDir/LogsDir from Core.Configuration.cs:100-106: blank in RootConfig.xml,
    # so WorkDir\Data and WorkDir\Logs, with WorkDir = the app base directory.
    data_dir: Path = _DEFAULT_APP_DIR / "Data"
    logs_dir: Path = _DEFAULT_APP_DIR / "Logs"

    @property
    def profile(self) -> str:
        return self.profile_dir.name

    @property
    def recipes_dir(self) -> Path:
        # Core.Configuration.cs:96-98: blank RecipesDir resolves to ConfigDir\<Prefix>\Recipes.
        return self.profile_dir / "Recipes"

    @classmethod
    def for_profile(cls, profile: str) -> "StackPaths":
        return cls(
            profile_dir=_DEFAULT_CONFIGS_DIR / profile,
            snapshot=_DEFAULT_SNAPSHOT_DIR / f"{profile}_Idle.emstate",
        )


# Identity of a process: (image name, creation time). A PID alone can be reused by a new process.
Identity = tuple[str, str | None]


class ProcessRunner(Protocol):
    def processes(self) -> dict[int, Identity]: ...
    def kill(self, pid: int) -> None: ...
    def start(self, argv: list[str], env: dict, cwd: Path) -> int: ...


class WindowsRunner:
    """Win32_Process query / taskkill by PID / Popen; the harness only ever kills PIDs it started."""

    _QUERY = ("Get-CimInstance Win32_Process -Filter \"" + " OR ".join("Name='" + i + "'" for i in STACK_IMAGES)
              + "\" | ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId, $_.Name,"
              " $_.CreationDate.ToUniversalTime().ToString('o') }")

    def processes(self) -> dict[int, Identity]:
        """Running Control/emulator processes: PID -> (image, UTC creation time)."""
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", self._QUERY],
                                capture_output=True, check=True).stdout
        found = {}
        for line in output.decode("utf-8", errors="replace").splitlines():
            pid, _, rest = line.strip().partition("|")
            image, _, created = rest.partition("|")
            if pid.isdigit() and image:
                found[int(pid)] = (image, created or None)
        return found

    def kill(self, pid: int) -> None:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False)

    def start(self, argv: list[str], env: dict, cwd: Path) -> int:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        return subprocess.Popen(argv, env=env, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True).pid


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def profile_hashes(paths: StackPaths) -> dict[str, str]:
    """SHA-256 of every profile file (relative POSIX keys, HASH_EXCLUDE skipped) plus AgentServer.xml and RootConfig.xml."""
    hashes = {}
    for path in sorted(paths.profile_dir.rglob("*")):
        key = path.relative_to(paths.profile_dir).as_posix()
        if path.is_file() and not key.startswith(HASH_EXCLUDE):
            hashes[key] = _sha256(path)
    hashes[AGENT_SERVER_KEY] = _sha256(paths.agent_server_xml)
    hashes[ROOT_CONFIG_KEY] = _sha256(paths.root_config)
    return hashes


def check_root_config(paths: StackPaths) -> None:
    """Control reads its profile from RootConfig.xml; refuse unless it selects ConfigDir\\ADS."""
    try:
        root = ET.parse(paths.root_config).getroot()
    except (OSError, ET.ParseError):
        raise StackError("root_config_not_emulator") from None
    config_dir, prefix = (root.findtext("ConfigDir") or "").strip(), (root.findtext("Prefix") or "").strip()
    expected = os.path.normcase(os.path.normpath(str(paths.profile_dir.parent)))
    if not config_dir or os.path.normcase(os.path.normpath(config_dir)) != expected or prefix != paths.profile:
        raise StackError("root_config_not_emulator")


def stack_processes(runner: ProcessRunner) -> dict[int, Identity]:
    """Running Control/emulator processes by PID, matched by exact image name, with their identity."""
    return {pid: ident for pid, ident in runner.processes().items() if ident[0] in STACK_IMAGES}


def stop_stack(runner: ProcessRunner, owned: dict[int, Identity], *, timeout_s: float = STOP_TIMEOUT_S,
               sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
    """Kill the owned PIDs (Control first) that are still the same process, and wait until they are gone.

    A PID whose image or creation time changed was reused by another process and is never touched.
    """
    def alive() -> dict[int, Identity]:
        running = runner.processes()
        return {pid: ident for pid, ident in owned.items() if running.get(pid) == ident}

    for image in STACK_IMAGES:
        for pid, ident in alive().items():
            if ident[0] == image:
                runner.kill(pid)
    deadline = clock() + timeout_s
    while alive():
        if clock() >= deadline:
            raise StackError("stack_stop_failed")
        sleep(0.5)


def start_stack(paths: StackPaths, runner: ProcessRunner, env: dict, *,
                sleep: Callable[[float], None] = time.sleep) -> dict[int, Identity]:
    """Check Control's RootConfig, start the emulator, then Control; returns the started processes' identities.

    A RootConfig that does not select the emulator profile starts nothing. A failed Control start stops the emulator.
    """
    check_root_config(paths)
    if not paths.snapshot.is_file():
        raise StackError("snapshot_missing")

    def launch(argv, launch_env, cwd, image):
        pid = runner.start(argv, env=launch_env, cwd=cwd)
        # Creation time is taken now; (image, None) never matches a live process, so it is never killed.
        ident = runner.processes().get(pid)
        return pid, ident if ident is not None and ident[0] == image else (image, None)

    pid, ident = launch([str(paths.emulator), "--project", paths.profile, "--state", str(paths.snapshot)],
                        dict(env), paths.emulator.parent, EMULATOR_IMAGE)
    started = {pid: ident}
    try:
        sleep(EMULATOR_HEAD_START_S)
        pid, ident = launch([str(paths.control)], {**env, "APS_DEV_CONTROL_MODE": "Remote"},
                            paths.control.parent, CONTROL_IMAGE)
        started[pid] = ident
    except BaseException:
        stop_stack(runner, started, sleep=sleep)
        raise
    return started


async def agent_port_open(host: str = AGENT_HOST, port: int = AGENT_PORT) -> bool:
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=1)
    except (OSError, TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return True


async def wait_ready(probe: Callable[[], Awaitable[dict]], deadline_s: float = READY_DEADLINE_S, *,
                     port_open: Callable[[], Awaitable[bool]] = agent_port_open,
                     sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                     clock: Callable[[], float] = time.monotonic) -> dict:
    """Wait for the agent port, then probe; the first call after a Control restart may fail once.

    Each probe, including its bridge session teardown, is bounded by the time left to the deadline.
    A campaign-fatal StackError from the probe (a wrong bridge destination) is raised as is.
    """
    deadline = clock() + deadline_s
    while not await port_open():
        if clock() >= deadline:
            raise StackError("stack_not_ready")
        await sleep(1)

    async def bounded() -> dict:
        remaining = deadline - clock()
        if remaining <= 0:
            raise StackError("stack_not_ready")
        try:
            return await asyncio.wait_for(probe(), timeout=remaining)
        except TimeoutError:
            raise StackError("stack_not_ready") from None

    for attempt in range(2):
        try:
            return await bounded()
        except StackError as exc:
            if exc.code != "stack_not_ready" or attempt:
                raise
            if clock() >= deadline:
                raise
        except Exception:
            if attempt:
                raise StackError("stack_not_ready") from None
        await sleep(PROBE_RETRY_DELAY_S)
    raise StackError("stack_not_ready")


def _chambers_idle(status: dict) -> bool:
    chambers = status.get("chambers")
    return (isinstance(chambers, list) and bool(chambers)
            and all(isinstance(c, dict) and c.get("state") == "Idle" for c in chambers)
            and status.get("isRecipeRunning") is False)


def check_gate(machine: dict, status: dict, expected_hashes: dict, paths: StackPaths) -> None:
    if machine.get("profile") != paths.profile or machine.get("isEmulator") is False:
        raise StackError("profile_not_emulator")
    if profile_hashes(paths) != expected_hashes:
        raise StackError("profile_hash_mismatch")
    if not _chambers_idle(status) or status.get("activeAlarms") != []:
        raise StackError("stack_not_idle")


def needs_reset(previous: dict | None, task: dict) -> bool:
    """previous is the preceding trial's result (status, journal_file); None for the first trial."""
    if previous is None or previous.get("status") != "completed" or task.get("fresh_state"):
        return True
    try:
        with Path(previous["journal_file"]).open(encoding="utf-8") as journal:
            for line in journal:
                record = json.loads(line)
                if record.get("type") == "tool_call" and record.get("name") in MUTATING_TOOLS:
                    return True
    except (OSError, KeyError, TypeError, ValueError):
        return True
    return False


def _is_link(path: Path) -> bool:
    """Symlink, junction or any other Windows reparse point."""
    if not os.path.lexists(path):
        return False
    if path.is_symlink() or path.is_junction():
        return True
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _contained(root: Path, relative: str) -> Path:
    """root/relative, refused if any component from root down is a link or it resolves outside root."""
    path = root
    if _is_link(root):
        raise StackError("recipe_path_unsafe")
    for part in Path(relative).parts:
        path = path / part
        if _is_link(path):
            raise StackError("recipe_path_unsafe")
    if not path.resolve().is_relative_to(root.resolve()) or path.resolve() == root.resolve():
        raise StackError("recipe_path_unsafe")
    return path


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    return target


def _relative_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file() or _is_link(p)}


def restore_recipes(recipes_dir: Path, baseline: Path, dest: Path) -> list[Path]:
    """Move recipes outside the baseline into dest, keep changed ones there, restore baseline bytes.

    Every path is validated (no link anywhere below the roots, resolved inside them) before any
    file is copied, overwritten or deleted; a recipe is copied out before it is deleted.
    """
    live, frozen = _relative_files(recipes_dir), _relative_files(baseline)
    saved = dest / "Recipes"
    for name in live.keys() | frozen.keys():
        _contained(recipes_dir, name)
        _contained(saved, name)
        if name in frozen:
            _contained(baseline, name)
    copied = []
    for name, path in live.items():
        if name not in frozen:
            copied.append(_copy(path, saved / name))
            path.unlink()
        elif path.read_bytes() != frozen[name].read_bytes():
            copied.append(_copy(path, saved / name))
            _copy(frozen[name], path)
    if frozen and not recipes_dir.exists() and not _is_link(recipes_dir):
        recipes_dir.mkdir(parents=True, exist_ok=True)
    for name, path in frozen.items():
        if name not in live:
            _copy(path, _contained(recipes_dir, name))
    return copied


def collect_artifacts(paths: StackPaths, dest: Path, started_utc: datetime) -> list[Path]:
    since = started_utc.timestamp()
    selections = (("Data", paths.data_dir, ("*.tsv", "*.run.json")), ("Logs", paths.logs_dir, ("Process_*.log",)))
    copied = []
    for folder, root, patterns in selections:
        names = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
        copied += [_copy(p, dest / folder / p.name) for p in names if p.stat().st_mtime > since]
    return copied


# Written by finish_trial into results\\<trial_id>\\; AC1 reads it (grading.py).
FINAL_STATUS = "final_status.json"


def _write_final_status(path: Path, status: dict, aborted: bool | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"system_status": status, "aborted": aborted}, indent=1) + "\n",
                    encoding="utf-8", newline="\n")


async def finish_trial(session_factory, dest: Path, started_utc: datetime, *, status_session,
                       paths: StackPaths = StackPaths(), recipes_baseline: Path,
                       idle_timeout_s: float = ABORT_IDLE_TIMEOUT_S,
                       sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                       clock: Callable[[], float] = time.monotonic) -> list[Path]:
    """Record the end state, abort a recipe the agent left running, collect the artifacts and restore Recipes\\.

    status_session() (Read tier) and session_factory() (Recipe tier) are async context managers yielding a
    bridge session with `async call(name, arguments=None) -> dict`. Before anything else, the system_status
    reply the agent left behind is written to dest\\final_status.json as {"system_status": ..., "aborted": null}
    before any abort; once the abort path ends the file is rewritten with "aborted" true or false. AC1 is scored from that file. If the status read fails, no file is
    written and the trial is unscorable. The Recipe session must report describe_machine.profile == ADS
    before recipe_status or abort_recipe is called; otherwise it is left untouched (profile_not_emulator).
    A campaign-fatal StackError (wrong destination or profile) takes precedence over other finish errors.
    """
    error = None
    snapshot = None
    try:
        async with status_session() as session:
            snapshot = await session.call("system_status")
    except StackError as exc:
        if exc.code in CAMPAIGN_FATAL:
            error = exc
    except Exception:
        pass
    copied = []
    final = dest / FINAL_STATUS
    if isinstance(snapshot, dict):
        # Persisted before the abort path, so a cancelled abort still leaves the end state; "aborted" is null
        # until the abort outcome is known.
        _write_final_status(final, snapshot, None)
        copied.append(final)
    else:
        snapshot, error = None, error or StackError("finish_status_failed")
    aborted = False
    try:
        async with session_factory() as session:
            desc = await session.call("describe_machine")
            if desc.get("profile") != paths.profile or desc.get("isEmulator") is False:
                raise StackError("profile_not_emulator")
            if (await session.call("recipe_status")).get("isRecipeRunning"):
                aborted = True
                await session.call("abort_recipe")
                deadline = clock() + idle_timeout_s
                while not _chambers_idle(await session.call("system_status")):
                    if clock() >= deadline:
                        raise StackError("abort_not_idle")
                    await sleep(2)
    except StackError as exc:
        if error is None or (exc.code in CAMPAIGN_FATAL and error.code not in CAMPAIGN_FATAL):
            error = exc
    except Exception:
        error = error or StackError("finish_session_failed")
    if snapshot is not None:
        _write_final_status(final, snapshot, aborted)
    copied += collect_artifacts(paths, dest, started_utc) + restore_recipes(paths.recipes_dir, recipes_baseline, dest)
    if error is not None:
        raise error
    return copied


def freeze_baseline(paths: StackPaths, campaign: Path) -> dict[str, str]:
    """Write the campaign's frozen profile hashes and Recipes\\ copy; both must then be frozen inputs."""
    campaign = Path(campaign)
    target = campaign / RECIPES_BASELINE
    if target.exists() or (campaign / STACK_BASELINE).exists():
        raise FileExistsError("stack baseline already frozen in " + str(campaign))
    hashes = profile_hashes(paths)
    if paths.recipes_dir.is_dir():
        shutil.copytree(paths.recipes_dir, target)
    else:
        target.mkdir(parents=True)
    (campaign / STACK_BASELINE).write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
    return hashes


def load_baseline(campaign: Path, input_sha256: dict[str, str]) -> tuple[dict[str, str], Path]:
    """The frozen expected hashes and the recipe baseline folder, checked against each other."""
    campaign = Path(campaign)
    path = campaign / STACK_BASELINE
    if STACK_BASELINE not in input_sha256 or _sha256(path) != input_sha256[STACK_BASELINE]:
        raise ValueError("Stack baseline is not a frozen campaign input: " + STACK_BASELINE)
    hashes = json.loads(path.read_text(encoding="utf-8"))
    baseline = campaign / RECIPES_BASELINE
    expected = {key[len("Recipes/"):]: value for key, value in hashes.items() if key.startswith("Recipes/")}
    actual = {name: _sha256(p) for name, p in _relative_files(baseline).items()}
    if actual != expected:
        raise ValueError("Campaign recipe baseline differs from the frozen profile hashes")
    return hashes, baseline


@contextlib.asynccontextmanager
async def bridge_session(key: str, secrets_dir: Path, endpoints: dict | None = None):
    """One APS.Mcp stdio session; the key's config selects the API key and therefore the tier.

    With `endpoints` (the campaign's frozen host/port per key), a config whose destination differs or is
    not local is refused with bridge_endpoint_mismatch before the bridge is started.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    if endpoints is not None:
        try:
            check_endpoint(key, secrets_dir, endpoints)
        except EndpointError:
            raise StackError("bridge_endpoint_mismatch") from None
    params = bridge_params(key, secrets_dir)
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                yield BridgeSession(session)


class BridgeSession:
    def __init__(self, session):
        self.session = session

    async def call(self, name: str, arguments: dict | None = None) -> dict:
        result = await self.session.call_tool(name, arguments or {})
        if result.isError:
            raise StackError("bridge_call_failed")
        text = "".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        try:
            value = json.loads(text)
        except ValueError:
            return {"text": text}
        return value if isinstance(value, dict) else {"value": value}


def bridge_probe(read_session) -> Callable[[], Awaitable[dict]]:
    """describe_machine + system_status in a fresh Read-tier session."""
    async def probe() -> dict:
        async with read_session() as session:
            return {"machine": await session.call("describe_machine"), "status": await session.call("system_status")}
    return probe


class StackController:
    """Owns the stack it started for one campaign process, by PID; it never touches any other stack."""

    def __init__(self, *, paths: StackPaths, runner: ProcessRunner, read_session, recipe_session,
                 expected_hashes: dict[str, str], recipes_baseline: Path, env: dict,
                 port_open: Callable[[], Awaitable[bool]] = agent_port_open,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 sync_sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, ready_deadline_s: float = READY_DEADLINE_S):
        self.paths, self.runner = paths, runner
        self.ready_deadline_s = ready_deadline_s
        self.read_session, self.recipe_session = read_session, recipe_session
        self.probe = bridge_probe(read_session)
        self.expected_hashes, self.recipes_baseline, self.env = expected_hashes, recipes_baseline, env
        self.port_open, self.sleep, self.sync_sleep, self.clock = port_open, sleep, sync_sleep, clock
        self.pids: dict[int, str] = {}

    @classmethod
    def for_campaign(cls, campaign: Path, input_sha256: dict[str, str], *, read_session, recipe_session,
                     paths: StackPaths | None = None) -> "StackController":
        if paths is None:
            cfg_file = Path(campaign) / "campaign.json"
            profile = PROFILE
            if cfg_file.is_file():
                try:
                    profile = ((json.loads(cfg_file.read_text(encoding="utf-8")).get("machine") or {}).get("profile")) or PROFILE
                except (OSError, ValueError):
                    profile = PROFILE
            paths = StackPaths() if profile == PROFILE else StackPaths.for_profile(profile)
        hashes, baseline = load_baseline(campaign, input_sha256)
        return cls(paths=paths, runner=WindowsRunner(), read_session=read_session, recipe_session=recipe_session,
                   expected_hashes=hashes, recipes_baseline=baseline, env=os.environ.copy())

    async def prepare(self, previous: dict | None, task: dict) -> dict:
        """Reset if needed, probe, gate; returns the stack_verified journal record.

        A Control or emulator process this controller did not start refuses the trial with
        foreign_stack_running and nothing is touched. Any later failure stops the owned PIDs,
        except profile_not_emulator: that stack is not the emulator, so it is left alone.
        """
        running = await asyncio.to_thread(stack_processes, self.runner)
        if any(self.pids.get(pid) != image for pid, image in running.items()):
            # Ownership of our own PIDs is kept, so close() still stops them at campaign end.
            raise StackError("foreign_stack_running")
        reset = running != self.pids or len(self.pids) != len(STACK_IMAGES) or needs_reset(previous, task)
        try:
            if reset:
                await asyncio.to_thread(stop_stack, self.runner, self.pids, sleep=self.sync_sleep, clock=self.clock)
                self.pids = {}
                try:
                    self.pids = await asyncio.to_thread(start_stack, self.paths, self.runner, self.env,
                                                        sleep=self.sync_sleep)
                except StackError:
                    raise
                except Exception:
                    raise StackError("stack_start_failed") from None
            ready = await wait_ready(self.probe, self.ready_deadline_s, port_open=self.port_open, sleep=self.sleep,
                                     clock=self.clock)
            check_gate(ready["machine"], ready["status"], self.expected_hashes, self.paths)
        except BaseException as exc:
            owned, self.pids = self.pids, {}
            if not (isinstance(exc, StackError) and exc.code == "profile_not_emulator"):
                stop_stack(self.runner, owned, sleep=self.sync_sleep, clock=self.clock)
            raise
        return {"type": "stack_verified", "profile": self.paths.profile, "reset": reset, "profile_files": len(self.expected_hashes)}

    async def close(self) -> None:
        """Stop the stack this controller started, if any (end of campaign)."""
        owned, self.pids = self.pids, {}
        await asyncio.to_thread(stop_stack, self.runner, owned, sleep=self.sync_sleep, clock=self.clock)

    async def finish(self, dest: Path, started_utc: datetime) -> list[Path]:
        return await finish_trial(self.recipe_session, dest, started_utc, status_session=self.read_session, paths=self.paths,
                                  recipes_baseline=self.recipes_baseline, sleep=self.sleep)


def selftest() -> int:
    """Start the stack, probe it read-only with the Read key, print the gate inputs, and stop the started PIDs again."""
    paths, runner = StackPaths(), WindowsRunner()
    read_session = functools.partial(bridge_session, "read", default_secrets_dir())
    running = stack_processes(runner)
    if running:
        print("SELFTEST_REFUSED " + json.dumps({"running": sorted({ident[0] for ident in running.values()})}), flush=True)
        return 2
    pids: dict[int, Identity] = {}
    try:
        pids = start_stack(paths, runner, os.environ.copy())
        ready = asyncio.run(wait_ready(bridge_probe(read_session)))
        machine, status = ready["machine"], ready["status"]
        hashes = profile_hashes(paths)
        check_gate(machine, status, hashes, paths)
        print("SELFTEST_PASS " + json.dumps({
            "profile": machine.get("profile"), "controlMode": machine.get("controlMode"),
            "profile_files": len(hashes),
            "chambers": [{"id": c.get("id"), "state": c.get("state")} for c in status.get("chambers", [])],
            "activeAlarms": status.get("activeAlarms"), "isRecipeRunning": status.get("isRecipeRunning"),
            "started_pids_are_the_stack": stack_processes(runner) == pids}), flush=True)
        return 0
    except StackError as exc:
        print("SELFTEST_FAIL " + json.dumps({"error_category": exc.code}), flush=True)
        return 1
    finally:
        stop_stack(runner, pids)
        remaining = sorted({ident[0] for ident in stack_processes(runner).values()})
        print("SELFTEST_STACK_STOPPED " + json.dumps({"remaining": remaining}),
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="Start the stack from the All_Pumped snapshot, probe it read-only, then stop it")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do; pass --selftest")
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
