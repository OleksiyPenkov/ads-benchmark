"""APS.Mcp bridge per API key: tool sets per key tier (spec 6.2) and stdio server parameters.

Each key has its own bridge config, `apsmcp.<key>.json`, in a secrets directory outside the
repository and the campaign (`%LOCALAPPDATA%\\llm-bench-ads\\`). The harness reads only the
config's `host` and `port` (its destination, frozen in campaign.json and checked before every
connection); the API key is read by the bridge alone, so it cannot reach a journal.
"""
from __future__ import annotations

import json
import os
from collections.abc import Collection
from pathlib import Path

BRIDGE = Path(os.environ.get("ADS_BENCH_MCP_BIN") or (Path.home() / "AppData" / "Local" / "Programs" / "APS Control Community" / "mcp" / "APS.Mcp.exe"))
SECRETS_SUBDIR = "llm-bench-ads"

_READ = ["describe_machine", "describe_hardware", "describe_recipes", "describe_dsl", "describe_interlocks",
         "describe_alarms", "system_status", "device_snapshot", "tail_log", "recipe_status", "list_runs",
         "fetch_run_files", "compute_deposition_plan"]
_RECIPE = _READ + ["validate_recipe", "upload_recipe", "start_recipe", "abort_recipe", "resolve_pause"]
KEY_TOOLS: dict[str, list[str]] = {"read": _READ, "recipe": _RECIPE, "manual": _RECIPE + ["device_command"]}
# Suite key -> (key name in AgentServer.xml, tier it must grant there).
KEY_NAMES: dict[str, tuple[str, str]] = {"read": ("lab-read", "Read"), "recipe": ("paper-arm-emulator", "Recipe"),
                                         "manual": ("behavior-validation", "Manual")}
# Spec 6.2: short tasks on the Read or Recipe key get only read tools and validate_recipe.
SHORT_TOOLS = frozenset(_READ + ["validate_recipe"])
# The only bridge destinations a campaign accepts: the emulator stack on this workstation.
LOCAL_HOSTS = frozenset(("127.0.0.1", "localhost"))


class EndpointError(ValueError):
    """A bridge config whose destination is unreadable, not local, or not the frozen one."""


def default_secrets_dir() -> Path:
    """%LOCALAPPDATA%\\llm-bench-ads, resolved when called."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise RuntimeError("LOCALAPPDATA is not set; the bridge configs cannot be located")
    return Path(local) / SECRETS_SUBDIR


def config_path(key: str, secrets_dir: Path) -> Path:
    if key not in KEY_TOOLS:
        raise ValueError("Unknown bridge key: " + str(key))
    return Path(secrets_dir) / f"apsmcp.{key}.json"


def bridge_params(key: str, secrets_dir: Path):
    """StdioServerParameters for APS.Mcp with the config of `key`; the bridge, not the harness, reads the key."""
    from mcp import StdioServerParameters

    return StdioServerParameters(command=str(BRIDGE), args=["--config", str(config_path(key, secrets_dir))],
                                 cwd=str(BRIDGE.parent))


def bridge_endpoint(key: str, secrets_dir: Path) -> dict:
    """{"host", "port"} of the key's config; no other field (the API key in particular) is returned."""
    try:
        config = json.loads(config_path(key, secrets_dir).read_text(encoding="utf-8-sig"))
        endpoint = {"host": config["host"], "port": config["port"]}
    except (OSError, ValueError, KeyError, TypeError):
        raise EndpointError("bridge config of key " + key + " has no readable host/port") from None
    if not isinstance(endpoint["host"], str) or type(endpoint["port"]) is not int:
        raise EndpointError("bridge config of key " + key + " has an invalid host/port")
    return endpoint


def bridge_endpoints(secrets_dir: Path) -> dict[str, dict]:
    """Destinations of all three key configs, each required to be local; frozen into campaign.json."""
    endpoints = {key: bridge_endpoint(key, secrets_dir) for key in KEY_TOOLS}
    for key, endpoint in endpoints.items():
        if endpoint["host"] not in LOCAL_HOSTS:
            raise EndpointError("bridge config of key " + key + " does not point at this workstation")
    return endpoints


def check_endpoint(key: str, secrets_dir: Path, frozen: dict) -> None:
    """Refuse a config whose host/port differ from the frozen ones or whose host is not local."""
    endpoint = bridge_endpoint(key, secrets_dir)
    if endpoint != frozen.get(key) or endpoint["host"] not in LOCAL_HOSTS:
        raise EndpointError("bridge config of key " + key + " differs from the frozen destination")


def check_task_tools(task: dict, local_tools: Collection[str] = ()) -> None:
    """Every task tool must be a registered local tool or lie inside its key's tier.

    `local_tools` are the manifest's cls == "local" names (host tools such as skill_read); they never
    reach the bridge. Any other name goes to APS.Mcp and must be in the key's set; Read/Recipe short
    tasks get only read tools and validate_recipe.
    """
    key = task.get("key")
    if key not in KEY_TOOLS:
        raise ValueError("Task " + str(task.get("id")) + " has no valid key (read, recipe or manual)")
    allowed = set(KEY_TOOLS[key])
    if key in ("read", "recipe") and task.get("type") == "short":
        allowed &= SHORT_TOOLS
    denied = [name for name in task.get("tools", []) if name not in local_tools and name not in allowed]
    if denied:
        raise ValueError("Task " + str(task.get("id")) + " uses tools outside the " + key + " key: " + ", ".join(denied))
