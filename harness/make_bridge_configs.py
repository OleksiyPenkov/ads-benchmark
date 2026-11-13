"""Write the per-key bridge configs apsmcp.<key>.json into %LOCALAPPDATA%\\llm-bench-ads\\.

Each config is a copy of apsmcp.json with only `apiKey` replaced.
AgentServer.xml stores each key as a SHA-256 hash with its tier, so the script checks the tier
there and takes the plaintext key from the first existing local config whose string value has
that hash. Output lists file paths and key names only; key values are never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from bridge import KEY_NAMES, config_path, default_secrets_dir

_APP_DIR = Path(
    os.environ.get("ADS_BENCH_APP_DIR")
    or (Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        / "Programs" / "APS Control Community" / "bin")
)
AGENT_SERVER = _APP_DIR / "AgentServer.xml"
TEMPLATE = _APP_DIR.parent / "mcp" / "apsmcp.json"
KEY_SOURCES = (
    TEMPLATE,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def registered_keys(agent_server: Path) -> dict[str, tuple[str, str]]:
    """Key name -> (upper-case SHA-256 hash, tier) from AgentServer.xml."""
    root = ET.parse(agent_server).getroot()
    return {key.get("name"): ((key.get("hash") or "").upper(), key.get("tier")) for key in root.iter("Key")}


def find_key(expected_hash: str, sources) -> str:
    for source in sources:
        if not Path(source).is_file():
            continue
        for text in _strings(_load_json(Path(source))):
            if _sha256(text) == expected_hash:
                return text
    raise ValueError("no local config holds this key")


def make_configs(secrets_dir: Path, *, agent_server: Path = AGENT_SERVER, template: Path = TEMPLATE,
                 sources=KEY_SOURCES) -> list[tuple[Path, str]]:
    """Check every key name and tier first, then write the three configs; returns (path, key name)."""
    registered = registered_keys(agent_server)
    base = _load_json(template)
    if not isinstance(base, dict) or "apiKey" not in base:
        raise ValueError("Bridge config template has no apiKey: " + str(template))
    plan = []
    for key, (name, tier) in KEY_NAMES.items():
        if name not in registered:
            raise ValueError("Key " + name + " is not registered in " + str(agent_server))
        expected_hash, actual_tier = registered[name]
        if actual_tier != tier:
            raise ValueError("Key " + name + " has tier " + str(actual_tier) + ", expected " + tier)
        try:
            secret = find_key(expected_hash, sources)
        except ValueError:
            raise ValueError("No local config holds the key " + name) from None
        plan.append((key, name, secret))
    secrets_dir = Path(secrets_dir)
    secrets_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, name, secret in plan:
        path = config_path(key, secrets_dir)
        path.write_text(json.dumps({**base, "apiKey": secret}, indent=2) + "\n", encoding="utf-8")
        written.append((path, name))
    return written


def main() -> int:
    try:
        written = make_configs(default_secrets_dir())
    except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
        print("REFUSED " + str(exc), file=sys.stderr)
        return 1
    for path, name in written:
        print(str(path) + "  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
