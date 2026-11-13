"""Host tool that delivers the frozen skill snapshot to the agent: skill_read."""
from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath

DEFAULT_SKILL_DIR = Path(
    os.environ.get(
        "ADS_BENCH_SKILL_DIR",
        str((Path(__file__).resolve().parents[2] / "aps-operator-skill" / "skills" / "aps-operator").resolve()),
    )
).as_posix() + "/"

SOURCE_ROOTS = (
    (DEFAULT_SKILL_DIR, "aps-operator"),
)
OUTSIDE = "path outside the skill snapshot"
MISSING = "file not found"

SKILL_READ = {
    "name": "skill_read",
    "description": ("Read one file of the aps-operator skill as UTF-8 text. The path is "
                    "relative to the skill snapshot (for example aps-operator/SKILL.md or "
                    "aps-operator/references/dsl-and-safety.md) or starts with the skill's source path ("
                    + ", ".join(root for root, _ in SOURCE_ROOTS)
                    + ") as the skill text names it."),
    "loadMode": "essential",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                   "additionalProperties": False},
}


def _result(text, is_error):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class LocalTools:
    def __init__(self, snapshot: Path, source_roots):
        """`source_roots` pairs an absolute source folder with the snapshot folder it maps to."""
        self.snapshot = Path(snapshot).resolve()
        self.source_roots = []
        for source, folder in source_roots:
            root = source.replace("\\", "/")
            self.source_roots.append((root if root.endswith("/") else root + "/", folder))
        self.names = frozenset(d["name"] for d in self.declarations())

    def declarations(self) -> list[dict]:
        return json.loads(json.dumps([SKILL_READ]))

    async def call(self, name: str, arguments: dict) -> dict:
        if name not in self.names:
            raise ValueError("Unknown local tool: " + str(name))
        return self.read(arguments["path"])

    def target(self, path):
        """The resolved snapshot file for `path`, or None when it resolves outside its boundary.

        A relative path is relative to the snapshot. An absolute path under a source root maps into that root's
        snapshot folder. The boundary is fixed by the request before any link is followed: the root's folder for an
        absolute path; for a relative path, the skill folder it names, else the snapshot. A link, the skill folder
        itself included, that leads out of the boundary is refused, so one skill never serves the other's files."""
        normal = path.replace("\\", "/")
        if normal.startswith("//"):
            return None
        pure = PureWindowsPath(normal)
        if pure.drive or pure.root:
            # Windows paths compare case-insensitively, drive letter included.
            for root, folder in self.source_roots:
                if normal.casefold().startswith(root.casefold()):
                    normal = normal[len(root):]
                    boundary = self.snapshot / folder
                    break
            else:
                return None
            if PureWindowsPath(normal).drive or PureWindowsPath(normal).root:
                return None
            requested = boundary / normal
        else:
            requested = self.snapshot / normal
            lexical = Path(os.path.normpath(requested))
            boundary = self.snapshot
            for _, folder in self.source_roots:
                if lexical.is_relative_to(self.snapshot / folder):
                    boundary = self.snapshot / folder
        try:
            # resolve() follows symlinks and junctions, so a link out of the boundary is caught here.
            resolved = requested.resolve()
        except (OSError, ValueError):
            return None
        if not resolved.is_relative_to(boundary):
            return None
        return resolved

    def read(self, path):
        target = self.target(path)
        if target is None:
            return _result(OUTSIDE, True)
        if not target.is_file():
            return _result(MISSING, True)
        try:
            return _result(target.read_text(encoding="utf-8"), False)
        except UnicodeDecodeError:
            return _result("file is not UTF-8 text", True)
        except OSError:
            return _result(MISSING, True)
