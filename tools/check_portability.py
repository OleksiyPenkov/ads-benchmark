from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF_PATH = pathlib.Path(__file__).resolve()

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "computed", "runs", "node_modules"}
TEXT_SUFFIXES = {
    ".py", ".ts", ".ps1", ".md", ".txt", ".json", ".xml", ".rcpx",
    ".emstate", ".yml", ".yaml", ".toml", ".example", ".cff",
}
TEXT_NAMES = {".gitignore", ".gitattributes", ".zenodo.json", "LICENSE", "Sources.xml.before-revert-2026-10-09"}

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("windows_drive_path", re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:\\|/(?!/))")),
    ("unix_user_path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")),
    ("eln_uri", re.compile(r"\beln://", re.IGNORECASE)),
    ("private_lan_ip", re.compile(r"\b10\.109\.")),
    ("private_plc_ip", re.compile(r"\b192\.168\.127\.(?!208\b)\d+")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("vendor_csmic", re.compile(r"CSMIC", re.IGNORECASE)),
    ("vendor_svac", re.compile(r"SVAC", re.IGNORECASE)),
    ("vendor_t500", re.compile(r"\bT500\b", re.IGNORECASE)),
    ("vendor_cluster_demo", re.compile(r"\bCluster_Demo\b", re.IGNORECASE)),
    ("vendor_penkov_517", re.compile(r"Penkov_517", re.IGNORECASE)),
    ("vendor_wsp", re.compile(r"\bWSP\b")),
    ("vendor_dc4001", re.compile(r"\bDC4001", re.IGNORECASE)),
    ("vendor_g3f", re.compile(r"\b(?:DCPulse)?G3F", re.IGNORECASE)),
    ("vendor_lesker", re.compile(r"\bLesker", re.IGNORECASE)),
    ("vendor_kcm91", re.compile(r"KCM91", re.IGNORECASE)),
    ("vendor_jyk", re.compile(r"JYK", re.IGNORECASE)),
    ("vendor_n6705c", re.compile(r"N6705C", re.IGNORECASE)),
    ("vendor_g500", re.compile(r"\b(?:Vac)?G500\b", re.IGNORECASE)),
    ("vendor_stepperdrive", re.compile(r"\bStepperDrive\b")),
    ("commercial_plc_policy", re.compile(r"PlcAccessPolicy|PlcSerialCollector")),
    (
        "commercial_licensing",
        re.compile(
            r"LicenseBootstrap|APS\.Licensing|APS\.Accounts\.(?:csproj|dll)|APS\.Production\.(?:csproj|dll)|"
            r"APS\.SystemControl\.Commercial|CommercialTests|EmulatorAdsTests|BehaviorValidation"
        ),
    ),
    ("vendor_buzheng", re.compile(r"Buzheng", re.IGNORECASE)),
    ("serial_com_port", re.compile(r"\bCOM\d+\b")),
]

XRAY_MIRROR_MISLABEL_RE = re.compile(r"\bx-?ray\s+(?:multilayer\s+)?mirror\b", re.IGNORECASE)


def iter_target_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == SELF_PATH:
            continue
        rel_parts = set(path.relative_to(REPO_ROOT).parts)
        if rel_parts & SKIP_DIRS:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return files


def scan_repository() -> list[str]:
    violations: list[str] = []
    for path in iter_target_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in FORBIDDEN_PATTERNS:
                match = pattern.search(line)
                if match:
                    violations.append(f"{rel}:{line_no}: [{name}] matched {match.group(0)!r}")
        if rel in {"suite/SC1-brief.md", "suite/sputter-suite.json", "harness/score_sc1.py"}:
            for line_no, line in enumerate(text.splitlines(), start=1):
                match = XRAY_MIRROR_MISLABEL_RE.search(line)
                if match and "not an x-ray mirror" not in line.lower():
                    violations.append(f"{rel}:{line_no}: [sc1_xray_mirror_mislabel] matched {match.group(0)!r}")
    return violations


def main() -> int:
    violations = scan_repository()
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        return 1
    print(f"OK: scanned {len(iter_target_files())} files with 0 portability/denylist violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
