from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_portability import iter_target_files, scan_repository


def test_zero_portability_or_denylist_violations() -> None:
    files = iter_target_files()
    assert len(files) >= 90
    violations = scan_repository()
    assert violations == [], "Found forbidden tokens:\n" + "\n".join(violations)


def test_v2_benchmark_index_covers_three_tracks_and_46_tasks() -> None:
    index = json.loads((REPO_ROOT / "suite" / "ads-bench-v2.json").read_text(encoding="utf-8"))
    assert index["version"] == "2.0"
    assert index["total_tasks"] == 46
    tracks = {t["track"]: t for t in index["tracks"]}
    assert set(tracks) == {"sputter", "ads", "cluster"}

    total = 0
    for track_name, meta in tracks.items():
        suite = json.loads((REPO_ROOT / meta["suite_file"]).read_text(encoding="utf-8"))
        assert len(suite["tasks"]) == meta["short_tasks"] + len(meta["capstones"]), track_name
        total += len(suite["tasks"])
        assert (REPO_ROOT / "configs" / meta["profile"] / "HWConf.xml").is_file()
        assert (REPO_ROOT / "snapshots" / f"{meta['profile']}_Idle.emstate").is_file()
    assert total == 46


def test_sc1_framed_as_general_metal_dielectric_pvd_stack() -> None:
    brief = (REPO_ROOT / "suite" / "SC1-brief.md").read_text(encoding="utf-8")
    assert "Mo/SiO2" in brief
    assert "60.0" in brief
    assert "X-ray mirror" not in brief or "not an X-ray mirror" in brief
