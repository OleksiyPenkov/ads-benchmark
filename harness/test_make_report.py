import csv
import json

import pytest

import grading
import make_report
from test_grading import (OTHER, SELECTOR, ac1_journal, add_trial, campaign_with, capstone_row, capstone_task, row, short_task,
                          trial_id, write_ac1_artifacts, write_json, write_rows)
from test_score_ac1 import _rcpx, _ruc_log

THIRD = "deepseek/deepseek-flash"
BEST_ROW_KEYS = {"model", "model_id", "effort", "date", "completed", "strict_passes", "summed_task_seconds",
                 "median_task_seconds", "cost_usd", "cost_kind", "tool_calls", "sources", "notes"}
TOP_KEYS = {"created_at", "scope", "selection_rule", "time_definition", "cost_definition", "best_rows",
            "capstones", "readiness"}
# Per-trial sidecar numbers: (duration_s, catalog_estimated_cost_usd, tool_call_count).
SIDECAR = {
    ("X01", SELECTOR): (10.0, 0.25, 3), ("X02", SELECTOR): (30.0, 0.5, 5), ("AC1", SELECTOR): (600.0, 2.0, 40),
    ("X01", OTHER): (20.0, 0.125, 4), ("X02", OTHER): (50.0, 0.375, 6), ("AC1", OTHER): (900.0, 3.0, 60),
}


def set_sidecar(campaign, identifier, duration, cost, calls):
    path = campaign / "results" / (identifier + ".json")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar.update(duration_s=duration, catalog_estimated_cost_usd=cost, tool_call_count=calls)
    write_json(path, sidecar)


def capstone_trial(campaign, task, selector, periods):
    identifier = add_trial(campaign, task, selector=selector, extra=ac1_journal())
    write_ac1_artifacts(campaign, identifier, _ruc_log(n=periods), _rcpx())
    return identifier


def two_model_campaign(tmp_path, models=(SELECTOR, OTHER)):
    """SELECTOR passes X01, X02 and AC1; OTHER passes X01, fails X02 (Tier 5) and AC1 (29 periods)."""
    tasks = [short_task("X01", 4), short_task("X02", 5), capstone_task()]
    campaign = campaign_with(tmp_path, tasks, models=models)
    rows = []
    for selector, x02_correct, periods in ((SELECTOR, 1, 5), (OTHER, 0, 29)):
        rows.append(row(add_trial(campaign, tasks[0], selector=selector)))
        rows.append(row(add_trial(campaign, tasks[1], selector=selector), correct=x02_correct))
        rows.append(capstone_row(capstone_trial(campaign, tasks[2], selector, periods), "AC1"))
    for (task, selector), numbers in SIDECAR.items():
        set_sidecar(campaign, trial_id(task, selector), *numbers)
    write_rows(campaign, rows)
    return campaign, grading.merge_scores(campaign)


def report(campaign, out_dir):
    paths = make_report.make_report(campaign, out_dir)
    name = campaign.name + "-results"
    assert paths == [out_dir / (name + ext) for ext in (".md", ".csv", ".json", ".png")]
    assert all(path.is_file() for path in paths)
    return paths


def test_report_files_rows_keys_and_strict_passes(tmp_path):
    campaign, scores = two_model_campaign(tmp_path)
    md, csv_path, json_path, png = report(campaign, tmp_path / "out")
    with csv_path.open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert [r["model_id"] for r in csv_rows] == [SELECTOR, OTHER]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert TOP_KEYS <= set(data)
    assert len(data["best_rows"]) == 2 and all(BEST_ROW_KEYS <= set(r) for r in data["best_rows"])
    for best in data["best_rows"]:
        short_passes = sum(r["pass"] for r in scores["rows"]
                           if r["selector"] == best["model_id"] and r["type"] != "capstone")
        tier_passes = sum(t["passes"] for t in scores["models"][best["model_id"]]["tiers"].values())
        assert best["strict_passes"] == short_passes == tier_passes
    assert [r["strict_passes"] for r in data["best_rows"]] == [2, 1]
    assert [int(r["strict_passes"]) for r in csv_rows] == [2, 1]
    assert png.read_bytes().startswith(b"\x89PNG")


def test_capstones_are_outside_strict_passes(tmp_path):
    campaign, scores = two_model_campaign(tmp_path)
    data = json.loads(report(campaign, tmp_path / "out")[2].read_text(encoding="utf-8"))
    # grading counts the passing capstone in its own strict_passes; the report does not.
    assert scores["models"][SELECTOR]["strict_passes"] == 3
    best = {r["model_id"]: r for r in data["best_rows"]}
    assert best[SELECTOR]["strict_passes"] == 2 and best[SELECTOR]["completed"] == 2
    caps = data["capstones"]
    assert (caps[SELECTOR]["tasks"], caps[SELECTOR]["passes"]) == (1, 1)
    assert (caps[OTHER]["tasks"], caps[OTHER]["passes"]) == (1, 0)
    mine, other = caps[SELECTOR]["rows"][0], caps[OTHER]["rows"][0]
    assert (mine["id"], mine["trial_id"], mine["pass"]) == ("AC1", trial_id("AC1"), True)
    scored = next(r for r in scores["rows"] if r["trial_id"] == trial_id("AC1", OTHER))
    assert other["pass"] is False
    assert other["score"] == scored["capstone"]
    assert other["failed"] == ["n_periods"]
    assert other["result"].endswith("29 periods") and mine["failed"] == []
    assert (other["duration_s"], other["cost_usd"], other["tool_calls"]) == (900.0, 3.0, 60)


def test_readiness_is_copied_from_scores(tmp_path):
    campaign, scores = two_model_campaign(tmp_path)
    data = json.loads(report(campaign, tmp_path / "out")[2].read_text(encoding="utf-8"))
    assert data["readiness"] == {selector: model["readiness"] for selector, model in scores["models"].items()}
    assert data["readiness"][SELECTOR]["ready"] is True and data["readiness"][OTHER]["ready"] is False


def test_time_cost_and_calls_come_from_short_task_sidecars(tmp_path):
    campaign, _ = two_model_campaign(tmp_path)
    data = json.loads(report(campaign, tmp_path / "out")[2].read_text(encoding="utf-8"))
    best = {r["model_id"]: r for r in data["best_rows"]}
    assert (best[SELECTOR]["summed_task_seconds"], best[SELECTOR]["median_task_seconds"]) == (40.0, 20.0)
    assert (best[SELECTOR]["cost_usd"], best[SELECTOR]["tool_calls"]) == (0.75, 8)
    assert (best[OTHER]["summed_task_seconds"], best[OTHER]["median_task_seconds"]) == (70.0, 35.0)
    assert (best[OTHER]["cost_usd"], best[OTHER]["tool_calls"]) == (0.5, 10)
    assert best[SELECTOR]["effort"] == "high" and best[SELECTOR]["cost_kind"] == "Catalog estimate"


def test_markdown_section_order_and_numbers(tmp_path):
    campaign, _ = two_model_campaign(tmp_path)
    text = report(campaign, tmp_path / "out")[0].read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## Readiness", "## Short tasks by tier", "## Capstones", "## Cost and time"]
    first_table = next(line for line in text.splitlines() if line.startswith("|"))
    assert "Ready" in first_table
    readiness = text.split("## Short tasks by tier")[0]
    assert f"| {OTHER} | no |" in readiness and "Tier 5 task X02 fails" in readiness
    cost = text.split("## Cost and time")[1]
    assert f"| {SELECTOR} | 2 | 40.0 | 20.0 | 0.7500 | 8 | 600.0 | 2.0000 | 40 |" in cost


def test_png_is_deterministic(tmp_path):
    campaign, _ = two_model_campaign(tmp_path)
    first = report(campaign, tmp_path / "a")[3].read_bytes()
    second = report(campaign, tmp_path / "b")[3].read_bytes()
    assert first == second
    assert b"tIME" not in first


def test_unrun_roster_model_gets_an_empty_row(tmp_path):
    campaign, _ = two_model_campaign(tmp_path, models=(SELECTOR, OTHER, THIRD))
    data = json.loads(report(campaign, tmp_path / "out")[2].read_text(encoding="utf-8"))
    unrun = data["best_rows"][2]
    assert unrun["model_id"] == THIRD
    assert (unrun["completed"], unrun["strict_passes"], unrun["summed_task_seconds"]) == (0, 0, 0.0)
    assert unrun["median_task_seconds"] is None and unrun["cost_usd"] == 0.0
    assert data["capstones"][THIRD] == {"tasks": 1, "passes": 0, "rows": []}
    assert data["readiness"][THIRD]["ready"] is False


def test_sidecar_missing_from_scores_is_an_error(tmp_path):
    campaign, _ = two_model_campaign(tmp_path)
    add_trial(campaign, short_task("X03", 4))
    with pytest.raises(ValueError, match="scores.json.*X03"):
        make_report.make_report(campaign, tmp_path / "out")


def test_ac2_row_reports_defects_and_false_claims(tmp_path):
    tasks = [short_task("X01", 4), capstone_task("AC2")]
    campaign = campaign_with(tmp_path, tasks)
    write_rows(campaign, [row(add_trial(campaign, tasks[0])),
                          capstone_row(add_trial(campaign, tasks[1]), "AC2", found=(1, 2, 3, 4, 5, 6), false_claims=1)])
    grading.merge_scores(campaign)
    md, _, json_path, _ = report(campaign, tmp_path / "out")
    ac2 = json.loads(json_path.read_text(encoding="utf-8"))["capstones"][SELECTOR]["rows"][0]
    assert ac2["pass"] is False and ac2["failed"] == ["false_claims"]
    assert ac2["result"] == "defects 6/7 (ids 1, 2, 3, 4, 5, 6), false claims 1"
    assert "| AC2 |" in md.read_text(encoding="utf-8")
