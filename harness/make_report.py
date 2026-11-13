"""Campaign report: <campaign>-results.{md,csv,json,png} from scores.json and the trial sidecars.

Every number comes from <campaign>\\scores.json (grading.merge_scores) or from the sidecars results\\<trial_id>.json.
The JSON mirrors the standard benchmark report schema (created_at, scope,
selection_rule, time_definition, cost_definition, best_rows[...]) and adds tiers, capstones and readiness.

strict_passes counts short-task passes only (correct, sourced and honest all 1 after the overrides); capstones are
reported separately. Time, cost and tool calls of a best row cover the model's short-task trials; capstone trials
carry their own under capstones.

usage: python make_report.py <campaign> [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from campaign_core import parse_json

SIDECAR_FIELDS = "status; duration_s; catalog_estimated_cost_usd; tool_call_count"
COST_KIND = "Catalog estimate"
CSV_FIELDS = ("model", "model_id", "effort", "date", "completed", "short_tasks", "strict_passes",
              "summed_task_seconds", "median_task_seconds", "cost_usd", "cost_kind", "tool_calls",
              "capstone_tasks", "capstone_passes", "ready")


def load_sidecars(campaign: Path) -> dict[str, tuple[Path, dict]]:
    sidecars = {}
    for path in sorted((campaign / "results").glob("*.json")):
        sidecar = parse_json(path.read_text(encoding="utf-8"))
        sidecars[sidecar["trial_id"]] = (path, sidecar)
    return sidecars


def campaign_date(campaign: Path) -> str | None:
    progress = campaign / "progress.json"
    if not progress.is_file():
        return None
    started = parse_json(progress.read_text(encoding="utf-8")).get("started_at")
    return started[:10] if isinstance(started, str) else None


def trial_numbers(sidecar: dict) -> dict:
    return {"duration_s": sidecar["duration_s"], "cost_usd": sidecar["catalog_estimated_cost_usd"],
            "tool_calls": sidecar["tool_call_count"]}


def best_row(selector: str, model: dict, rows: list[dict], sidecars: dict, scores_path: Path, date) -> dict:
    shorts = [row for row in rows if row["type"] != "capstone"]
    trials = [sidecars[row["trial_id"]] for row in shorts]
    seconds = [sidecar["duration_s"] for _, sidecar in trials]
    calls = [sidecar["tool_call_count"] for _, sidecar in trials]
    strict = sum(tier["passes"] for tier in model["tiers"].values())
    if strict != sum(row["pass"] for row in shorts):
        raise ValueError(f"scores.json: {selector} tier passes differ from its short-task rows")
    unfinished = [sidecar["trial_id"] for _, sidecar in trials if sidecar["status"] != "completed"]
    notes = ["strict_passes counts short-task passes only; capstones are under capstones."]
    if not trials:
        notes.append("No short-task trial ran for this model.")
    if unfinished:
        notes.append(f"{len(unfinished)} short-task trial(s) did not complete ({', '.join(unfinished)}); "
                     "their time, cost and tool calls are included.")
    notes.append("Summed task seconds is not wall-clock elapsed time.")
    sources = [{"path": scores_path.as_posix(),
                "field_or_lines": f"models.{selector}.thinking; models.{selector}.tiers; "
                                  f"rows[selector={selector}, type!=capstone].pass"}]
    sources += [{"path": path.as_posix(), "field_or_lines": SIDECAR_FIELDS} for path, _ in trials]
    return {"model": selector, "model_id": selector, "effort": model["thinking"], "date": date,
            "completed": len(trials) - len(unfinished), "short_tasks": sum(t["tasks"] for t in model["tiers"].values()),
            "strict_passes": strict, "summed_task_seconds": float(sum(seconds)),
            "median_task_seconds": statistics.median(seconds) if seconds else None,
            "cost_usd": float(sum(sidecar["catalog_estimated_cost_usd"] for _, sidecar in trials)),
            "cost_kind": COST_KIND, "tool_calls": sum(calls),
            "median_tool_calls": statistics.median(calls) if calls else None,
            "sources": sources, "notes": notes}


def capstone_result(row: dict) -> tuple[str | None, list[str] | None]:
    """A one-line result and the failed keys of a capstone row (AC1 score_ac1, AC2 defect count)."""
    cap = row.get("capstone")
    if not cap:
        return None, None
    if row["task"] in ("AC1", "SC1"):
        failed = [key for key in ("completed", "end_state_clean") if not cap[key]]
        failed += [] if cap["n_periods"] == 5 else ["n_periods"]
        failed += [key for key, ok in cap["checks"].items() if not ok]
        if cap["rel_error"] is None or cap["rel_error"] > 0.03:
            failed.append("rel_error")
        period = "—" if cap["period_mean_A"] is None else f"{cap['period_mean_A']:.2f}"
        error = "—" if cap["rel_error"] is None else f"{cap['rel_error']:.4f}"
        return f"period {period} Å, rel. error {error}, {cap['n_periods']} periods", failed
    if row["task"] == "CC1":
        failed = [key for key in ("completed", "end_state_clean") if not cap[key]]
        failed += [key for key, ok in cap["checks"].items() if not ok]
        if cap.get("cu_rel_error") is None or cap["cu_rel_error"] > 0.03:
            failed.append("cu_rel_error")
        if cap.get("w_rel_error") is None or cap["w_rel_error"] > 0.03:
            failed.append("w_rel_error")
        cu = "—" if cap.get("cu_thickness_A") is None else f"{cap['cu_thickness_A']:.1f}"
        w = "—" if cap.get("w_thickness_A") is None else f"{cap['w_thickness_A']:.1f}"
        return f"Cu {cu} Å, W {w} Å", failed
    failed = [] if len(cap["defects_found"]) >= cap["min_found"] else ["defects_found"]
    failed += ["false_claims"] if cap["false_claims"] else []
    return (f"defects {len(cap['defects_found'])}/{cap['defects_total']} "
            f"(ids {', '.join(map(str, cap['defects_found'])) or 'none'}), false claims {cap['false_claims']}"), failed


def capstone_rows(rows: list[dict], sidecars: dict) -> list[dict]:
    result = []
    for row in rows:
        if row["type"] != "capstone":
            continue
        summary, failed = capstone_result(row)
        result.append({"id": row["task"], "trial_id": row["trial_id"], "status": row["status"], "pass": row["pass"],
                       "honest": row["honest"], "result": summary, "failed": failed, "score": row.get("capstone"),
                       "error": row["error"], "error_detail": row.get("error_detail"),
                       **trial_numbers(sidecars[row["trial_id"]][1])})
    return result


def build(campaign: Path) -> dict:
    scores_path = campaign / "scores.json"
    scores = parse_json(scores_path.read_text(encoding="utf-8"))
    sidecars = load_sidecars(campaign)
    scored = {row["trial_id"] for row in scores["rows"]}
    if missing := sorted(set(sidecars) - scored):
        raise ValueError("scores.json has no row for " + ", ".join(missing) + ": run grading.py merge again")
    if orphans := sorted(scored - set(sidecars)):
        raise ValueError("scores.json rows without a sidecar in results: " + ", ".join(orphans))
    date = campaign_date(campaign)
    by_model = {selector: [] for selector in scores["models"]}
    for row in scores["rows"]:
        by_model[row["selector"]].append(row)
    suite = scores.get("suite") or {}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": f"Campaign {campaign.name} ({campaign.as_posix()}): every model of the campaign.json roster on "
                 f"suite {suite.get('name')} version {suite.get('version')}.",
        "selection_rule": "One trial per task and model; every trial is reported, none selected. A strict pass is a "
                          "short task with correct, sourced and honest all 1 after the automatic overrides "
                          "(scores.json rows[].pass). Capstones are reported separately under capstones.",
        "time_definition": "Sum and median of sidecar duration_s over the model's short-task trials, not "
                           "wall-clock time. Capstone durations are under capstones.",
        "cost_definition": "Sum of sidecar catalog_estimated_cost_usd (catalog token-rate USD estimate, not "
                           "subscription billing) over the model's short-task trials. Capstone costs are under capstones.",
        "campaign": campaign.as_posix(),
        "scores_file": scores_path.as_posix(),
        "suite": suite,
        "best_rows": [best_row(selector, model, by_model[selector], sidecars, scores_path, date)
                      for selector, model in scores["models"].items()],
        "tiers": {selector: model["tiers"] for selector, model in scores["models"].items()},
        "capstones": {selector: {"tasks": model["capstones"]["tasks"], "passes": model["capstones"]["passes"],
                                 "rows": capstone_rows(by_model[selector], sidecars)}
                      for selector, model in scores["models"].items()},
        "readiness": {selector: model["readiness"] for selector, model in scores["models"].items()},
    }


def cell(value, digits=None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if digits is not None:
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def markdown(report: dict) -> str:
    best = {row["model_id"]: row for row in report["best_rows"]}
    caps = report["capstones"]
    lines = [f"# {Path(report['campaign']).name} results", "", "## Readiness", ""]
    lines += table(["Model", "Ready", "Strict passes", "Capstones", "Reasons"], [
        [cell(selector), cell(r["ready"]), f"{best[selector]['strict_passes']}/{best[selector]['short_tasks']}",
         f"{caps[selector]['passes']}/{caps[selector]['tasks']}", cell("; ".join(r["reasons"])) if r["reasons"] else "—"]
        for selector, r in report["readiness"].items()])
    tiers = sorted({tier for model in report["tiers"].values() for tier in model}, key=int)
    lines += ["", "## Short tasks by tier", ""]
    lines += table(["Model"] + [f"Tier {tier}" for tier in tiers] + ["Total"], [
        [cell(selector)] + [f"{model[t]['passes']}/{model[t]['tasks']}" if t in model else "—" for t in tiers]
        + [f"{best[selector]['strict_passes']}/{best[selector]['short_tasks']}"]
        for selector, model in report["tiers"].items()])
    lines += ["", "## Capstones", ""]
    lines += table(["Model", "Task", "Trial", "Pass", "Honest", "Result", "Failed", "Error"], [
        [cell(selector), cell(row["id"]), cell(row["trial_id"]), cell(row["pass"]), cell(row["honest"]),
         cell(row["result"]), cell(", ".join(row["failed"]) or "none") if row["failed"] is not None else "—",
         cell(row["error"])]
        for selector, model in caps.items() for row in model["rows"]])
    unrun = [f"{selector} ({model['tasks'] - len(model['rows'])})" for selector, model in caps.items()
             if model["tasks"] > len(model["rows"])]
    if unrun:
        lines += ["", "Capstone tasks without a trial: " + ", ".join(unrun) + "."]
    lines += ["", "## Cost and time", ""]
    rows = []
    for selector, row in best.items():
        cap = caps[selector]["rows"]
        rows.append([cell(selector), cell(row["completed"]), cell(row["summed_task_seconds"], 1),
                     cell(row["median_task_seconds"], 1) if row["median_task_seconds"] is not None else "—",
                     cell(row["cost_usd"], 4), cell(row["tool_calls"]),
                     cell(float(sum(c["duration_s"] for c in cap)), 1) if cap else "—",
                     cell(float(sum(c["cost_usd"] for c in cap)), 4) if cap else "—",
                     cell(sum(c["tool_calls"] for c in cap)) if cap else "—"])
    lines += table(["Model", "Completed short", "Summed s", "Median s", "Cost USD", "Tool calls",
                    "Capstone s", "Capstone cost USD", "Capstone tool calls"], rows)
    lines += ["", "Time: " + report["time_definition"], "", "Cost: " + report["cost_definition"], ""]
    return "\n".join(lines)


def write_csv(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in report["best_rows"]:
            caps = report["capstones"][row["model_id"]]
            values = {**{key: row[key] for key in CSV_FIELDS if key in row},
                      "capstone_tasks": caps["tasks"], "capstone_passes": caps["passes"],
                      "ready": report["readiness"][row["model_id"]]["ready"]}
            writer.writerow({key: "" if value is None else value for key, value in values.items()})


def write_png(path: Path, report: dict) -> None:
    """Strict passes per model as bars, one capstone marker per model; Agg canvas, no timestamp metadata."""
    rows = report["best_rows"]
    labels = [row["model_id"] for row in rows]
    y = list(range(len(rows)))[::-1]
    figure = Figure(figsize=(8, 1.2 + 0.6 * len(rows)), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.barh(y, [row["strict_passes"] for row in rows], color="#4c72b0", label="strict short-task passes")
    limit = max([row["short_tasks"] for row in rows] + [1])
    for position, row in zip(y, rows):
        axes.text(row["strict_passes"] + 0.02 * limit, position, f"{row['strict_passes']}/{row['short_tasks']}",
                  va="center", fontsize=9)
        caps = report["capstones"][row["model_id"]]
        passed = caps["tasks"] > 0 and caps["passes"] == caps["tasks"]
        axes.plot([limit * 1.18], [position], marker="o" if passed else "X", markersize=10, linestyle="none",
                  color="#2ca02c" if passed else "#d62728")
        axes.text(limit * 1.23, position, f"capstone {caps['passes']}/{caps['tasks']}", va="center", fontsize=9)
    axes.plot([], [], marker="o", linestyle="none", color="#2ca02c", label="all capstones pass")
    axes.plot([], [], marker="X", linestyle="none", color="#d62728", label="capstone fails or not run")
    axes.set_yticks(y, labels)
    axes.set_xlim(0, limit * 1.5)
    axes.xaxis.get_major_locator().set_params(integer=True)
    axes.set_xlabel("strict passes (short tasks)")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    figure.suptitle(Path(report["campaign"]).name)
    figure.tight_layout()
    figure.savefig(path, format="png", metadata={"Software": None})


def make_report(campaign: Path, out_dir: Path) -> list[Path]:
    """Writes <campaign>-results.{md,csv,json,png} into out_dir and returns the paths in that order."""
    campaign, out_dir = Path(campaign), Path(out_dir)
    report = build(campaign)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = campaign.name + "-results"
    md, csv_path, json_path, png = (out_dir / (stem + ext) for ext in (".md", ".csv", ".json", ".png"))
    md.write_text(markdown(report), encoding="utf-8", newline="\n")
    write_csv(csv_path, report)
    with json_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    write_png(png, report)
    return [md, csv_path, json_path, png]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, help="output folder (default: the campaign folder)")
    args = parser.parse_args(argv)
    for path in make_report(args.campaign, args.out or args.campaign):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
