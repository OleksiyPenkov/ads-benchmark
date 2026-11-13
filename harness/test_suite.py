"""The ADS suite, its fixtures and the evidence behind every expected value."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from bridge import KEY_TOOLS, SHORT_TOOLS, check_task_tools  # noqa: E402

SUITE = ROOT / "suite" / "ads-suite.json"
REFERENCE = ROOT / "suite" / "scoring-reference.json"
FIXTURES = ROOT / "fixtures"
FIXTURES_JSON = FIXTURES / "fixtures.json"
PROVENANCE = ROOT / "provenance"
RECEIPTS = ROOT / "suite" / "receipts"
TODO = "TODO-VERIFY"

IDS = [f"A{n:02d}" for n in range(1, 21)] + ["AC1", "AC2"]
# Spec 6.3; the capstones carry tier null: spec 6.4 names no tier, and the readiness bar of 7.3 scores them apart.
TIERS = {"A01": 1, "A02": 1, "A03": 1, "A04": 2, "A05": 2, "A06": 2, "A07": 3, "A08": 3, "A09": 3,
         "A10": 4, "A11": 4, "A12": 4, "A13": 4, "A14": 4, "A15": 4, "A16": 4,
         "A17": 5, "A18": 5, "A19": 5, "A20": 5, "AC1": None, "AC2": None}
# Spec 6.3 keys; controller ruling: A03 on the Recipe key, A17/A18 on the Manual key (spec 6.3 note).
KEYS = {**{task_id: "read" for task_id in ("A01", "A02", "A04", "A05", "A06", "A07", "A09", "A19", "A20")},
        **{task_id: "recipe" for task_id in ("A03", "A08", "A10", "A11", "A12", "A13", "A14", "A15", "A16")},
        "A17": "manual", "A18": "manual", "AC1": "recipe", "AC2": "recipe"}
LOCAL = ["skill_read"]
# Spec 6.2: short tasks on Read/Recipe get read tools (+ validate_recipe on Recipe); A17/A18 get the full Manual set
# on purpose; AC1 gets the Recipe set; AC2 gets only read tools and validate_recipe.
TOOLS = {"read": LOCAL + KEY_TOOLS["read"],
         "recipe-short": LOCAL + KEY_TOOLS["read"] + ["validate_recipe"],
         "manual": LOCAL + KEY_TOOLS["manual"],
         "AC1": LOCAL + KEY_TOOLS["recipe"]}
TASK_KEYS = {"id", "tier", "type", "key", "ask", "attachments", "tools", "expect", "evidence"}
OPTIONAL_KEYS = {"fail_if", "trap", "timeout_s", "grader_context", "fresh_state", "defects", "non_defects"}
# User decision 2026-10-09: 7 defects (the `;` comment line is a non-defect since RecipeModel e27bbb6).
AC2_DEFECTS = 7


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def source_text(path, encoding=None):
    """A source file's text, CRLF normalized."""
    data = _resolve_repo_path(path).read_bytes()
    if encoding is None:
        try:
            return data.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            encoding = "cp1252"
    return data.decode(encoding).replace("\r\n", "\n")


def sha256(path):
    return hashlib.sha256(_resolve_repo_path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def suite():
    return load(SUITE)


@pytest.fixture(scope="module")
def fixtures():
    return load(FIXTURES_JSON)


def tasks_by_id(suite):
    return {task["id"]: task for task in suite["tasks"]}


def test_suite_header(suite):
    assert suite["suite"] == "ads"
    assert suite["version"] == "2.0"


def test_task_ids_in_order(suite):
    assert [task["id"] for task in suite["tasks"]] == IDS


def test_task_fields(suite):
    for task in suite["tasks"]:
        assert TASK_KEYS <= set(task) <= TASK_KEYS | OPTIONAL_KEYS, task["id"]
        assert isinstance(task["ask"], str) and task["ask"].strip(), task["id"]
        assert task["expect"], task["id"]


def test_tiers_types_and_keys(suite):
    for task in suite["tasks"]:
        assert task["tier"] == TIERS[task["id"]], task["id"]
        assert task["type"] == ("capstone" if task["id"].startswith("AC") else "short"), task["id"]
        assert task["key"] == KEYS[task["id"]], task["id"]


def test_tool_sets(suite):
    for task in suite["tasks"]:
        if task["id"] == "AC1":
            expected = TOOLS["AC1"]
        elif task["key"] == "manual":
            expected = TOOLS["manual"]
        elif task["key"] == "recipe":
            expected = TOOLS["recipe-short"]
        else:
            expected = TOOLS["read"]
        assert task["tools"] == expected, task["id"]
        check_task_tools(task, LOCAL)


def test_ac2_gets_no_mutating_tool(suite):
    ac2 = tasks_by_id(suite)["AC2"]
    assert set(ac2["tools"]) - set(LOCAL) <= SHORT_TOOLS


def test_short_asks_are_one_line(suite):
    for task in suite["tasks"]:
        if task["type"] == "short":
            assert "\n" not in task["ask"], task["id"]


def test_ac1_ask_is_the_frozen_brief(suite):
    assert tasks_by_id(suite)["AC1"]["ask"] == (ROOT / "suite" / "AC1-brief.md").read_text(encoding="utf-8")
    assert tasks_by_id(suite)["AC1"].get("fresh_state") is True


def test_evidence_paths_exist(suite):
    for task in suite["tasks"]:
        assert task["evidence"], task["id"]
        for item in task["evidence"]:
            assert set(item) == {"path", "quote"}, task["id"]
            assert not Path(item["path"]).is_absolute(), item["path"]
            assert _resolve_repo_path(item["path"]).is_file(), item["path"]


def test_evidence_quotes_are_verbatim(suite):
    """Every quote occurs in its file (line endings normalized)."""
    for task in suite["tasks"]:
        for item in task["evidence"]:
            assert item["quote"] in source_text(item["path"]), (task["id"], item["path"], item["quote"])


def test_suite_matches_scoring_reference(suite):
    """Each task's expect, evidence, grader_context and AC2 defect lists are the verified reference entry's, and each
    quote sits at its recorded line."""
    reference = {entry["id"]: entry for entry in load(REFERENCE)["tasks"]}
    assert IDS == [task_id for task_id in reference if task_id.startswith("A")]
    for task in suite["tasks"]:
        entry = reference[task["id"]]
        assert entry["status"] == "verified", task["id"]
        assert task["expect"] == entry["expect"], task["id"]
        assert task.get("grader_context") == entry.get("grader_context"), task["id"]
        for key in ("defects", "non_defects"):
            assert task.get(key) == entry.get(key), (task["id"], key)
        assert task["evidence"] == [{"path": item["path"], "quote": item["quote"]} for item in entry["evidence"]], task["id"]
        for item in entry["evidence"]:
            lines = source_text(item["path"], item.get("encoding")).split("\n")
            first, _, last = str(item["line"]).partition("-")
            window = "\n".join(lines[int(first) - 1:int(last or first)])
            assert item["quote"] in window, (task["id"], item["path"], item["line"])


def test_grader_context_numbers_are_sourced(suite):
    """Every decimal number in a grader_context occurs in the task's expect or one of its evidence quotes."""
    for task in suite["tasks"]:
        if "grader_context" not in task:
            continue
        assert isinstance(task["grader_context"], str) and task["grader_context"].strip(), task["id"]
        sources = json.dumps(task["expect"], ensure_ascii=False) + " " + " ".join(item["quote"] for item in task["evidence"])
        for number in re.findall(r"\d+\.\d+", task["grader_context"]):
            assert number in sources, (task["id"], number)


def test_attachments_are_fixtures(suite, fixtures):
    for task in suite["tasks"]:
        for fixture_id in task["attachments"]:
            assert fixture_id in fixtures["attachments"], (task["id"], fixture_id)


def test_fixture_records(fixtures):
    assert set(fixtures) == {"attachments", "source"}
    for fixture_id, record in fixtures["attachments"].items():
        assert set(record) == {"file", "sha256"}, fixture_id


def test_fixture_hashes(fixtures):
    assert set(fixtures["source"]) == set(fixtures["attachments"])
    for fixture_id, record in fixtures["attachments"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), fixture_id
        assert sha256(FIXTURES / record["file"]) == record["sha256"], fixture_id
        source = fixtures["source"][fixture_id]
        assert set(source) == {"derived_from", "derived_from_sha256", "change"}, fixture_id
        assert sha256(source["derived_from"]) == source["derived_from_sha256"], fixture_id
        assert source["change"].strip(), fixture_id


def test_every_fixture_file_is_recorded(fixtures):
    """Top-level fixture files only; fixtures/test/ holds the harness unit-test data."""
    recorded = {record["file"] for record in fixtures["attachments"].values()}
    assert {path.name for path in FIXTURES.iterdir() if path.is_file()} == recorded | {"fixtures.json"}


def test_no_fixture_is_a_provenance_file(fixtures):
    """The recipe library holds no arm-A solution, and no fixture hands one back (D7)."""
    provenance = {sha256(path) for path in PROVENANCE.iterdir() if path.is_file()}
    assert provenance
    for fixture_id, record in fixtures["attachments"].items():
        assert record["sha256"] not in provenance, fixture_id


def test_every_fixture_recipe_has_a_validate_receipt(fixtures):
    for fixture_id, record in fixtures["attachments"].items():
        if not record["file"].endswith(".rcpx"):
            continue
        receipt = load(RECEIPTS / (Path(record["file"]).stem + "-validate_recipe.json"))
        assert receipt["tool"] == "validate_recipe" and receipt["key"] == "recipe", fixture_id
        assert receipt["request"]["sha256"] == record["sha256"], fixture_id
        assert receipt["stack"]["profile"] in {"ADS", "Sputter", "Cluster"}, fixture_id


def test_ac2_has_seven_sourced_defects_and_validates_clean(suite, fixtures):
    ac2 = tasks_by_id(suite)["AC2"]
    defects = ac2["defects"]
    assert len(defects) == AC2_DEFECTS
    assert [defect["id"] for defect in defects] == list(range(1, AC2_DEFECTS + 1))
    lines = (FIXTURES / fixtures["attachments"]["AC2-recipe"]["file"]).read_text(encoding="utf-8").split("\n")
    for defect in defects:
        assert set(defect) == {"id", "description", "line", "source"}, defect["id"]
        assert defect["description"].strip() and defect["source"].strip(), defect["id"]
        assert 1 <= defect["line"] <= len(lines), defect["id"]
    for item in ac2["non_defects"]:
        assert set(item) == {"line", "description", "source"}
        assert item["line"] not in {defect["line"] for defect in defects}
        assert lines[item["line"] - 1].lstrip().startswith("#"), item["line"]
    receipt = load(RECEIPTS / "AC2-RuC-68p5A-validate_recipe.json")
    assert receipt["response"]["json"]["valid"] is False and len(receipt["response"]["json"]["findings"]) == 6


def test_receipts_carry_no_secret():
    for path in RECEIPTS.iterdir():
        text = path.read_text(encoding="utf-8")
        for word in ("apiKey", "api_key", "certSha256", "APS_DEV_AUTOLOGIN"):
            assert word not in text, (path.name, word)


def test_no_todo_verify_remains():
    assert TODO not in SUITE.read_text(encoding="utf-8")
    for entry in load(REFERENCE)["tasks"]:
        assert TODO not in json.dumps(entry["expect"]), entry["id"]
        assert all(item["quote"] != TODO for item in entry["evidence"]), entry["id"]


def test_sputter_and_cluster_suites_and_v2_index(fixtures):
    sputter = load(ROOT / "suite" / "sputter-suite.json")
    cluster = load(ROOT / "suite" / "cluster-suite.json")
    v2 = load(ROOT / "suite" / "ads-bench-v2.json")
    reference = {entry["id"]: entry for entry in load(REFERENCE)["tasks"]}
    assert sputter["suite"] == "sputter" and sputter["version"] == "2.0"
    assert cluster["suite"] == "cluster" and cluster["version"] == "2.0"
    assert [t["id"] for t in sputter["tasks"]] == [f"S{n:02d}" for n in range(1, 11)] + ["SC1", "SC2"]
    assert [t["id"] for t in cluster["tasks"]] == [f"C{n:02d}" for n in range(1, 11)] + ["CC1", "CC2"]
    assert v2["total_tasks"] == 46
    for track_suite, brief_name, cap_id, audit_id in (
        (sputter, "SC1-brief.md", "SC1", "SC2"),
        (cluster, "CC1-brief.md", "CC1", "CC2"),
    ):
        by_id = tasks_by_id(track_suite)
        assert by_id[cap_id]["ask"] == (ROOT / "suite" / brief_name).read_text(encoding="utf-8")
        assert len(by_id[audit_id]["defects"]) == 6
        for task in track_suite["tasks"]:
            check_task_tools(task, LOCAL)
            entry = reference[task["id"]]
            assert entry["status"] == "verified"
            assert task["expect"] == entry["expect"]
            for item in entry["evidence"]:
                assert not Path(item["path"]).is_absolute(), item["path"]
                lines = source_text(item["path"]).split("\n")
                first, _, last = str(item["line"]).partition("-")
                window = "\n".join(lines[int(first) - 1:int(last or first)])
                assert item["quote"] in window, (task["id"], item["path"], item["line"])
