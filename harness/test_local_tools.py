import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_manifest import CAPSTONE, DECLARATIONS, make_campaign, sha

_DRV = Path(__file__).resolve().drive or "X:"
ADS_SOURCE = f"{_DRV}/opt/lab-skills/skills/aps-operator/"
VACUUM_SOURCE = f"{_DRV}/opt/control/skills/vacuum-operations/"
ROOTS = [(ADS_SOURCE, "ads-operator"), (VACUUM_SOURCE, "vacuum-operations")]
ADS_SKILL = "# ADS operator\nRu at 0.4711 \u00c5/s.\n"
VACUUM_SKILL = "# Vacuum operations\nPump down first.\n"
NOTE = "Keep the chamber below 1e-6 mbar.\n"


def snapshot(tmp_path):
    root = tmp_path / "campaign" / "inputs" / "skill"
    (root / "ads-operator" / "notes").mkdir(parents=True)
    (root / "vacuum-operations").mkdir(parents=True)
    (root / "ads-operator" / "SKILL.md").write_text(ADS_SKILL, encoding="utf-8")
    (root / "ads-operator" / "notes" / "pressure.md").write_text(NOTE, encoding="utf-8")
    (root / "vacuum-operations" / "SKILL.md").write_text(VACUUM_SKILL, encoding="utf-8")
    (tmp_path / "campaign" / "campaign.json").write_text("{}", encoding="utf-8")
    return root


def tools(root):
    from local_tools import LocalTools
    return LocalTools(root, ROOTS)


def read(local, path):
    return asyncio.run(local.call("skill_read", {"path": path}))


def text(result):
    assert [block["type"] for block in result["content"]] == ["text"]
    return result["content"][0]["text"]


def test_relative_read(tmp_path):
    local = tools(snapshot(tmp_path))
    for path, expected in (("ads-operator/SKILL.md", ADS_SKILL), ("vacuum-operations\\SKILL.md", VACUUM_SKILL),
                           ("ads-operator/notes/pressure.md", NOTE)):
        result = read(local, path)
        assert result["isError"] is False, path
        assert text(result) == expected


def test_absolute_source_path_mapped(tmp_path):
    local = tools(snapshot(tmp_path))
    for path, expected in ((ADS_SOURCE + "SKILL.md", ADS_SKILL),
                           (ADS_SOURCE + "notes/pressure.md", NOTE),
                           (f"{_DRV}\\opt\\lab-skills\\skills\\aps-operator\\notes\\pressure.md", NOTE),
                           (VACUUM_SOURCE + "SKILL.md", VACUUM_SKILL),
                           (f"{_DRV}\\opt\\control\\skills\\vacuum-operations\\SKILL.md", VACUUM_SKILL)):
        result = read(local, path)
        assert result["isError"] is False, path
        assert text(result) == expected


def test_case_insensitive_drive_and_prefix(tmp_path):
    local = tools(snapshot(tmp_path))
    for path, expected in ((f"{_DRV.lower()}/opt/lab-skills/SKILLS/APS-OPERATOR/notes/pressure.md", NOTE),
                           (f"{_DRV}\\opt\\lab-skills\\skills/aps-operator\\SKILL.md", ADS_SKILL),
                           (f"{_DRV.lower()}/opt/control/SKILLS/Vacuum-Operations/SKILL.md", VACUUM_SKILL)):
        result = read(local, path)
        assert result["isError"] is False, path
        assert text(result) == expected


def test_escape_refused(tmp_path):
    local = tools(snapshot(tmp_path))
    for path in ("..\\..\\campaign.json", "../../campaign.json", f"{_DRV}/Windows/win.ini", "\\\\server\\share\\x",
                 "//server/share/x", ADS_SOURCE + "../ads-operator-v0/x.md", "/etc/passwd", "C:x",
                 # An absolute path stays inside the folder its own root maps to.
                 ADS_SOURCE + "../vacuum-operations/SKILL.md", VACUUM_SOURCE + "../ads-operator/SKILL.md",
                 f"{_DRV}/opt/lab-skills/skills/aps-operator-evil/SKILL.md",
                 f"{_DRV}/opt/control/skills/SKILL.md"):
        result = read(local, path)
        assert result["isError"] is True, path
        assert text(result) == "path outside the skill snapshot"


def test_missing_file(tmp_path):
    local = tools(snapshot(tmp_path))
    for path in ("absent.md", "SKILL.md", ADS_SOURCE + "absent.md", VACUUM_SOURCE + "notes/absent.md",
                 "ads-operator/notes", "vacuum-operations"):
        result = read(local, path)
        assert result["isError"] is True, path
        assert text(result) == "file not found"


def link_directory(link, target):
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return
    except (ImportError, AttributeError, OSError):
        pass
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create a junction or a directory symlink here")


def test_junction_escape_refused(tmp_path):
    root = snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link_directory(root / "ads-operator" / "link", outside)
    assert (root / "ads-operator" / "link" / "secret.txt").read_text(encoding="utf-8") == "secret"
    local = tools(root)
    for path in ("ads-operator/link/secret.txt", ADS_SOURCE + "link/secret.txt"):
        result = read(local, path)
        assert result["isError"] is True, path
        assert text(result) == "path outside the skill snapshot"


def test_cross_root_junction_refused(tmp_path):
    root = snapshot(tmp_path)
    import shutil
    shutil.rmtree(root / "ads-operator")
    link_directory(root / "ads-operator", root / "vacuum-operations")
    assert (root / "ads-operator" / "SKILL.md").read_text(encoding="utf-8") == VACUUM_SKILL
    local = tools(root)
    for path in (ADS_SOURCE + "SKILL.md", "ads-operator/SKILL.md", "ads-operator\\SKILL.md"):
        result = read(local, path)
        assert result["isError"] is True, path
        assert text(result) == "path outside the skill snapshot"
    result = read(local, VACUUM_SOURCE + "SKILL.md")
    assert result["isError"] is False
    assert text(result) == VACUUM_SKILL


def test_nested_link_into_other_skill_refused(tmp_path):
    root = snapshot(tmp_path)
    link_directory(root / "ads-operator" / "other", root / "vacuum-operations")
    local = tools(root)
    for path in (ADS_SOURCE + "other/SKILL.md", "ads-operator/other/SKILL.md"):
        result = read(local, path)
        assert result["isError"] is True, path
        assert text(result) == "path outside the skill snapshot"


def test_source_roots_constant_names_the_real_skills():
    from local_tools import SOURCE_ROOTS
    from local_tools import DEFAULT_SKILL_DIR
    assert [tuple(pair) for pair in SOURCE_ROOTS] == [(DEFAULT_SKILL_DIR, "aps-operator")]
    for source, _ in SOURCE_ROOTS:
        real = Path(source) / "SKILL.md"
        if not real.parent.exists():
            pytest.skip("skill source folder is not on this machine: " + source)
        assert real.is_file()


def test_only_skill_read_is_declared(tmp_path):
    local = tools(snapshot(tmp_path))
    assert [d["name"] for d in local.declarations()] == ["skill_read"]
    with pytest.raises(ValueError, match="validate_preset"):
        asyncio.run(local.call("validate_preset", {}))


def test_build_system_prompt():
    from campaign_worker import build_system_prompt
    assert build_system_prompt("Base.\n", "# ADS body\n", "# Vacuum body\n") == (
        "Base.\n\n\n# Skill: ads-operator\n\n# ADS body\n\n\n# Skill: vacuum-operations\n\n# Vacuum body\n")


def test_local_declaration_drift(tmp_path):
    from campaign_worker import TrialFailure, load_trial
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    drifted = [*DECLARATIONS[:2], {**DECLARATIONS[2], "description": "Read any file."}]
    campaign = make_campaign(tmp_path, declarations=drifted)
    with pytest.raises(TrialFailure, match="local_tool_declaration_drift"):
        load_trial(campaign, CAPSTONE, model, sha(campaign / "campaign.json"))
    # A local tool the harness does not provide (XRR's validate_preset) is drift too.
    extra = [*DECLARATIONS, {"name": "validate_preset", "description": "Run a script.", "loadMode": "essential",
                             "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]
    campaign = make_campaign(tmp_path / "extra", declarations=extra,
                             tools=[{"name": d["name"], "cls": c} for d, c in
                                    zip(extra, ("read", "write", "local", "local"))])
    with pytest.raises(TrialFailure, match="local_tool_declaration_drift"):
        load_trial(campaign, CAPSTONE, model, sha(campaign / "campaign.json"))


def test_trial_dispatches_local_tool_without_session(tmp_path):
    from campaign_worker import Trial, load_trial
    campaign = make_campaign(tmp_path)
    skill = campaign / "inputs" / "skill"
    skill.mkdir()
    from local_tools import DEFAULT_SKILL_DIR
    (skill / "aps-operator").mkdir()
    (skill / "aps-operator" / "SKILL.md").write_text(ADS_SKILL, encoding="utf-8")
    model = {"selector": "anthropic/claude-opus-5-5", "thinking": "high"}
    manifest, _, guard, _, local = load_trial(campaign, CAPSTONE, model, sha(campaign / "campaign.json"))
    events, sent = [], []
    trial = Trial(CAPSTONE, model, guard, SimpleNamespace(emit=events.append), manifest.tool_prefix, local)

    class NoSession:
        async def call_tool(self, *args):
            raise AssertionError("local tool reached the MCP session")

    async def send(frame):
        sent.append(frame)

    trial.session, trial.rpc = NoSession(), SimpleNamespace(send=send)
    arguments = {"path": DEFAULT_SKILL_DIR + "SKILL.md"}
    message = {"type": "message_end", "messageId": "m1", "message": {
        "role": "assistant", "provider": "anthropic", "model": "claude-opus-5-5", "stopReason": "toolUse",
        "usage": {"cost": {"total": 0.01}},
        "content": [{"type": "toolCall", "id": "c1", "name": "aps_skill_read", "arguments": arguments}]}}
    host = {"type": "host_tool_call", "id": "h1", "toolCallId": "c1", "toolName": "aps_skill_read",
            "arguments": {**arguments, "i": "Reading the procedure"}}

    async def scenario():
        await trial.handle(message)
        await trial.handle(host)

    asyncio.run(scenario())
    call = next(event for event in events if event["type"] == "tool_call")
    result = next(event for event in events if event["type"] == "tool_result")
    assert set(call) == {"type", "host_request_id", "call_id", "name", "wire_name", "arguments", "host_arguments"}
    assert call["name"] == "skill_read" and call["arguments"] == arguments
    assert set(result) == {"type", "host_request_id", "call_id", "name", "wire_name", "result"}
    assert result["result"] == {"content": [{"type": "text", "text": ADS_SKILL}], "isError": False}
    assert sent[0]["isError"] is False
    assert json.loads(sent[0]["result"]["content"][0]["text"]) == result["result"]
