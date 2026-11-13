import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import topology_mutator as tm

FAMILY_SEEDS = [(seed, tm.family_for_seed(seed)) for seed in (*tm.TRAIN_SEEDS, *tm.EVAL_SEEDS)]
FILES = ("HWConf.xml", "HWConf.Emulator.xml", "Sources.xml", "Worker.xml", "Alarms.xml", "Interlocks.xml", "manifest.json")
DEVICE_TAGS = {"valve": "Valve", "pump": "Pump", "sensor": "Sensor", "mfc": "MFC", "powersupply": "Power",
               "controller": "Controller", "stage": "Stage", "drive": "Drive", "robot": "Robot"}
CSHARP_FILE = (
    Path(os.environ["ADS_CONTROL_COMMUNITY_DIR"]) / "Control/HW_Libraries/Demo/LanPowerSupply.cs"
    if os.environ.get("ADS_CONTROL_COMMUNITY_DIR")
    else None
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("topologies")
    manifests = tm.generate_all(root)
    return root, manifests


def topo_dir(root, seed):
    split = "train" if seed in tm.TRAIN_SEEDS else "eval"
    return root / split / f"topo_{seed}"


def hw_devices(path):
    devices = {}
    for element in ET.parse(path).getroot().iter():
        kind = DEVICE_TAGS.get(element.tag)
        if kind and element.text and element.text.strip():
            devices.setdefault(kind, set()).add(element.text.strip())
    return devices


def interlock_files(directory):
    root = ET.parse(directory / "Interlocks.xml").getroot()
    return [directory / "Interlocks" / node.text.strip() for node in root.iter("file")]


def test_generate_all_layout_and_counts(generated):
    root, manifests = generated
    assert len(manifests) == 30
    assert sorted(p.name for p in (root / "train").iterdir()) == [f"topo_{s}" for s in tm.TRAIN_SEEDS]
    assert sorted(p.name for p in (root / "eval").iterdir()) == [f"topo_{s}" for s in tm.EVAL_SEEDS]


@pytest.mark.parametrize("seed,family", FAMILY_SEEDS)
def test_files_are_well_formed_and_manifest_round_trips(generated, seed, family):
    root, manifests = generated
    directory = topo_dir(root, seed)
    for name in FILES:
        assert (directory / name).is_file(), name
    for path in directory.rglob("*.xml"):
        ET.parse(path)
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8")) == manifests[seed]
    assert manifests[seed]["profile"] == f"SYNTH_PVD_{seed}"
    assert manifests[seed]["family"] == family


@pytest.mark.parametrize("seed,family", FAMILY_SEEDS)
def test_cross_file_referential_integrity(generated, seed, family):
    root, _ = generated
    directory = topo_dir(root, seed)
    devices = hw_devices(directory / "HWConf.xml")
    emulator_devices = hw_devices(directory / "HWConf.Emulator.xml")
    assert devices == emulator_devices

    for source in ET.parse(directory / "Sources.xml").getroot().iter("source"):
        psu = (source.findtext("powerSupplyId") or "").strip()
        if psu:
            assert psu in devices["Power"], psu
    declared = {n.get("id") for n in ET.parse(directory / "Sources.xml").getroot().iter("powersource")}
    assert declared <= devices["Power"]

    for mfc in ET.parse(directory / "HWConf.xml").getroot().iter("mfc"):
        gates = [v for v in (mfc.get("valve") or "").split(",") if v]
        assert 1 <= len(gates) <= 2
        assert set(gates) <= devices["Valve"]

    id_pattern = re.compile(r"^(?P<kind>valve|pump|power)_(?P<dev>.+)_(?P<verb>open|close|on|off)$", re.IGNORECASE)
    kind_map = {"valve": "Valve", "pump": "Pump", "power": "Power"}
    line_pattern = re.compile(r"^(Valve|Pump|Sensor)\s+(\S+)\s+")
    found_lines = 0
    for path in interlock_files(directory):
        for block in ET.parse(path).getroot().iter("interlock"):
            match = id_pattern.match(block.get("id"))
            assert match, block.get("id")
            assert match["dev"] in devices[kind_map[match["kind"].lower()]], block.get("id")
            for line in block.iter("l"):
                parsed = line_pattern.match(line.text.strip())
                assert parsed, line.text
                assert parsed[2] in devices[parsed[1]], line.text
                found_lines += 1
    assert found_lines > 0

    alarms = ET.parse(directory / "Alarms.xml").getroot()
    for alarm in alarms.iter("alarm"):
        if alarm.get("sensor"):
            assert alarm.get("sensor") in devices["Sensor"]
        if alarm.get("device"):
            assert alarm.get("device") in devices["Pump"]
        if alarm.get("unless"):
            assert alarm.get("unless").split()[0] in devices["Valve"]
    for transition in alarms.iter("transition"):
        if transition.get("sensor"):
            assert transition.get("sensor") in devices["Sensor"]
        if transition.get("valve"):
            assert transition.get("valve") in devices["Valve"]


@pytest.mark.parametrize("seed,family", FAMILY_SEEDS)
def test_alarm_delay_threshold_and_worker_match_manifest(generated, seed, family):
    root, manifests = generated
    directory = topo_dir(root, seed)
    manifest = manifests[seed]

    alarms = ET.parse(directory / "Alarms.xml").getroot()
    idle_rules = {}
    for chamber in alarms.iter("chamber"):
        for alarm in chamber.iter("alarm"):
            if alarm.get("state") == "idle" and alarm.get("sensor"):
                idle_rules[chamber.get("id")] = alarm
    assert idle_rules
    kinds = {c["id"]: c["kind"] for c in manifest["chambers"]}
    for chamber_id, rule in idle_rules.items():
        expected = manifest["alarms"][chamber_id]
        assert float(rule.get("val")) == expected["idle_threshold_torr"]
        assert float(rule.get("delay", "0")) == expected["delay_s"]
        assert rule.get("unless") == expected["unless"]
        if kinds[chamber_id] == "process":
            assert expected["delay_s"] in tm.ALARM_DELAYS_S
            assert expected["idle_threshold_torr"] in tm.IDLE_THRESHOLDS_TORR

    ops = {op.get("method"): op for op in ET.parse(directory / "Worker.xml").getroot().iter("op")}
    pwait = ops["Pwait"]
    assert (pwait.get("device") is not None) == manifest["pwait"]["bound"]
    assert len(pwait.findall("var")) == (4 if manifest["pwait"]["bound"] else 3)
    assert manifest["pwait"]["args"] == (2 if manifest["pwait"]["bound"] else 3)
    deposit = ops["Deposit"]
    assert len(deposit.findall("var")) == (4 if manifest["deposit"]["args"] == 3 else 2)
    if manifest["pwait"]["bound"]:
        assert pwait.get("device") in hw_devices(directory / "HWConf.xml")["Controller"]


@pytest.mark.parametrize("seed,family", FAMILY_SEEDS)
def test_psu_drivers_and_set_parameters(generated, seed, family):
    root, manifests = generated
    manifest = manifests[seed]
    hw_modes = {}
    for group in ET.parse(topo_dir(root, seed) / "HWConf.xml").getroot().iter("powersupplies"):
        assert group.get("hwtype") == "APS.Hardware.Demo.LanPowerSupply"
        for psu in group.findall("powersupply"):
            psu_id = psu.text.strip()
            mode = psu.get("mode")
            hw_modes[psu_id] = "LanPowerSupply_ION" if mode == "DC" and psu_id.startswith("ION") else f"LanPowerSupply_{mode}"
    assert set(manifest["psus"]) == set(hw_modes)
    for psu_id, info in manifest["psus"].items():
        assert info["driver"] == hw_modes[psu_id]
        assert info["set_params"] == tm.DRIVERS[info["driver"]]["params"]


@pytest.mark.parametrize(
    "driver,field_name,expected",
    [
        ("LanPowerSupply_DC", "s_dcNames", ["Power", "Current"]),
        ("LanPowerSupply_PDC", "s_pdcNames", ["Power", "Current", "Frequency", "DutyCycle"]),
        ("LanPowerSupply_RF", "s_rfNames", ["Power"]),
        ("LanPowerSupply_ION", "s_dcNames", ["Power", "Current"]),
    ],
)
def test_driver_parameter_lists_match_the_csharp_drivers(driver, field_name, expected):
    assert tm.DRIVERS[driver]["params"] == expected
    if CSHARP_FILE is None or not CSHARP_FILE.is_file():
        return
    text = CSHARP_FILE.read_text(encoding="utf-8-sig")
    block = re.search(rf"{field_name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
    assert block is not None
    assert re.findall(r'"([A-Za-z]+)"', block.group(1)) == tm.DRIVERS[driver]["params"]


def test_seeds_do_not_collide(generated):
    _, manifests = generated
    profiles = [m["profile"] for m in manifests.values()]
    assert len(set(profiles)) == 30
    fingerprints = {tm.structure_fingerprint(m) for m in manifests.values()}
    assert len(fingerprints) == 30
    train = [manifests[s] for s in tm.TRAIN_SEEDS]
    assert len({tm.gas_valve_signature(m) for m in train}) >= 10
    assert len({tuple((s["name"], s["psu"]) for s in m["sources"]) for m in train}) == 25


def test_generation_is_deterministic(tmp_path):
    first = tm.generate_topology(107, "single_sputter", tmp_path / "a")
    second = tm.generate_topology(107, "single_sputter", tmp_path / "b")
    assert first == second
    for name in FILES:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_randomized_axes_cover_both_branches(generated):
    _, manifests = generated
    train = [manifests[s] for s in tm.TRAIN_SEEDS]
    assert {m["pwait"]["bound"] for m in train} == {True, False}
    assert {m["deposit"]["args"] for m in train} == {2, 3}
    assert {m["rotation"]["operational"] for m in train} == {True, False}
    assert {m["interlocks"]["power_threshold_torr"] for m in train} == set(tm.POWER_THRESHOLDS_TORR)
    process_alarms = [m["alarms"][c["id"]] for m in train for c in m["chambers"] if c["kind"] == "process"]
    assert {a["delay_s"] for a in process_alarms} == set(tm.ALARM_DELAYS_S)
    assert {a["idle_threshold_torr"] for a in process_alarms} == set(tm.IDLE_THRESHOLDS_TORR)
    assert {a["unless"] is None for a in process_alarms} == {True, False}
    assert any(s["manual_only"] for m in train for s in m["sources"])
    assert any(s["material"] == "Ion" and not s["manual_only"] for m in train for s in m["sources"])
    assert {len(g["valves"]) for m in train for g in m["gases"]} == {1, 2}


def test_single_sputter_constraints(generated):
    _, manifests = generated
    for seed in tm.TRAIN_SEEDS:
        manifest = manifests[seed]
        if manifest["family"] != "single_sputter":
            continue
        sputter = [s for s in manifest["sources"] if s["material"] != "Ion"]
        assert 3 <= len(sputter) <= 6
        assert all(s["material"] in tm.MATERIAL_POOL for s in sputter)
        assert 5 <= len(next(iter(manifest["interlocks"]["vent"].values()))) <= 9
        assert tm.gas_valve_signature(manifest) != tm.ADS_GAS_SIGNATURE


def test_cluster_family_uses_unbound_pwait_and_per_chamber_sensors(generated):
    _, manifests = generated
    cluster = [manifests[s] for s in (*tm.TRAIN_SEEDS, *tm.EVAL_SEEDS) if manifests[s]["family"] == "cluster"]
    assert len(cluster) == 6
    for manifest in cluster:
        assert manifest["pwait"] == {"bound": False, "device": None, "args": 3}
        assert manifest["deposit"]["args"] == 2
        process = [c for c in manifest["chambers"] if c["kind"] == "process"]
        assert 2 <= len(process) <= 4
        assert len({c["sensor"] for c in manifest["chambers"]}) == len(manifest["chambers"])
        assert {s["chamber"] for s in manifest["sources"]} <= {c["id"] for c in process}
        per_chamber = [sum(1 for s in manifest["sources"] if s["chamber"] == c["id"]) for c in process]
        assert min(per_chamber) >= 1 and max(per_chamber) == 2


def test_seed_configs_are_only_read(tmp_path):
    before = {p: p.stat().st_mtime_ns for p in tm.CONFIGS_ROOT.glob("*/Worker.xml")}
    tm.generate_topology(101, "single_sputter", tmp_path)
    assert before == {p: p.stat().st_mtime_ns for p in tm.CONFIGS_ROOT.glob("*/Worker.xml")}



def test_zero_collisions_against_all_four_held_out_configs(generated):
    _, manifests = generated
    sigs = tm.held_out_signatures()
    assert len(sigs) == 4
    for seed, manifest in manifests.items():
        assert tm.held_out_collisions(manifest) == [], f"seed {seed} collided with held-out config"
