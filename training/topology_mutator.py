import hashlib
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = Path(os.environ.get("ADS_BENCH_CONFIGS_DIR") or (_REPO_ROOT / "configs"))
COMPUTED_ROOT = Path(os.environ.get("ADS_BENCH_COMPUTED_DIR") or (_REPO_ROOT / "computed"))
TOPOLOGY_ROOT = COMPUTED_ROOT / "topologies"

TRAIN_SEEDS = range(101, 126)
EVAL_SEEDS = range(201, 206)

MATERIAL_POOL = ("W", "Mo", "Si", "B4C", "Ru", "C", "Cr", "Ti", "Cu", "Co", "Pt", "Al")
MATERIAL_LABELS = {
    "W": "Tungsten", "Mo": "Molybdenum", "Si": "Silicon", "B4C": "Boron carbide", "Ru": "Ruthenium",
    "C": "Graphite", "Cr": "Chromium", "Ti": "Titanium", "Cu": "Copper", "Co": "Cobalt",
    "Pt": "Platinum", "Al": "Aluminium",
}
IDLE_THRESHOLDS_TORR = (5e-5, 1e-4, 2e-4)
ALARM_DELAYS_S = (30.0, 45.0, 60.0, 90.0, 120.0)
POWER_THRESHOLDS_TORR = (500.0, 750.0)
PROCESSING_THRESHOLD_TORR = 0.2
GAUGE_FLOOR_TORR = 1e-5

DRIVERS = {
    "LanPowerSupply_DC": {
        "prefix": "DC", "mode": "DC", "power_type": "DC", "hwtype": "APS.Hardware.Demo.LanPowerSupply",
        "params": ["Power", "Current"],
    },
    "LanPowerSupply_PDC": {
        "prefix": "PDC", "mode": "PDC", "power_type": "DC pulsed", "hwtype": "APS.Hardware.Demo.LanPowerSupply",
        "params": ["Power", "Current", "Frequency", "DutyCycle"],
    },
    "LanPowerSupply_RF": {
        "prefix": "RF", "mode": "RF", "power_type": "RF", "hwtype": "APS.Hardware.Demo.LanPowerSupply",
        "params": ["Power"],
    },
    "LanPowerSupply_ION": {
        "prefix": "ION", "mode": "DC", "power_type": "Ion source", "hwtype": "APS.Hardware.Demo.LanPowerSupply",
        "params": ["Power", "Current"],
    },
}
SPUTTER_DRIVERS = ("LanPowerSupply_DC", "LanPowerSupply_PDC", "LanPowerSupply_RF")

KIND_TAGS = {
    "Valve": ("valves", "valve"), "Pump": ("pumps", "pump"), "MFC": ("mfcs", "mfc"),
    "Sensor": ("sensors", "sensor"), "Stage": ("stages", "stage"), "Controller": ("controllers", "controller"),
    "Power": ("powersupplies", "powersupply"), "Drive": ("drives", "drive"), "Robot": ("robots", "robot"),
}
KIND_ORDER = ("Valve", "MFC", "Sensor", "Pump", "Stage", "Controller", "Power", "Drive", "Robot")
ADS_GAS_SIGNATURE = (("Ar", ("V2", "V3")), ("N2", ("V1", "V4")))
MFC_NAME_STYLES = ("{gas}{i}", "MFC_{gas}{i}", "{gas}_MFC{i}")
GAS_DESCRIPTIONS = {"Ar": "Ar inlet", "N2": "N2 inlet", "O2": "O2 inlet"}
DEVICE_STATES = ("processing", "idle", "maintenance")

# SHA-256 digests of the 4th held-out physical chamber configuration (profile name, device set, MFC gates, and active source-PSU pairs).
HELD_OUT_PHYSICAL_SIGNATURE = {
    "name_sha256": "0fe2b63a8658ebb1c54d65f99ee9d71f068589af08841793cd8d8f90b986e1af",
    "devices_sha256": "76ea3672ae3f6d5dd899babf97e30a3978192b41d207bb370fecefbae093778a",
    "source_pairs_sha256": "ebce108e4aa7cb177d3dd76826bdc76a26e09aa68562762488d518312b268207",
    "mfc_gates_sha256": "b5269bb97ceaa1b3882f4218684d4bafe505ad2297733b462e61446e38537cfb",
    "pair_sha256s": (
        "2dd0ba1b31c35af083881dbf7e5d268425c6b4e04661252ece00e557bbdbb662",
        "594764e24c03056aa961bb4c01488cb3f0dcb9049893d161bfcc5ddf0a7863d1",
        "8f3ca5cd2e540cf0f541685a931dabe657a861d7687771f8b323e7ff0f9e21a2",
        "276d046c6d3c4b287fe1e87ade396b83b1f1e7fc958d764557f2b5375ca157bf",
        "f726fe8127b9980c55b3f878e6d4f2574112fdcae0d570ded88c640877342286",
    ),
}
PUBLIC_HELD_OUT_PROFILES = ("Sputter", "ADS", "Cluster")


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()


def _config_dir_signature(config_dir: Path) -> dict:
    hw = ET.parse(config_dir / "HWConf.xml").getroot()
    devs = sorted({
        (el.text or "").strip()
        for c in hw
        for el in c
        if (el.text or "").strip() and c.tag not in ("libraries", "plcs")
    })
    mfcs = {}
    for c in hw:
        if c.tag == "mfcs":
            for el in c.findall("mfc"):
                mid = (el.text or "").strip()
                mfcs[mid] = list(sorted(v for v in el.get("valve", "").split(",") if v))
    src = ET.parse(config_dir / "Sources.xml").getroot()
    all_pairs = sorted({
        ((n.get("name") or (n.findtext("material") or "").strip()), (n.findtext("powerSupplyId") or "").strip())
        for n in src.findall("source")
        if (n.get("name") or (n.findtext("material") or "").strip() or (n.findtext("powerSupplyId") or "").strip())
    })
    active_pairs = [(name, psu) for name, psu in all_pairs if name and psu]
    return {
        "name_sha256": hashlib.sha256(config_dir.name.encode("utf-8")).hexdigest(),
        "devices_sha256": _sha256_json([list(x) if isinstance(x, tuple) else x for x in devs]),
        "source_pairs_sha256": _sha256_json([list(p) for p in all_pairs]),
        "mfc_gates_sha256": _sha256_json([[k, v] for k, v in sorted(mfcs.items())]),
        "pair_sha256s": tuple(_sha256_json(list(p)) for p in active_pairs),
    }


def held_out_signatures(configs_root: Path = CONFIGS_ROOT) -> list[dict]:
    sigs = [dict(HELD_OUT_PHYSICAL_SIGNATURE)]
    for name in PUBLIC_HELD_OUT_PROFILES:
        d = Path(configs_root) / name
        if (d / "HWConf.xml").is_file() and (d / "Sources.xml").is_file():
            sigs.append(_config_dir_signature(d))
    return sigs


def manifest_collision_signature(manifest: dict) -> dict:
    devs = sorted({d["id"] for items in manifest["devices"].values() for d in items})
    mfcs = {mid: list(sorted(info["gate_valves"])) for mid, info in manifest["mfcs"].items()}
    all_pairs = sorted({(s["name"], s["psu"] or "") for s in manifest["sources"]})
    active_pairs = [(name, psu) for name, psu in all_pairs if name and psu]
    return {
        "name_sha256": hashlib.sha256(manifest["profile"].encode("utf-8")).hexdigest(),
        "devices_sha256": _sha256_json(devs),
        "source_pairs_sha256": _sha256_json([list(p) for p in all_pairs]),
        "mfc_gates_sha256": _sha256_json([[k, v] for k, v in sorted(mfcs.items())]),
        "pair_sha256s": tuple(_sha256_json(list(p)) for p in active_pairs),
    }


def held_out_collisions(manifest: dict, configs_root: Path = CONFIGS_ROOT) -> list[str]:
    sig = manifest_collision_signature(manifest)
    collisions = []
    for index, held in enumerate(held_out_signatures(configs_root)):
        for field in ("name_sha256", "devices_sha256", "source_pairs_sha256", "mfc_gates_sha256"):
            if sig[field] == held[field]:
                collisions.append(f"held_out_{index}:{field}")
        overlap = set(sig["pair_sha256s"]) & set(held["pair_sha256s"])
        if overlap:
            collisions.append(f"held_out_{index}:active_source_pair")
    return collisions


def family_for_seed(seed: int) -> str:
    return "cluster" if seed % 5 == 0 else "single_sputter"


def structure_fingerprint(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k not in ("seed", "profile")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def gas_valve_signature(manifest: dict) -> tuple:
    merged = {}
    for entry in manifest["gases"]:
        merged.setdefault(entry["gas"], set()).update(entry["valves"])
    return tuple((gas, tuple(sorted(valves))) for gas, valves in sorted(merged.items()))


class _Devices:
    def __init__(self):
        self.items = {kind: [] for kind in KIND_TAGS}

    def add(self, kind, dev_id, description, chamber=None, hwtype=None, plc_host=None, **attrs):
        self.items[kind].append({
            "id": dev_id, "description": description, "chamber": chamber, "hwtype": hwtype, "host": plc_host,
            "attrs": {k: v if isinstance(v, (list, dict)) else str(v) for k, v in attrs.items()},
        })


def _psu_names(drivers):
    counters = {}
    names = []
    for driver in drivers:
        prefix = DRIVERS[driver]["prefix"]
        counters[prefix] = counters.get(prefix, 0) + 1
        names.append(f"{prefix}_{counters[prefix]}")
    return names


def _add_psu(devices, psu_id, driver, index):
    spec = DRIVERS[driver]
    attrs = {"host": "127.0.0.1", "port": 5025 + index, "mode": spec["mode"]}
    devices.add("Power", psu_id, f"{spec['power_type']} supply {psu_id}", None, spec["hwtype"], None, **attrs)


def _pick_gases(rng):
    return ["Ar", *rng.choice((["N2"], ["O2"], ["N2", "O2"]))]


def _alarm_entry(sensor, idle_threshold, delay, unless, message, transitions, device_rules, processing=True):
    rules = []
    if processing:
        rules.append({"sensor": sensor, "value": PROCESSING_THRESHOLD_TORR, "condition": 1, "state": "processing",
                      "message": message, "delay_s": 0.0, "unless": None})
    rules.append({"sensor": sensor, "value": idle_threshold, "condition": 1, "state": "idle",
                  "message": message, "delay_s": delay, "unless": unless})
    return {"sensor": sensor, "idle_threshold_torr": idle_threshold,
            "processing_threshold_torr": PROCESSING_THRESHOLD_TORR if processing else None,
            "delay_s": delay, "unless": unless, "message": message, "rules": rules,
            "deviceStatusRules": device_rules, "transitions": transitions}


def _device_rules(pump, message, states=DEVICE_STATES):
    return [{"device": pump, "statusBit": 3, "state": state, "message": message} for state in states]


def _process_transitions(sensor, vent, gate):
    return {
        "sensor": [{"from": "startup", "to": "idle", "sensor": sensor, "cond": "less", "val": 1e-6}],
        "valve": [
            {"from": "idle", "to": "maintenance", "valve": vent, "state": "open", "require": f"{gate} closed"},
            {"from": "maintenance", "to": "startup", "valve": vent, "state": "closed", "require": None},
        ],
        "group": [
            {"from": "startup", "to": "processing", "group": "gasvalve", "state": "anyOpen"},
            {"from": "idle", "to": "processing", "group": "gasvalve", "state": "anyOpen"},
            {"from": "processing", "to": "idle", "group": "gas", "state": "allOff"},
        ],
    }


def _build_single(rng):
    devices = _Devices()
    chamber = "Main"
    prefix = rng.choice(("V", "GV"))
    mfc_style = rng.choice(MFC_NAME_STYLES)
    gases = _pick_gases(rng)
    valve_numbers = iter(rng.sample(range(1, 9), 2 * len(gases)))

    vent = rng.choice(("A1", "VENT", "AV1"))
    gate = rng.choice(("GVM", "GV_MAIN", "GATE1"))
    turbo = rng.choice(("TP", "TMP1"))
    backing = rng.choice(("SP1", "RP1"))
    sensor = rng.choice(("pm", "PG1", "Pmain"))
    bound = rng.random() < 0.5
    deposit_args = rng.choice((2, 3))
    rotation_operational = rng.random() < 0.5

    n_sputter = rng.randint(3, 6)
    with_ion = rng.random() < 0.5
    ion_manual = with_ion and rng.random() < 0.5
    n_total = n_sputter + (1 if with_ion else 0)
    flanges = sorted(rng.sample(range(1, 9), n_total))
    ion_slot = rng.randrange(n_total) if with_ion else None
    materials = iter(rng.sample(MATERIAL_POOL, n_sputter))
    sputter_drivers = [rng.choice(SPUTTER_DRIVERS) for _ in range(n_sputter)]
    driver_list = list(sputter_drivers) + (["LanPowerSupply_ION"] if with_ion and not ion_manual else [])
    psu_ids = _psu_names(driver_list)

    gas_entries, mfcs = [], {}
    for gas in gases:
        valves = [f"{prefix}{next(valve_numbers)}" for _ in range(rng.choice((1, 2)))]
        mfc_ids = [mfc_style.format(gas=gas, i=i + 1) for i in range(rng.choice((1, 2)))]
        for valve in valves:
            devices.add("Valve", valve, GAS_DESCRIPTIONS[gas], chamber, "APS.Hardware.S7.S7200GasValve", "main", address=12)
        for mfc in mfc_ids:
            devices.add("MFC", mfc, f"{gas} MFC", chamber, "APS.Hardware.S7.S7200MFC", "main",
                        address=160 + 8 * len(mfcs), valve=",".join(valves))
            mfcs[mfc] = {"gas": gas, "chamber": chamber, "gate_valves": list(valves)}
        gas_entries.append({"gas": gas, "chamber": chamber, "valves": valves, "mfcs": mfc_ids, "gating": "OR"})

    sources, inlet_valves = [], []
    sputter_iter = iter(zip(materials, sputter_drivers, psu_ids))
    for slot, flange in enumerate(flanges):
        if slot == ion_slot:
            sources.append({
                "flange": flange, "name": "Ion", "material": "Ion", "material_text": "Ion", "chamber": chamber,
                "psu": None if ion_manual else psu_ids[n_sputter],
                "driver": None if ion_manual else "LanPowerSupply_ION",
                "power_type": "Ion source (manual)" if ion_manual else "Ion source",
                "magnetron_id": "ION", "manual_only": ion_manual, "inlet_valve": "VI", "shutter": None,
            })
            devices.add("Valve", "VI", "Ion source inlet", chamber, "APS.Hardware.S7.S7200GasValve", "main", address=2)
            continue
        material, driver, psu = next(sputter_iter)
        inlet = f"M{len(inlet_valves) + 1}"
        inlet_valves.append(inlet)
        sources.append({
            "flange": flange, "name": material, "material": material,
            "material_text": rng.choice((material, MATERIAL_LABELS[material])), "chamber": chamber,
            "psu": psu, "driver": driver, "power_type": DRIVERS[driver]["power_type"],
            "magnetron_id": rng.choice(("MAG1", "MAG4", "RECT")), "manual_only": False,
            "inlet_valve": inlet, "shutter": None,
        })
        devices.add("Valve", inlet, f"Gas inlet at {material} source", chamber, "APS.Hardware.S7.S7200GasValve", "main",
                    address=12)
    devices.add("Valve", "PP", "Pre-pump", chamber, "APS.Hardware.S7.S7200Valve", "main", address=2)
    devices.add("Valve", "FL", "Fore line", chamber, "APS.Hardware.S7.S7200Valve", "main", address=2)
    devices.add("Valve", vent, "Air vent", chamber, "APS.Hardware.S7.S7200Valve", "main", address=2, vent="true",
                conductance="3.0")
    devices.add("Valve", gate, "Gate Main", chamber, "APS.Hardware.S7.S7200InputValve", "main", address=2)
    devices.add("Sensor", sensor, "System", chamber, "APS.Hardware.S7.S7200Sensor", "main", address=100)
    devices.add("Sensor", "tmp", "Temp", chamber, "APS.Hardware.S7.S7200Sensor", "main", address=124, property="Temperature")
    devices.add("Pump", turbo, "Main pump", chamber, "APS.Hardware.S7.S7200TurboPump", "main", address=17, gate=gate,
                foreline="FL")
    devices.add("Pump", backing, "Backing pump", chamber, "APS.Hardware.S7.S7200Pump", "main", address=0, gate="PP")
    devices.add("Stage", "VS", "Substrate", None, "APS.Hardware.S7.S7200Stage", "main", address=152,
                presets="Load:0;Process:25")
    rotation_attrs = {"address": 1974, "feedbackAddr": 1978}
    if not rotation_operational:
        rotation_attrs["operational"] = "false"
    devices.add("Stage", "rotation", "Substrate Rotation", None, "APS.Hardware.S7.S7200RotationStage", "main",
                **rotation_attrs)
    if bound:
        devices.add("Controller", "CTRL_1", "Throttle controller", chamber, "APS.Hardware.Demo.LanThrottleController",
                    None, host="127.0.0.1", port=5035)
    devices.add("Drive", "Shutter", "Shutter", None, "APS.Hardware.Demo.LanRotaryDrive", None, host="127.0.0.1", port=5036)
    devices.add("Drive", "Holder", "Holder", None, "APS.Hardware.Demo.LanRotaryDrive", None, host="127.0.0.1", port=5037)
    for index, (psu, driver) in enumerate(zip(psu_ids, driver_list)):
        _add_psu(devices, psu, driver, index)

    delay = rng.choice(ALARM_DELAYS_S)
    idle_threshold = rng.choice(IDLE_THRESHOLDS_TORR)
    unless = f"{vent} open" if rng.random() < 0.5 else None
    power_threshold = rng.choice(POWER_THRESHOLDS_TORR)
    alarms = {chamber: _alarm_entry(
        sensor, idle_threshold, delay, unless, "Main chamber pressure too high!",
        _process_transitions(sensor, vent, gate),
        _device_rules(turbo, "Turbo pump hardware alarm!"))}

    gas_valves = [v for entry in gas_entries for v in entry["valves"]]
    prereq_pool = ["PP", gate, *inlet_valves, *(["VI"] if with_ion else []), *gas_valves]
    rng.shuffle(prereq_pool)
    interlocks = {
        "vent": {vent: prereq_pool[: rng.randint(5, min(9, len(prereq_pool)))]},
        "pump_valves": {"FL": [f"Pump {backing} on"], "PP": [f"Pump {backing} on"]},
        "gate": {gate: [f"Sensor {sensor} less 2.0e-1", "Valve PP closed"]},
        "pump_off": {backing: ["Valve FL closed", "Valve PP closed"]},
        "slit": {},
        "power_threshold_torr": power_threshold,
        "power": {psu: {"sensor": sensor, "threshold_torr": power_threshold} for psu in psu_ids},
    }
    return {
        "chambers": [{"id": chamber, "kind": "process", "sensor": sensor, "vent_valve": vent, "gate_valve": gate}],
        "gases": gas_entries, "mfcs": mfcs, "sources": sources,
        "psus": {p: {"driver": d, "set_params": list(DRIVERS[d]["params"])} for p, d in zip(psu_ids, driver_list)},
        "pwait": {"bound": bound, "device": "CTRL_1" if bound else None, "args": 2 if bound else 3},
        "deposit": {"args": deposit_args},
        "rotation": {"present": True, "operational": rotation_operational},
        "alarms": alarms, "interlocks": interlocks, "devices": devices,
    }


def _build_cluster(rng):
    devices = _Devices()
    n_process = rng.randint(2, 4)
    process = [f"C{i + 1}" for i in range(n_process)]
    total = rng.randint(max(3, n_process + 1), min(6, 2 * n_process))
    counts = {c: 1 for c in process}
    for _ in range(total - n_process):
        counts[rng.choice([c for c in process if counts[c] < 2])] += 1
    materials = iter(rng.sample(MATERIAL_POOL, total))
    drivers = [rng.choice(SPUTTER_DRIVERS) for _ in range(total)]
    psu_iter = iter(zip(_psu_names(drivers), drivers))
    flange_iter = iter(rng.sample(range(1, 9), total))

    gas_entries, mfcs, sources, psus, power = [], {}, [], {}, {}
    chambers = [
        {"id": "TM", "kind": "transfer", "sensor": "TM_pm", "vent_valve": None, "gate_valve": "TM_GV"},
        {"id": "LL", "kind": "loadlock", "sensor": "LL_pm", "vent_valve": "LL_A1", "gate_valve": "LL_GV"},
    ]
    for name in ("TM", "LL"):
        devices.add("Sensor", f"{name}_pm", f"{name} pressure", name, "APS.Hardware.S7.S7200Sensor", "tm", address=100)
        devices.add("Valve", f"{name}_FL", f"{name} foreline", name, "APS.Hardware.S7.S7200Valve", "tm", address=2)
        devices.add("Valve", f"{name}_RL", f"{name} roughing line", name, "APS.Hardware.S7.S7200Valve", "tm", address=3)
        devices.add("Valve", f"{name}_GV", f"Gate {name}", name, "APS.Hardware.S7.S7200InputValve", "tm", address=2)
        devices.add("Pump", f"{name}_TP", f"{name} turbo", name, "APS.Hardware.S7.S7200TurboPump", "tm", address=17,
                    gate=f"{name}_GV", foreline=f"{name}_FL")
        devices.add("Pump", f"{name}_SP", f"{name} backing", name, "APS.Hardware.S7.S7200Pump", "tm", address=0,
                    gate=f"{name}_RL")
    devices.add("Valve", "LL_A1", "LL vent", "LL", "APS.Hardware.S7.S7200Valve", "tm", address=2, vent="true",
                conductance="3.0")
    for slit in ("LL", *process):
        devices.add("Valve", f"SV_{slit}", f"Slit TM-{slit}", "TM", "APS.Hardware.S7.S7200Valve", "tm", address=3,
                    connects=f"TM,{slit}")
    faces = [{"name": "LL", "slit": "SV_LL"}, *({"name": c, "slit": f"SV_{c}"} for c in process)]
    devices.add("Robot", "ROBOT", "Transfer robot", "TM", "APS.Hardware.Demo.LanTransferRobot", None,
                host="127.0.0.1", port=5038, faces=faces)

    delay = rng.choice(ALARM_DELAYS_S)
    idle_threshold = rng.choice(IDLE_THRESHOLDS_TORR)
    use_unless = rng.random() < 0.5
    power_threshold = rng.choice(POWER_THRESHOLDS_TORR)

    alarms = {
        "TM": _alarm_entry(
            "TM_pm", 1e-3, 0.0, None, "TM pressure too high!",
            {"sensor": [{"from": "startup", "to": "idle", "sensor": "TM_pm", "cond": "less", "val": 1e-5}],
             "valve": [], "group": []},
            _device_rules("TM_TP", "TM turbo pump hardware alarm!", ("idle",)), processing=False),
        "LL": _alarm_entry(
            "LL_pm", 1e-1, 0.0, "LL_A1 open", "LL pressure too high!",
            {"sensor": [{"from": "startup", "to": "idle", "sensor": "LL_pm", "cond": "less", "val": 1e-3}],
             "valve": [{"from": "idle", "to": "maintenance", "valve": "LL_A1", "state": "open", "require": "SV_LL closed"},
                       {"from": "maintenance", "to": "startup", "valve": "LL_A1", "state": "closed", "require": None}],
             "group": []},
            _device_rules("LL_TP", "LL turbo pump hardware alarm!", ("idle",)), processing=False),
    }
    interlocks = {
        "vent": {"LL_A1": ["SV_LL", "LL_GV", "LL_RL"]}, "pump_valves": {}, "gate": {}, "pump_off": {},
        "slit": {"SV_LL": ["Sensor LL_pm less 1e-4", "Sensor TM_pm less 1e-4"]},
    }
    for name in ("TM", "LL"):
        interlocks["pump_valves"][f"{name}_FL"] = [f"Pump {name}_SP on", f"Valve {name}_RL closed"]
        interlocks["pump_off"][f"{name}_SP"] = [f"Valve {name}_FL closed", f"Valve {name}_RL closed"]

    for chamber in process:
        host = chamber.lower()
        sensor, vent, gate = f"{chamber}_pm", f"{chamber}_A1", f"{chamber}_GV"
        chambers.append({"id": chamber, "kind": "process", "sensor": sensor, "vent_valve": vent, "gate_valve": gate})
        devices.add("Sensor", sensor, f"{chamber} pressure", chamber, "APS.Hardware.S7.S7200Sensor", host, address=100)
        devices.add("Sensor", f"{chamber}_tmp", f"{chamber} temperature", chamber, "APS.Hardware.S7.S7200Sensor", host,
                    address=124, property="Temperature")
        devices.add("Valve", f"{chamber}_FL", f"{chamber} foreline", chamber, "APS.Hardware.S7.S7200Valve", host, address=2)
        devices.add("Valve", f"{chamber}_RL", f"{chamber} roughing line", chamber, "APS.Hardware.S7.S7200Valve", host, address=3)
        devices.add("Valve", vent, f"{chamber} vent", chamber, "APS.Hardware.S7.S7200Valve", host, address=3, vent="true",
                    conductance="3.0")
        devices.add("Valve", gate, f"Gate {chamber}", chamber, "APS.Hardware.S7.S7200InputValve", host, address=2)
        devices.add("Pump", f"{chamber}_TP", f"{chamber} turbo", chamber, "APS.Hardware.S7.S7200TurboPump", host, address=17,
                    gate=gate, foreline=f"{chamber}_FL")
        devices.add("Pump", f"{chamber}_SP", f"{chamber} backing", chamber, "APS.Hardware.S7.S7200Pump", host, address=0,
                    gate=f"{chamber}_RL")
        devices.add("Stage", f"{chamber}_VS", "Substrate lift", chamber, "APS.Hardware.S7.S7200Stage", host, address=308,
                    presets="Load:0;Process:25")
        devices.add("Controller", f"{chamber}_TV", f"{chamber} throttle valve", chamber,
                    "APS.Hardware.S7.S7200ThrottleValve", host, address=200)

        gas_valves = []
        for gas in (_pick_gases(rng) if rng.random() < 0.4 else ["Ar"]):
            infix = "" if gas == "Ar" else f"_{gas}"
            valves = [f"{chamber}_V{infix}" + ("" if n == 0 else f"_{n + 1}") for n in range(rng.choice((1, 2)))]
            for valve in valves:
                devices.add("Valve", valve, GAS_DESCRIPTIONS[gas], chamber, "APS.Hardware.S7.S7200GasValve", host, address=12)
            mfc_id = f"{chamber}_{gas}"
            devices.add("MFC", mfc_id, f"{gas} MFC", chamber, "APS.Hardware.S7.S7200MFC", host, address=160,
                        valve=",".join(valves))
            mfcs[mfc_id] = {"gas": gas, "chamber": chamber, "gate_valves": list(valves)}
            gas_entries.append({"gas": gas, "chamber": chamber, "valves": valves, "mfcs": [mfc_id], "gating": "OR"})
            gas_valves.extend(valves)

        shutters = []
        for s_index in range(counts[chamber]):
            material = next(materials)
            psu, driver = next(psu_iter)
            shutter = f"{chamber}_Sh" + ("" if s_index == 0 else "_B")
            shutters.append(shutter)
            devices.add("Valve", shutter, f"{chamber} {material} magnetron shutter", chamber,
                        "APS.Hardware.S7.S7200Valve", host, address=3)
            sources.append({
                "flange": next(flange_iter), "name": f"{chamber}_{material}", "material": material,
                "material_text": rng.choice((material, MATERIAL_LABELS[material])), "chamber": chamber,
                "psu": psu, "driver": driver, "power_type": DRIVERS[driver]["power_type"],
                "magnetron_id": f"MAG_{chamber}{'' if s_index == 0 else 'B'}", "manual_only": False,
                "inlet_valve": None, "shutter": shutter,
            })
            psus[psu] = {"driver": driver, "set_params": list(DRIVERS[driver]["params"])}
            power[psu] = {"sensor": sensor, "threshold_torr": power_threshold}
            _add_psu(devices, psu, driver, len(psus) - 1)

        alarms[chamber] = _alarm_entry(
            sensor, idle_threshold, delay, f"{vent} open" if use_unless else None, f"{chamber} pressure too high!",
            _process_transitions(sensor, vent, gate),
            _device_rules(f"{chamber}_TP", f"{chamber} turbo pump hardware alarm!"))
        extras = [*shutters, *gas_valves]
        rng.shuffle(extras)
        interlocks["vent"][vent] = [f"SV_{chamber}", gate, f"{chamber}_RL", *extras[: rng.randint(1, 3)]]
        interlocks["pump_valves"][f"{chamber}_FL"] = [f"Pump {chamber}_SP on", f"Valve {chamber}_RL closed"]
        interlocks["pump_off"][f"{chamber}_SP"] = [f"Valve {chamber}_FL closed", f"Valve {chamber}_RL closed"]
        interlocks["gate"][gate] = [f"Sensor {sensor} less 5.0e-2", f"Valve {chamber}_RL closed"]
        interlocks["slit"][f"SV_{chamber}"] = [f"Sensor {sensor} less 1e-4", "Sensor TM_pm less 1e-4"]

    interlocks.update(power_threshold_torr=power_threshold, power=power)
    return {
        "chambers": chambers, "gases": gas_entries, "mfcs": mfcs, "sources": sources, "psus": psus,
        "pwait": {"bound": False, "device": None, "args": 3}, "deposit": {"args": 2},
        "rotation": {"present": False, "operational": False},
        "alarms": alarms, "interlocks": interlocks, "devices": devices,
    }


def build_manifest(seed: int, family: Literal["single_sputter", "cluster"]) -> dict:
    rng = random.Random(seed)
    while True:
        if family == "single_sputter":
            parts = _build_single(rng)
            if gas_valve_signature(parts) == ADS_GAS_SIGNATURE:
                continue
        elif family == "cluster":
            parts = _build_cluster(rng)
        else:
            raise ValueError(family)
        devices = parts.pop("devices")
        manifest = {
            "schema_version": 1, "seed": seed, "family": family, "profile": f"SYNTH_PVD_{seed}",
            **parts,
            "gauge": {"floor_torr": GAUGE_FLOOR_TORR},
            "devices": {kind: devices.items[kind] for kind in KIND_TAGS if devices.items[kind]},
        }
        if not held_out_collisions(manifest):
            return json.loads(json.dumps(manifest))


def _render_worker(manifest):
    cluster = manifest["family"] == "cluster"
    root = ET.parse(CONFIGS_ROOT / ("Cluster" if cluster else "ADS") / "Worker.xml").getroot()
    operations = root.find("operations")
    ops = {op.get("method"): deepcopy(op) for op in operations.findall("op")}
    if cluster:
        keep = ("Valve", "MFC", "Set", "Power", "Pwait", "Deposit", "Warmup", "Ramp", "Transfer", "Home", "Abort")
    else:
        keep = ("Valve", "MFC", "Set", "Power", "Pwait", "Deposit", "Warmup", "Ramp", "Holder", "Shutter")
        pwait, deposit = ops["Pwait"], ops["Deposit"]
        if manifest["pwait"]["bound"]:
            pwait.set("device", manifest["pwait"]["device"])
        else:
            del pwait.attrib["device"]
            pwait.remove(pwait.find("var"))
        if manifest["deposit"]["args"] == 2:
            del deposit.attrib["device"]
            for var in deposit.findall("var"):
                deposit.remove(var)
            for kind in ("s", "f"):
                ET.SubElement(deposit, "var").text = kind
    worker = ET.Element("worker")
    worker.append(deepcopy(root.find("library")))
    new_operations = ET.SubElement(worker, "operations", typename=operations.get("typename"))
    for method in keep:
        new_operations.append(ops[method])
    if not cluster:
        worker.append(deepcopy(root.find("vars")))
    return worker


def _indent_tostring(element):
    ET.indent(element, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(element, encoding="unicode") + "\n"


def _render_hwconf(manifest, emulator):
    root = ET.Element("hardware")
    libraries = ET.SubElement(root, "libraries")
    ET.SubElement(libraries, "library", lib="S7").text = "APS.Hardware.S7.dll"
    ET.SubElement(libraries, "library", lib="Demo").text = "APS.Hardware.Demo.dll"
    plcs = ET.SubElement(root, "plcs", lib="S7")
    hosts = ["main"] if manifest["family"] == "single_sputter" else [
        "tm", *(c["id"].lower() for c in manifest["chambers"] if c["kind"] == "process")]
    for offset, host in enumerate(hosts):
        ET.SubElement(plcs, "plc", type="APS.Hardware.S7.S7200PLC", address=f"127.0.0.{1 + offset}").text = host
    groups = {}
    for kind in KIND_ORDER:
        for device in manifest["devices"].get(kind, []):
            groups.setdefault((kind, device["hwtype"], device["host"]), []).append(device)
    for (kind, hwtype, host), devices in groups.items():
        container_tag, item_tag = KIND_TAGS[kind]
        lib_name = hwtype.split(".")[2] if hwtype.startswith("APS.Hardware.") else hwtype.split(".")[0]
        attrs = {"lib": lib_name, "hwtype": hwtype}
        if host:
            attrs["host"] = host
        container = ET.SubElement(root, container_tag, **attrs)
        for device in devices:
            item_attrs = {"descr": device["description"], **{k: v for k, v in device["attrs"].items() if k != "faces"}}
            if device["chamber"] and kind in ("Valve", "Pump", "MFC", "Sensor", "Controller", "Robot"):
                item_attrs["chamber"] = device["chamber"]
            if emulator and hwtype.endswith("LanThrottleController"):
                item_attrs.update(Kp="8", Ki="0.5", Kd="1.0")
            item = ET.SubElement(container, item_tag, **item_attrs)
            item.text = device["id"]
            for index, face in enumerate(device["attrs"].get("faces", []), start=1):
                ET.SubElement(item, "face", name=face["name"], idx=str(index), slit=face["slit"],
                              angle=str(60 * (index - 1)))
    return root


def _render_sources(manifest):
    root = ET.Element("sources")
    magnetrons = ET.SubElement(root, "magnetrons")
    seen = []
    for source in manifest["sources"]:
        if source["magnetron_id"] not in seen:
            seen.append(source["magnetron_id"])
            ET.SubElement(magnetrons, "magnetron", id=source["magnetron_id"]).text = f"Magnetron {source['magnetron_id']}"
    powersources = ET.SubElement(root, "powersources")
    for psu in manifest["psus"]:
        ET.SubElement(powersources, "powersource", id=psu)
    ET.SubElement(root, "layout", start="90", direction="ccw")
    by_flange = {s["flange"]: s for s in manifest["sources"]}
    for flange in range(1, max(6, max(by_flange)) + 1):
        source = by_flange.get(flange)
        if source is None:
            node = ET.SubElement(root, "source", flange=str(flange))
            for tag in ("material", "magnetronId", "powerSupplyId", "model", "powerType"):
                ET.SubElement(node, tag)
            ET.SubElement(node, "description").text = f"Flange {flange}"
            continue
        attrs = {"flange": str(flange), "name": source["name"]}
        if source["shutter"]:
            attrs.update(chamber=source["chamber"], shutterId=source["shutter"])
        node = ET.SubElement(root, "source", **attrs)
        ET.SubElement(node, "material").text = source["material_text"]
        ET.SubElement(node, "magnetronId").text = source["magnetron_id"]
        ET.SubElement(node, "powerSupplyId").text = source["psu"] or ""
        ET.SubElement(node, "model").text = f"Magnetron {source['magnetron_id']}"
        ET.SubElement(node, "powerType").text = source["power_type"]
        ET.SubElement(node, "description").text = f"Flange {flange}"
        ET.SubElement(node, "comment")
        if source["manual_only"]:
            ET.SubElement(node, "isManualOnly").text = "true"
        ET.SubElement(node, "installedDate").text = "2026-01-01"
        ET.SubElement(node, "totalDepositionTime").text = "00:00:00"
    return root


def _render_alarms(manifest):
    root = ET.Element("alarms")
    for chamber, entry in manifest["alarms"].items():
        node = ET.SubElement(root, "chamber", id=chamber)
        states = ET.SubElement(node, "states", initial="startup")
        transitions = entry["transitions"]
        for t in transitions["sensor"]:
            ET.SubElement(states, "transition", {"from": t["from"], "to": t["to"], "sensor": t["sensor"],
                                                 "cond": t["cond"], "val": repr(t["val"])})
        for t in transitions["valve"]:
            attrs = {"from": t["from"], "to": t["to"], "valve": t["valve"], "state": t["state"]}
            if t["require"]:
                attrs["require"] = t["require"]
            ET.SubElement(states, "transition", attrs)
        for t in transitions["group"]:
            ET.SubElement(states, "transition", {"from": t["from"], "to": t["to"], "group": t["group"], "state": t["state"]})
        for rule in entry["rules"]:
            attrs = {"sensor": rule["sensor"], "cond": "more", "val": repr(rule["value"]), "state": rule["state"]}
            if rule["delay_s"]:
                attrs["delay"] = f"{rule['delay_s']:g}"
            if rule["unless"]:
                attrs["unless"] = rule["unless"]
            ET.SubElement(node, "alarm", attrs).text = rule["message"]
        for rule in entry["deviceStatusRules"]:
            ET.SubElement(node, "alarm", device=rule["device"], status="alarm", state=rule["state"]).text = rule["message"]
    return root


def _interlock_block(root, interlock_id, lines):
    block = ET.SubElement(root, "interlock", id=interlock_id)
    for line in lines:
        ET.SubElement(block, "l").text = line


def _render_interlocks(manifest):
    ilk = manifest["interlocks"]
    main = ET.Element("interlocks")
    for vent, prereq in ilk["vent"].items():
        _interlock_block(main, f"Valve_{vent}_open", [f"Valve {v} closed" for v in prereq])
    for group in ("pump_valves", "gate", "slit"):
        for valve, lines in ilk[group].items():
            _interlock_block(main, f"Valve_{valve}_open", lines)
    for pump, lines in ilk["pump_off"].items():
        _interlock_block(main, f"Pump_{pump}_off", lines)
    power = ET.Element("interlocks")
    for psu, rule in ilk["power"].items():
        _interlock_block(power, f"power_{psu}_on", [f"Sensor {rule['sensor']} less {rule['threshold_torr']}"])
    index = ET.Element("interlocks")
    includes = ET.SubElement(index, "includes")
    for name in ("Interlocks.Main.xml", "Interlocks.Power.xml"):
        ET.SubElement(includes, "file").text = name
    return index, main, power


def write_topology_files(manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Interlocks").mkdir(exist_ok=True)
    index, main, power = _render_interlocks(manifest)
    files = {
        "HWConf.xml": _render_hwconf(manifest, False),
        "HWConf.Emulator.xml": _render_hwconf(manifest, True),
        "Sources.xml": _render_sources(manifest),
        "Worker.xml": _render_worker(manifest),
        "Alarms.xml": _render_alarms(manifest),
        "Interlocks.xml": index,
        "Interlocks/Interlocks.Main.xml": main,
        "Interlocks/Interlocks.Power.xml": power,
    }
    for name, element in files.items():
        (out_dir / name).write_text(_indent_tostring(element), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_topology(seed: int, family: Literal["single_sputter", "cluster"], out_dir: Path) -> dict:
    manifest = build_manifest(seed, family)
    write_topology_files(manifest, Path(out_dir))
    return manifest


def generate_all(root: Path = TOPOLOGY_ROOT) -> dict:
    manifests = {}
    for split, seeds in (("train", TRAIN_SEEDS), ("eval", EVAL_SEEDS)):
        for seed in seeds:
            manifests[seed] = generate_topology(seed, family_for_seed(seed), Path(root) / split / f"topo_{seed}")
    return manifests


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic PVD topologies (train topo_101..125 and eval topo_201..205).")
    parser.add_argument("root", nargs="?", type=Path, default=TOPOLOGY_ROOT, help="Output directory for generated topologies")
    args = parser.parse_args(argv)
    manifests = generate_all(args.root)
    print(json.dumps({"root": str(args.root), "topologies": len(manifests)}))


if __name__ == "__main__":
    main(sys.argv[1:])
