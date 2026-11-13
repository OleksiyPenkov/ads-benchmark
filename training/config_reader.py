import xml.etree.ElementTree as ET
from pathlib import Path

import topology_mutator as tm

CONTAINER_KINDS = {container: kind for kind, (container, _) in tm.KIND_TAGS.items()}
NON_ATTR = {"descr", "chamber"}


def _psu_driver(hwtype: str, dev_id: str, attrs: dict) -> str:
    base = (hwtype or "").split(".")[-1]
    if base == "LanPowerSupply":
        mode = (attrs.get("mode") or "DC").upper()
        if mode == "DC" and dev_id.upper().startswith("ION"):
            return "LanPowerSupply_ION"
        return f"LanPowerSupply_{mode}"
    return base


def _hardware(root):
    devices = {}
    mfcs = {}
    psus = {}
    for container in root:
        kind = CONTAINER_KINDS.get(container.tag)
        if kind is None:
            continue
        item_tag = tm.KIND_TAGS[kind][1]
        hwtype = container.get("hwtype")
        for item in container.findall(item_tag):
            dev_id = (item.text or "").strip()
            attrs = {k: v for k, v in item.attrib.items() if k not in NON_ATTR}
            devices.setdefault(kind, []).append({
                "id": dev_id, "description": item.get("descr", ""), "chamber": item.get("chamber"),
                "hwtype": hwtype, "host": container.get("host"), "attrs": attrs,
            })
            if kind == "MFC":
                gates = [v for v in item.get("valve", "").split(",") if v]
                mfcs[dev_id] = {"gas": None, "chamber": item.get("chamber"), "gate_valves": gates}
            if kind == "Power":
                driver = _psu_driver(hwtype, dev_id, attrs)
                psus[dev_id] = {"driver": driver, "set_params": list(tm.DRIVERS.get(driver, {}).get("params", []))}
    return devices, mfcs, psus


def _sources(root):
    sources = []
    for node in root.findall("source"):
        material = (node.findtext("material") or "").strip()
        psu = (node.findtext("powerSupplyId") or "").strip()
        name = node.get("name") or material
        if not name and not psu:
            continue
        sources.append({
            "flange": int(node.get("flange")), "name": name, "material": material, "material_text": material,
            "chamber": node.get("chamber"), "psu": psu or None, "driver": None,
            "power_type": (node.findtext("powerType") or "").strip(),
            "magnetron_id": (node.findtext("magnetronId") or "").strip(),
            "manual_only": (node.findtext("isManualOnly") or "").strip() == "true",
            "inlet_valve": None, "shutter": node.get("shutterId"),
        })
    return sources


def _worker(root):
    ops = {op.get("method"): op for op in root.iter("op")}
    pwait = ops.get("Pwait")
    deposit = ops.get("Deposit")
    if pwait is None:
        pwait_info = {"bound": False, "device": None, "args": None}
    else:
        bound = pwait.get("device") is not None
        pwait_info = {"bound": bound, "device": pwait.get("device"), "args": 2 if bound else 3}
    deposit_info = {"args": None if deposit is None else (3 if len(deposit.findall("var")) == 4 else 2)}
    return pwait_info, deposit_info


def validator_manifest(config_dir):
    config_dir = Path(config_dir)
    devices, mfcs, psus = _hardware(ET.parse(config_dir / "HWConf.xml").getroot())
    sources = _sources(ET.parse(config_dir / "Sources.xml").getroot())
    driver_of = {psu: info["driver"] for psu, info in psus.items()}
    for source in sources:
        source["driver"] = driver_of.get(source["psu"])
    pwait, deposit = _worker(ET.parse(config_dir / "Worker.xml").getroot())
    rotation = next((d for d in devices.get("Stage", []) if d["id"] == "rotation"), None)
    return {
        "profile": config_dir.name, "devices": devices, "mfcs": mfcs, "psus": psus, "sources": sources,
        "pwait": pwait, "deposit": deposit,
        "rotation": {"present": rotation is not None,
                     "operational": rotation is not None and rotation["attrs"].get("operational") != "false"},
        "gauge": {"floor_torr": tm.GAUGE_FLOOR_TORR},
    }
