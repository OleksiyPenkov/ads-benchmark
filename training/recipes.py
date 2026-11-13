import re
from dataclasses import dataclass
from xml.sax.saxutils import escape

PRESSURE_SUFFIX = {"kT": 1e3, "T": 1.0, "mT": 1e-3, "uT": 1e-6, "nT": 1e-9}
PRESSURE_RE = re.compile(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)(kT|mT|uT|nT|T)?$")
PULSED_DRIVERS = ("LanPowerSupply_PDC",)
RF_DRIVERS = ("LanPowerSupply_RF",)
BAD_RETRACT_TOKENS = ("false", "none", "off", "shut")


def parse_pressure(token):
    match = PRESSURE_RE.match(token.strip())
    if not match:
        return None, False
    value = float(match.group(1))
    suffix = match.group(2)
    return value * PRESSURE_SUFFIX.get(suffix, 1.0), suffix is not None


@dataclass
class Param:
    name: str
    value: str
    unit: str
    description: str
    kind: int
    group: str


@dataclass
class Recipe:
    title: str
    notes: str
    params: list
    tagged: list

    def param(self, name):
        return next(p for p in self.params if p.name == name)

    @property
    def body(self):
        return [line for _, line in self.tagged]

    def script_lines(self):
        return self.body

    def to_rcpx(self):
        out = [
            '<?xml version="1.0" encoding="utf-8"?>',
            "<Recipe>",
            f"  <Title>{escape(self.title)}</Title>",
            f"  <Notes>{escape(self.notes)}</Notes>",
            "  <Parameters>",
        ]
        for p in self.params:
            out.append(
                f'    <Param Name="{escape(p.name)}" Value="{escape(p.value)}" Unit="{escape(p.unit)}" '
                f'Description="{escape(p.description)}" Type="{p.kind}" Group="{escape(p.group)}" />'
            )
        out.append("  </Parameters>")
        out.append("  <Script>" + escape("\n".join(self.body)) + "</Script>")
        out.append("</Recipe>")
        return "\n".join(out)


def parse_rcpx(text):
    params = [
        Param(m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5)), m.group(6))
        for m in re.finditer(
            r'<Param\s+Name="([^"]*)"\s+Value="([^"]*)"\s+Unit="([^"]*)"\s+Description="([^"]*)"\s+Type="(\d+)"\s+Group="([^"]*)"\s*/>',
            text,
        )
    ]
    script = re.search(r"<Script>(.*?)</Script>", text, re.DOTALL)
    if not script:
        return None
    return Recipe("", "", params, [("", line) for line in script.group(1).splitlines()])


def _chamber_info(manifest, chamber):
    gas = next(g for g in manifest["gases"] if g["gas"] == "Ar" and g["chamber"] == chamber)
    sensor = next(c["sensor"] for c in manifest["chambers"] if c["id"] == chamber)
    return gas, sensor


def good_recipe(manifest, spec):
    chamber = spec["chamber"]
    cluster = manifest["family"] == "cluster"
    gas, sensor = _chamber_info(manifest, chamber)
    heavy, light = spec["heavy"], spec["light"]
    heavy_name, light_name = heavy["name"], light["name"]
    pwait_prefix = "" if manifest["pwait"]["bound"] else f"{sensor} "
    retract = " Home" if manifest["deposit"]["args"] == 3 else ""
    params = [
        Param("P", f"{spec['pressure_mT']:.2f}mT", "mTorr", "Pressure", 0, "Deposition Conditions"),
        Param("ArFlow", f"{spec['flow_sccm']:g}", "sccm", "Ar flow per line", 1, "Deposition Conditions"),
        Param("N", str(spec["n"]), "N", "Bilayers", 1, "Deposition Conditions"),
    ]
    if not cluster:
        params.insert(2, Param("StageHeight", "0", "mm", "Height", 1, "Deposition Conditions"))
    sources = [heavy, light]
    if any(s["driver"] in PULSED_DRIVERS for s in sources):
        params += [Param("DCFreq", f"{spec['freq_kHz']:g}", "kHz", "Freq", 1, "Pulsed DC"),
                   Param("DCDuty", f"{spec['duty']:g}", "%", "Duty", 1, "Pulsed DC")]
    for source in sources:
        label = f"Layer {source['name']}"
        if source["driver"] in RF_DRIVERS:
            params.append(Param(f"pwr_{source['name']}", f"{spec['setpoints'][source['name']]:g}", "W", "Power", 1, label))
        else:
            params.append(Param(f"cur_{source['name']}", f"{spec['setpoints'][source['name']]:g}", "mA", "Current", 1, label))
        params.append(Param(f"time_{source['name']}", f"{spec['dwell'][source['name']]:g}", "sec", "Time per period", 1, label))
    if spec.get("adhesion_s"):
        params.append(Param(f"time_{heavy_name}_adh", f"{spec['adhesion_s']:g}", "sec", "Time (adhesion)", 1,
                            f"Layer {heavy_name}"))

    lines = []
    if not cluster:
        lines.append(("stage", "stage VS StageHeight"))
        if manifest["rotation"]["operational"] and spec.get("rotate"):
            lines.append(("rotate", f"rotate {spec['rotate']}"))
    lines.append(("comment", "# --- process gas ---"))
    lines.append((f"gas_open:{gas['valves'][0]}", f"valve {gas['valves'][0]} open"))
    for source in sources:
        if source["inlet_valve"]:
            lines.append((f"inlet_open:{source['inlet_valve']}", f"valve {source['inlet_valve']} open"))
    for mfc in gas["mfcs"]:
        lines.append((f"mfc_on:{mfc}", f"mfc {mfc} ArFlow"))
    lines.append(("pwait_P", f"pwait {pwait_prefix}P 600"))
    lines.append(("comment", "# --- supplies ---"))
    for source in sources:
        psu = source["psu"]
        if source["driver"] in PULSED_DRIVERS:
            lines.append((f"set:{psu}:Frequency", f"set {psu} Frequency DCFreq"))
            lines.append((f"set:{psu}:DutyCycle", f"set {psu} DutyCycle DCDuty"))
            lines.append((f"set:{psu}:Current", f"set {psu} Current cur_{source['name']}"))
        elif source["driver"] in RF_DRIVERS:
            lines.append((f"set:{psu}:Power", f"set {psu} Power pwr_{source['name']}"))
        else:
            lines.append((f"set:{psu}:Current", f"set {psu} Current cur_{source['name']}"))
    for source in sources:
        lines.append((f"power_on:{source['psu']}", f"power {source['psu']} on"))
    lines.append(("comment", "# --- layers ---"))
    if spec.get("adhesion_s"):
        lines.append((f"dep:{heavy_name}:adh", f"deposit {heavy_name} time_{heavy_name}_adh{retract}"))
    lines.append(("loop", "loop N"))
    lines.append((f"dep:{light_name}", f"deposit {light_name} time_{light_name}{retract}"))
    lines.append((f"dep:{heavy_name}", f"deposit {heavy_name} time_{heavy_name}{retract}"))
    lines.append(("endloop", "endloop"))
    lines.append(("comment", "# --- shut down ---"))
    for source in sources:
        lines.append((f"power_off:{source['psu']}", f"power {source['psu']} off"))
    if manifest["pwait"]["bound"]:
        lines.append(("release", "pwait 0 1"))
    for mfc in gas["mfcs"]:
        lines.append((f"mfc_off:{mfc}", f"mfc {mfc} 0"))
    for source in sources:
        if source["inlet_valve"]:
            lines.append((f"inlet_close:{source['inlet_valve']}", f"valve {source['inlet_valve']} close"))
    lines.append((f"gas_close:{gas['valves'][0]}", f"valve {gas['valves'][0]} close"))
    if not cluster:
        lines.append(("holder", "holder Home"))
    title = f"{heavy_name}/{light_name} multilayer"
    return Recipe(title, f"{heavy_name} fraction {spec['gamma']:g}; plain Ar; planned layer times", params, lines)


def applicable_defects(manifest, recipe):
    kinds = ["message", "warmup_no_power", "unrouted_mfc", "close_without_open", "set_param", "unknown_source",
             "bare_pressure"]
    if manifest["deposit"]["args"] == 3:
        kinds.append("retract")
    if manifest["rotation"]["present"] and not manifest["rotation"]["operational"] and manifest["family"] != "cluster":
        kinds.append("rotate")
    if manifest["pwait"]["bound"]:
        kinds += ["release_floor", "pwait_extra_sensor"]
    else:
        kinds.append("pwait_arity")
    if not any(tag.startswith("inlet_open:") for tag, _ in recipe.tagged):
        kinds.remove("close_without_open")
    return kinds


def _index(recipe, prefix):
    return next(i for i, (tag, _) in enumerate(recipe.tagged) if tag.startswith(prefix))


def inject_defect(recipe, kind, manifest, rng):
    tagged = recipe.tagged
    if kind == "message":
        at = _index(recipe, "dep:")
        tagged.insert(at, ("message", "message Check that both discharges are lit, then press OK"))
    elif kind == "retract":
        at = _index(recipe, "dep:")
        text = tagged[at][1].rsplit(" ", 1)[0] + f" {rng.choice(BAD_RETRACT_TOKENS)}"
        tagged[at] = (tagged[at][0], text)
    elif kind == "rotate":
        tagged.insert(_index(recipe, "comment"), ("rotate", "rotate 20"))
    elif kind == "warmup_no_power":
        tagged[_index(recipe, "power_on:")] = ("warmup", "warmup 60 10")
    elif kind == "unrouted_mfc":
        del tagged[_index(recipe, "gas_open:")]
        del tagged[_index(recipe, "gas_close:")]
    elif kind == "close_without_open":
        at = _index(recipe, "inlet_open:")
        del tagged[at]
    elif kind == "release_floor":
        at = _index(recipe, "release")
        tagged[at] = ("release", "pwait Pbase 10")
        recipe.params.append(Param("Pbase", "0.01mT", "mTorr", "Release pressure", 0, "Deposition Conditions"))
    elif kind == "set_param":
        at = _index(recipe, "set:")
        text = tagged[at][1]
        parts = text.split(" ")
        parts[2] = "Set" + parts[2]
        tagged[at] = (tagged[at][0], " ".join(parts))
    elif kind == "unknown_source":
        at = _index(recipe, "dep:")
        parts = tagged[at][1].split(" ")
        parts[1] = rng.choice(("Gr", "Carbon", "Tungsten"))
        tagged[at] = (tagged[at][0], " ".join(parts))
    elif kind == "bare_pressure":
        recipe.param("P").value = recipe.param("P").value.replace("mT", "")
    elif kind == "pwait_extra_sensor":
        at = _index(recipe, "pwait_P")
        sensor = next(c["sensor"] for c in manifest["chambers"] if c["kind"] == "process")
        tagged[at] = (tagged[at][0], f"pwait {sensor} P 600")
    elif kind == "pwait_arity":
        at = _index(recipe, "pwait_P")
        tagged[at] = (tagged[at][0], "pwait P 600")
    else:
        raise ValueError(kind)
    return recipe
