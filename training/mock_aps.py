import json
import math
import random
import uuid

import recipes as rc
from tool_schemas import MUTATING_TOOLS

HOST_COMMANDS = (
    "mutating commands (recipe.upload/start/abort/resolve_pause, device.command) need controlMode == \"Remote\" "
    "(Online-Remote); only an operator can switch it, under Service \u2192 Control mode\u2026 \u2014 do not poll for it"
)
TIER_RANK = {"Read": 0, "Recipe": 1, "Manual": 2}
ALLOWED_MUTATING = {
    "Read": [],
    "Recipe": ["upload_recipe", "start_recipe", "abort_recipe", "resolve_pause"],
    "Manual": ["upload_recipe", "start_recipe", "abort_recipe", "resolve_pause", "device_command"],
}
TOOL_MIN_TIER = {"validate_recipe": "Recipe", "upload_recipe": "Recipe", "start_recipe": "Recipe",
                 "abort_recipe": "Recipe", "resolve_pause": "Recipe", "device_command": "Manual"}
COMMANDS = ["var", "loop", "endloop", "function", "endfunction", "if", "else", "endif", "label", "goto", "begin", "end",
            "delay", "wait", "pwait", "swait", "check", "rotation", "call", "state", "valve", "mfc", "pump", "stage",
            "power", "ramp", "set", "warmup", "holder", "shutter", "deposit",
            "rotate", "message", "start", "finish", "isolate", "transfer", "home", "abort"]
KNOWN_COMMANDS = {c.lower() for c in COMMANDS}
GROUP_NAMES = {"gas": "Gas", "gasvalve": "GasValve"}
GROUP_CONDITIONS = {"allOff": "AllOff", "anyOpen": "AnyOpen"}
BLOCKING_MESSAGE = ("'message' opens a blocking operator dialog on the Control UI that halts unattended execution "
                    "and survives abort_recipe")


def error(code, message):
    return {"error": {"code": code, "message": message}}


def compute_plan(args):
    required = ("period_A", "gamma", "heavy_rate_A_per_s", "light_rate_A_per_s", "bilayer_count")
    for key in required:
        if key not in args:
            return error("bad_request", f"missing required argument '{key}'")
    try:
        period = float(args["period_A"])
        gamma = float(args["gamma"])
        heavy_rate = float(args["heavy_rate_A_per_s"])
        light_rate = float(args["light_rate_A_per_s"])
        count = int(args["bilayer_count"])
    except (TypeError, ValueError):
        return error("bad_request", "arguments must be numbers")
    if not all(math.isfinite(v) for v in (period, gamma, heavy_rate, light_rate)):
        return error("bad_request", "arguments must be finite")
    if period <= 0 or heavy_rate <= 0 or light_rate <= 0:
        return error("bad_request", "period_A and both rates must be greater than zero")
    if not 0 < gamma < 1:
        return error("bad_request", "gamma must satisfy 0 < gamma < 1")
    if count < 1:
        return error("bad_request", "bilayer_count must be at least 1")
    heavy_t, light_t = gamma * period, (1 - gamma) * period
    heavy_exact, light_exact = heavy_t / heavy_rate, light_t / light_rate
    heavy_s, light_s = round(heavy_exact, 2), round(light_exact, 2)
    heavy_int, light_int = round(heavy_exact), round(light_exact)
    result = {
        "measurementStatus": "planned_from_rate_and_dwell_not_measured",
        "period_A": period, "gamma": gamma, "bilayer_count": count,
        "heavy_thickness_A": round(heavy_t, 6), "light_thickness_A": round(light_t, 6),
        "heavy_dwell_exact_s": round(heavy_exact, 6), "light_dwell_exact_s": round(light_exact, 6),
        "heavy_dwell_s": heavy_s, "light_dwell_s": light_s,
        "period_from_rounded_dwells_A": round(heavy_s * heavy_rate + light_s * light_rate, 6),
        "integer_rounded_check": {
            "heavy_dwell_int_s": heavy_int, "light_dwell_int_s": light_int,
            "period_from_integer_dwells_A": round(heavy_int * heavy_rate + light_int * light_rate, 2),
        },
    }
    adhesion = args.get("adhesion_thickness_A")
    if adhesion is not None:
        adhesion_rate = args.get("adhesion_rate_A_per_s")
        if float(adhesion) < 0 or (float(adhesion) > 0 and (adhesion_rate is None or float(adhesion_rate) <= 0)):
            return error("bad_request", "adhesion_rate_A_per_s must be greater than zero when adhesion_thickness_A is given")
        result["adhesion_dwell_s"] = round(float(adhesion) / float(adhesion_rate), 2) if float(adhesion) > 0 else 0.0
    dead_time = args.get("dead_time_s")
    if dead_time is not None:
        dead_rate = args.get("dead_time_rate_A_per_s")
        if float(dead_time) < 0 or dead_rate is None or float(dead_rate) <= 0:
            return error("bad_request", "dead_time_rate_A_per_s must be greater than zero when dead_time_s is given")
        result["dead_time_heavy_thickness_A"] = round(float(dead_rate) * (heavy_s - float(dead_time)), 2)
    return result


def _installed(manifest):
    return [s["name"] for s in manifest["sources"]]


def _device_ids(manifest, kind):
    return {d["id"] for d in manifest["devices"].get(kind, [])}


def _col(line, token):
    at = line.find(token)
    return (at if at >= 0 else 0) + 1


def validate_rcpx(manifest, content):
    recipe = rc.parse_rcpx(content)
    if recipe is None:
        return {"valid": False, "findings": [{"line": 1, "column": 1, "severity": "error",
                                              "message": "Recipe XML has no <Script> block"}]}
    findings = []

    def add(line, column, severity, message):
        findings.append({"line": line, "column": column, "severity": severity, "message": message})

    declared = {p.name: p.value for p in recipe.params}
    valves = _device_ids(manifest, "Valve")
    mfcs = manifest["mfcs"]
    sensors = _device_ids(manifest, "Sensor")
    psus = manifest["psus"]
    installed = _installed(manifest)
    source_psu = {s["name"]: s["psu"] for s in manifest["sources"]}
    bound = manifest["pwait"]["bound"]
    energized, currently_open, ever_open, flows, reported = set(), set(), set(), {}, set()
    last_pwait = None

    def resolve(token):
        return declared.get(token, token)

    def check_routing(line):
        for mfc_id, flow in flows.items():
            gates = mfcs[mfc_id]["gate_valves"]
            if flow > 0 and gates and mfc_id not in reported and not any(g.lower() in currently_open for g in gates):
                reported.add(mfc_id)
                add(line, 1, "warning",
                    f"mfc '{mfc_id}' has non-zero flow but none of its inlet gate valves ({', '.join(gates)}) are open")

    for number, raw in enumerate(recipe.script_lines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        tokens = text.split()
        cmd, args = tokens[0].lower(), tokens[1:]
        if cmd not in KNOWN_COMMANDS:
            add(number, 1, "error", f"Unknown command '{tokens[0]}'")
        elif cmd == "var" and args:
            declared[args[0]] = " ".join(args[1:])
        elif cmd == "valve" and len(args) >= 2:
            valve, action = args[0], resolve(args[1]).lower()
            if valve not in valves:
                add(number, _col(raw, valve), "error", f"Unknown valve '{valve}'")
            elif action in ("open", "true"):
                currently_open.add(valve.lower())
                ever_open.add(valve.lower())
            elif action in ("close", "false"):
                if valve.lower() not in ever_open:
                    add(number, 1, "warning", f"valve '{valve}' is closed but was never opened in this recipe")
                currently_open.discard(valve.lower())
        elif cmd == "mfc" and len(args) >= 2:
            if args[0] not in mfcs:
                add(number, _col(raw, args[0]), "error", f"Unknown MFC '{args[0]}'")
            else:
                try:
                    flows[args[0]] = float(resolve(args[1]))
                except ValueError:
                    flows[args[0]] = 0.0
        elif cmd == "set" and len(args) >= 3:
            psu, param = args[0], args[1]
            if psu not in psus:
                add(number, _col(raw, psu), "error", f"Unknown power supply '{psu}'")
            elif param not in psus[psu]["set_params"]:
                add(number, _col(raw, param), "error",
                    f"'set {psu} {param}': unknown parameter '{param}' for {psu} \u2014 the driver would refuse it at run "
                    f"time. Valid (exact spelling): {', '.join(psus[psu]['set_params'])}")
        elif cmd == "power" and len(args) >= 2:
            psu, action = args[0], resolve(args[1]).lower()
            if psu not in psus:
                add(number, _col(raw, psu), "error", f"Unknown power supply '{psu}'")
            elif action in ("on", "true"):
                energized.add(psu.lower())
            elif action in ("off", "false"):
                energized.discard(psu.lower())
        elif cmd == "isolate":
            currently_open.clear()
            energized.clear()
        elif cmd == "pwait":
            expected = manifest["pwait"]["args"]
            if bound and len(args) == 3:
                add(number, _col(raw, args[0]), "warning",
                    f"'pwait' takes 2 argument(s) on this machine (target timeout) \u2014 the throttle controller is its "
                    f"own gauge, so '{args[0]}' is ignored. Remove it: 'pwait {args[1]} {args[2]}'.")
            elif len(args) != expected:
                form = "target timeout" if bound else "sensor target timeout"
                add(number, 1, "error", f"'pwait' takes {expected} argument(s) on this machine ({form})")
            if len(args) >= 2:
                target = resolve(args[-2] if bound or len(args) == 2 else args[1])
                if not bound and len(args) == 3 and args[0] not in sensors:
                    add(number, _col(raw, args[0]), "error", f"Unknown sensor '{args[0]}'")
                torr, has_unit = rc.parse_pressure(target)
                if torr is not None and not has_unit and torr > 0:
                    add(number, _col(raw, args[-2]), "warning",
                        f"pressure target '{target}' is a bare number and is read as Torr ({target} Torr = "
                        f"{float(target) * 1000:g} mTorr); add a unit suffix such as 'mT'")
                last_pwait = (number, target)
            check_routing(number)
        elif cmd == "deposit":
            expected = manifest["deposit"]["args"]
            if len(args) != expected:
                form = "source seconds retract" if expected == 3 else "source seconds"
                add(number, 1, "error", f"'deposit' takes {expected} argument(s) on this machine ({form})")
            if args:
                source = resolve(args[0])
                if source not in installed:
                    add(number, _col(raw, args[0]), "error",
                        f"Unknown target '{source}' for 'deposit'. Installed sources: {', '.join(installed)}")
                elif source_psu.get(source) and source_psu[source].lower() not in energized:
                    add(number, 1, "warning",
                        f"deposit '{source}' runs while its power supply '{source_psu[source]}' is not energized "
                        f"('warmup' does not leave power on; add 'power {source_psu[source]} on' before depositing)")
            if len(args) >= 3:
                retract = resolve(args[2])
                if not retract.startswith("$") and not (
                        retract in ("Home", "0") or retract.lstrip("-").isdigit() or retract in installed):
                    add(number, _col(raw, args[2]), "error",
                        f"Unknown retract position '{retract}' for 'deposit'. Third argument parks the shutter and must be "
                        f"Home, 0, a flange number, or an installed source: {', '.join(sorted(installed))}")
            check_routing(number)
        elif cmd in ("rotate", "rotation") and args and not manifest["rotation"]["operational"]:
            value = resolve(args[0]).lower()
            active = value in ("on", "true") if cmd == "rotation" else value not in ("0", "0.0", "off", "false")
            if active:
                add(number, 1, "warning", f"substrate rotation ('{cmd}') is not operational on this machine")
        elif cmd == "message":
            add(number, 1, "warning", BLOCKING_MESSAGE)
    if bound and last_pwait is not None:
        number, target = last_pwait
        torr, _ = rc.parse_pressure(target)
        floor = manifest["gauge"]["floor_torr"]
        if torr is not None and torr > 0 and torr <= floor:
            add(number, 1, "warning",
                f"final pwait target '{target}' is at or below the gauge floor (0.01 mTorr): the runtime "
                f"opens the throttle fully and then waits for the gauge to read at or below the floor, so "
                f"the step fails with a timeout if the chamber cannot pump that low; use target 0 "
                f"('pwait 0 1') to open the throttle and continue at once")
    findings.sort(key=lambda f: (f["line"], f["column"]))
    return {"valid": not any(f["severity"] == "error" for f in findings), "findings": findings}


def estimate_run_seconds(recipe):
    values = {p.name: p.value for p in recipe.params}
    total, multiplier, stack = 0.0, 1, []
    for line in recipe.body:
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0] == "loop":
            stack.append(multiplier)
            multiplier *= int(float(values.get(tokens[1], tokens[1])))
        elif tokens[0] == "endloop":
            multiplier = stack.pop()
        elif tokens[0] == "deposit":
            total += multiplier * float(values.get(tokens[2], tokens[2]))
        else:
            total += multiplier * 20.0
    return total


class MockApsEnvironment:
    def __init__(self, topology_manifest, tier="Read", control_mode="Remote"):
        self.m = topology_manifest
        self.tier = tier
        self.control_mode = control_mode
        self.rng = random.Random(self.m["seed"] * 7919 + 17)
        self.base_pressure = {c["sensor"]: self.rng.choice((2, 3, 4, 5, 6, 7)) * 1e-7 for c in self.m["chambers"]}
        self.pressure_override = {}
        self.open_valves = set()
        self.log_lines = None
        self.uploaded = {}
        self.run = None
        self.clock = 0.0
        self.last_run = None
        self.calls = []
        gate_valves = {c["gate_valve"] for c in self.m["chambers"]}
        self.open_valves |= gate_valves
        self.open_valves |= {d["id"] for d in self.m["devices"]["Valve"] if d["id"].endswith("FL") or d["id"] == "FL"}

    def set_state(self, pressure_torr=None, open_valves=None, closed_valves=()):
        if pressure_torr is not None:
            sensor = self.m["chambers"][0]["sensor"]
            self.pressure_override[sensor] = pressure_torr
        if open_valves is not None:
            self.open_valves |= set(open_valves)
        self.open_valves -= set(closed_valves)

    def call(self, name, arguments=None):
        arguments = arguments or {}
        self.calls.append(name)
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return error("unknown_tool", f"no such tool '{name}'")
        if name in TOOL_MIN_TIER and TIER_RANK[self.tier] < TIER_RANK[TOOL_MIN_TIER[name]]:
            return error("tier_denied", f"'{name}' needs a {TOOL_MIN_TIER[name]}-tier key; this session's key is {self.tier}")
        if name in MUTATING_TOOLS and self.control_mode != "Remote":
            return error("control_mode", f"'{name}' needs controlMode Remote; Control is in {self.control_mode}")
        return handler(**arguments)

    def call_json(self, name, arguments=None):
        return json.dumps(self.call(name, arguments), separators=(",", ":"), ensure_ascii=False)

    def _describe_machine(self):
        mutating = ALLOWED_MUTATING[self.tier]
        return {
            "profile": self.m["profile"], "appVersion": f"1.7.0.{1400 + self.m['seed'] % 89}",
            "machineName": f"SYN{self.m['seed']}", "chambers": [c["id"] for c in self.m["chambers"]], "protocol": 1,
            "controlMode": self.control_mode, "hostCommands": HOST_COMMANDS, "sessionTier": self.tier,
            "allowedMutatingTools": list(mutating),
            "canExecuteMutatingNow": TIER_RANK[self.tier] >= TIER_RANK["Recipe"] and self.control_mode == "Remote",
        }

    def _describe_hardware(self):
        devices = {}
        for kind, items in self.m["devices"].items():
            entries = []
            for d in items:
                entry = {"id": d["id"], "plcId": d["id"], "description": d["description"], "connected": True,
                         "chamber": d["chamber"]}
                if kind == "MFC":
                    gates = self.m["mfcs"][d["id"]]["gate_valves"]
                    entry["gateValves"] = list(gates)
                    entry["gatingLogic"] = "OR" if gates else "none"
                if kind == "Stage":
                    operational = not (d["id"] == "rotation" and not self.m["rotation"]["operational"])
                    entry["isOperational"] = operational
                entries.append(entry)
            devices[kind] = entries
        flanges = []
        for s in sorted(self.m["sources"], key=lambda s: s["flange"]):
            is_ion = s["material"] == "Ion"
            flanges.append({
                "material": s["material_text"], "powerSupplyId": s["psu"], "powerType": s["power_type"],
                "magnetronInstanceId": None if is_ion else str(uuid.uuid5(uuid.NAMESPACE_OID, f"{self.m['seed']}-{s['flange']}")),
                "magnetronId": s["magnetron_id"], "name": None if is_ion else s["name"], "effective": s["name"],
            })
        return {"devices": devices, "sources": {"flanges": flanges, "powerSources": []}}

    def _ops(self):
        bound, deposit3 = self.m["pwait"]["bound"], self.m["deposit"]["args"] == 3
        ops = [
            {"name": "mfc", "boundDevice": None, "vars": ["MFC", "f"]},
            {"name": "valve", "boundDevice": None, "vars": ["Valve", "b"]},
            {"name": "pwait", "boundDevice": self.m["pwait"]["device"], "vars": ["Controller", "s", "f", "f"] if bound else ["s", "f", "f"]},
            {"name": "set", "boundDevice": None, "vars": ["Power", "s", "f"]},
            {"name": "power", "boundDevice": None, "vars": ["Power", "b"]},
            {"name": "deposit", "boundDevice": "Holder" if deposit3 else None,
             "vars": ["Drive", "s", "f", "s"] if deposit3 else ["s", "f"]},
        ]
        if self.m["family"] == "single_sputter":
            ops += [{"name": "warmup", "boundDevice": None, "vars": ["Power", "f", "f"]},
                    {"name": "holder", "boundDevice": "Holder", "vars": ["Drive", "s"]}]
        else:
            ops += [{"name": "transfer", "boundDevice": "ROBOT", "vars": ["Robot", "s", "s"]},
                    {"name": "home", "boundDevice": "ROBOT", "vars": ["Robot"]}]
        return ops

    def _describe_dsl(self):
        drivers = {}
        for info in self.m["psus"].values():
            drivers.setdefault(info["driver"], info["set_params"])
        driver_lines = "\r\n".join(f"                                   # {d}: {', '.join(p)}" for d, p in drivers.items())
        if self.m["pwait"]["bound"]:
            pwait_note = (f"canonical form is 2 args (target timeout); this machine binds {self.m['pwait']['device']}; a "
                          f"leading <sensor> validates but is IGNORED. Target 0 = open the throttle and return at once")
        else:
            pwait_note = "pwait <sensor> <target> <timeoutSec> takes 3 args; the sensor is required and is the one waited on"
        deposit_note = ("deposit <source> <seconds> <retract>; <retract> = where the SHUTTER parks afterwards (Home = closed; "
                        "Home, 0, a flange number or an installed source)"
                        if self.m["deposit"]["args"] == 3 else "deposit <source> <seconds>; takes no retract argument")
        reference = (
            "APS recipe DSL \u2014 one command per line, '#' comments.\r\nCore commands:\r\n"
            "  valve <id> open|close            MFC <id> <sccm>          pump <id> on|off\r\n"
            f"  pwait: arity is WORKER-DERIVED. {pwait_note}. Target ALWAYS carries a unit suffix.\r\n"
            f"  {deposit_note}\r\n"
            "  set <psu> <Param> <value>        # <Param> must be a name the PSU's driver accepts (exact spelling):\r\n"
            f"{driver_lines}\r\n"
            "Pressure units: Torr with suffixes T/mT/uT/nT/kT. Numbers with suffixes are single tokens.\r\n"
            "PSU rule: setting a power level does NOT energize the output \u2014 set the level AND `power <psu> on`.\r\n"
            "Control mode: upload_recipe, start_recipe, abort_recipe, resolve_pause and device_command run ONLY while "
            "Control is in Online-Remote.")
        by_kind = {kind: [d["id"] for d in items] for kind, items in self.m["devices"].items()}
        return {"reference": reference, "commands": COMMANDS, "operations": self._ops(), "devicesByKind": by_kind,
                "installedSources": _installed(self.m)}

    @staticmethod
    def _condition(line):
        tokens = line.split()
        kind, device = tokens[0], tokens[1]
        if kind == "Sensor":
            return {"device": device, "value": float(tokens[3]), "condition": -1 if tokens[2] == "less" else 1,
                    "stringCondition": None}
        return {"device": device, "value": tokens[2] in ("open", "on"), "condition": 0, "stringCondition": None}

    def _describe_interlocks(self):
        ilk = self.m["interlocks"]
        out = []

        def add(kind, device, verb, lines):
            flag = "true" if verb in ("open", "on") else "false"
            out.append({"id": f"{kind}_{device.lower()}_{flag}", "conditions": [self._condition(x) for x in lines]})

        for vent, prereq in ilk["vent"].items():
            add("valve", vent, "open", [f"Valve {v} closed" for v in prereq])
        for group in ("pump_valves", "gate", "slit"):
            for valve, lines in ilk[group].items():
                add("valve", valve, "open", lines)
        for pump, lines in ilk["pump_off"].items():
            add("pump", pump, "off", lines)
        for psu, rule in ilk["power"].items():
            add("power", psu, "on", [f"Sensor {rule['sensor']} less {rule['threshold_torr']}"])
        return out

    def _describe_alarms(self):
        out = []
        for chamber, entry in self.m["alarms"].items():
            rules = [{"sensor": r["sensor"], "value": r["value"], "condition": r["condition"], "state": r["state"].capitalize(),
                      "message": r["message"], "delaySeconds": r["delay_s"], "unless": r["unless"], "unit": "Torr"}
                     for r in entry["rules"]]
            transitions = entry["transitions"]
            out.append({
                "chamber": chamber, "rules": rules,
                "deviceStatusRules": [{**d, "state": d["state"].capitalize()} for d in entry["deviceStatusRules"]],
                "transitions": {
                    "sensor": [{"from": t["from"].capitalize(), "to": t["to"].capitalize(), "sensor": t["sensor"],
                                "condition": -1 if t["cond"] == "less" else 1, "value": t["val"]} for t in transitions["sensor"]],
                    "valve": [{"from": t["from"].capitalize(), "to": t["to"].capitalize(), "valve": t["valve"],
                               "open": t["state"] == "open", "requirements": [t["require"]] if t["require"] else []}
                              for t in transitions["valve"]],
                    "group": [{"from": t["from"].capitalize(), "to": t["to"].capitalize(), "group": GROUP_NAMES[t["group"]],
                               "condition": GROUP_CONDITIONS[t["state"]]} for t in transitions["group"]],
                },
            })
        return out

    def _pressure(self, sensor):
        return self.pressure_override.get(sensor, self.base_pressure.get(sensor, 3e-7))

    def _system_status(self):
        running = self.run is not None and not self._run_ended()
        return {
            "chambers": [{"id": c["id"], "state": "Processing" if running and c["kind"] == "process" else "Idle"}
                         for c in self.m["chambers"]],
            "activeAlarms": [], "isRecipeRunning": running, "isDepositionActive": False,
            "controlMode": self.control_mode, "recipe": self.run["name"] if running else None,
        }

    def _device_snapshot(self):
        devices = self.m["devices"]
        snapshot = {
            "valves": [{"id": d["id"], "open": d["id"] in self.open_valves, "connected": True} for d in devices["Valve"]],
            "pumps": [{"id": d["id"], "on": True, "connected": True} for d in devices.get("Pump", [])],
            "mfcs": [{"id": d["id"], "flow": 0.0, "setpoint": 0.0, "connected": True} for d in devices.get("MFC", [])],
            "sensors": [{"id": d["id"], "value": self._pressure(d["id"]) if d["id"] in self.base_pressure else 24.0,
                         "unit": "Torr" if d["id"] in self.base_pressure else "C", "connected": True}
                        for d in devices["Sensor"]],
            "powerSupplies": [{"id": d["id"], "outputOn": False, "setpoint": 0.0, "connected": True}
                              for d in devices.get("Power", [])],
        }
        return snapshot

    def _validate_recipe(self, content, isRcpx=True):
        return validate_rcpx(self.m, content)

    def _compute_deposition_plan(self, **args):
        return compute_plan(args)

    def _describe_recipes(self):
        return {"user": [{"name": n} for n in sorted(self.uploaded)], "embedded": [{"name": "SYSTEM_AIR_RELEASE"}]}

    def _upload_recipe(self, name, content, overwrite=False):
        result = validate_rcpx(self.m, content)
        if not result["valid"]:
            return {"error": {"code": "invalid_recipe", "message": "recipe has errors", "findings": result["findings"]}}
        if name in self.uploaded and not overwrite:
            return error("already_exists", f"recipe '{name}' already exists; pass overwrite=true to replace it")
        self.uploaded[name] = content
        warnings = sum(1 for f in result["findings"] if f["severity"] == "warning")
        return {"saved": name, "path": f"configs/{self.m['profile']}/Recipes/{name}.rcpx",
                "warnings": warnings}

    def _start_recipe(self, name, parametersJson=None):
        if name not in self.uploaded and name != "SYSTEM_AIR_RELEASE":
            return error("not_found", f"no recipe named '{name}'")
        if self.run is not None and not self._run_ended():
            return error("busy", "a recipe is already running")
        recipe = rc.parse_rcpx(self.uploaded[name]) if name in self.uploaded else None
        total = estimate_run_seconds(recipe) if recipe else 90.0
        self.run = {"name": name, "start": self.clock, "total": total, "recipe": recipe}
        return {"accepted": True}

    def _abort_recipe(self):
        if self.run is None or self._run_ended():
            return error("not_running", "no recipe is running")
        self.run["total"] = self.clock - self.run["start"]
        return {"accepted": True}

    def _resolve_pause(self, decision):
        return error("not_paused", "the run is not paused")

    def _device_command(self, deviceType, deviceId, argsJson):
        return {"accepted": True}

    def _run_ended(self):
        return self.clock - self.run["start"] >= self.run["total"]

    @staticmethod
    def _hms(seconds):
        seconds = max(0.0, seconds)
        return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{seconds % 60:07.4f}"

    def _recipe_status(self, waitSeconds=0):
        if self.run is None:
            return {"isRecipeRunning": False, "isUserRecipe": False, "failureProtocolActive": False,
                    "progress": None, "lastOutcome": None, "waitedSeconds": 0, "runEnded": False}
        remaining = self.run["total"] - (self.clock - self.run["start"])
        waited = 0
        if waitSeconds > 0 and remaining > 0:
            waited = min(int(waitSeconds), 300, math.ceil(remaining))
            self.clock += waited
        elapsed = self.clock - self.run["start"]
        ended = elapsed >= self.run["total"]
        body = [line for line in (self.run["recipe"].body if self.run["recipe"] else []) if line.split() and not line.startswith("#")]
        total_steps = max(len(body), 1)
        step = min(total_steps, int(total_steps * min(elapsed / self.run["total"], 1.0)) + 1)
        description = body[step - 1] if body else "SYSTEM_AIR_RELEASE"
        status = {
            "isRecipeRunning": not ended, "isUserRecipe": self.run["recipe"] is not None,
            "failureProtocolActive": False,
            "serverTime": f"2026-11-10T14:{int(self.clock // 60) % 60:02d}:{int(self.clock % 60):02d}",
            "progress": None if ended else {
                "name": self.run["name"], "currentStep": step, "totalSteps": total_steps,
                "stepDescription": description, "state": 1, "elapsed": self._hms(elapsed),
                "estimatedRemaining": self._hms(self.run["total"] - elapsed), "currentStack": 0, "totalStacks": 0},
            "lastOutcome": "Complete" if ended else None, "waitedSeconds": waited, "runEnded": ended,
        }
        if ended:
            self.last_run = dict(self.run, end=self.clock)
        return status

    def _list_runs(self, count=5):
        runs = []
        if self.last_run:
            minutes, seconds = divmod(int(self.last_run["total"]), 60)
            runs.append({
                "runId": f"2026-11-10_14-02-11_{self.last_run['name']}", "recipe": self.last_run["name"],
                "startedAt": "2026-11-10T14:02:11", "endedAt": "2026-11-10T14:44:23", "outcome": "complete",
                "message": f"({minutes:02d}m {seconds:02d}s)",
                "artifacts": [{"kind": "log", "fileName": f"{self.last_run['name']}.log", "bytes": 4332}]})
        runs.append({"runId": "2026-11-09_16-10-40_SYSTEM_START", "recipe": "SYSTEM_START",
                     "startedAt": "2026-11-09T16:10:40", "endedAt": "2026-11-09T16:12:38", "outcome": "complete",
                     "message": "(01m 58s)", "artifacts": [{"kind": "log", "fileName": "SYSTEM_START.log", "bytes": 2210}]})
        return {"runs": runs[: int(count)]}

    def _fetch_run_files(self, runId="latest", kinds="tsv,log,recipe,snapshot"):
        return error("not_available", "downloadDir is not configured in this session")

    def _tail_log(self, count=50):
        lines = self.log_lines
        if lines is None:
            lines = [
                {"ts": "09:00:01.120", "source": "Agent", "message": "Agent server listening on 127.0.0.1:8026", "level": "Detail"},
                {"ts": "09:00:01.842", "source": "Core", "message": f"Hardware configuration loaded for {self.m['profile']}",
                 "level": "Detail"},
            ]
            if self.run is not None:
                lines.append({"ts": "14:02:11.004", "source": "Execute", "message": f"Recipe started: {self.run['name']}",
                              "level": "Info"})
                if self._run_ended():
                    lines.append({"ts": "14:44:23.310", "source": "Execute", "message": f"Recipe complete: {self.run['name']}",
                                  "level": "Info"})
        return lines[-int(count):]
