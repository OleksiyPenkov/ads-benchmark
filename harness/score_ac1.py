"""AC1 scorer: the Ru/C capstone, scored from the dwell times Control actually executed.

The inputs are the Control process log of the trial, the ``.rcpx`` the agent ran, and the
``system_status`` reply the harness reads after the trial. Nothing is measured: the emulator
has no metrology, so thicknesses are executed dwell times multiplied by the brief rates (D5).

Process-log format (taken from ``fixtures\\test\\emulator-05-process-log.txt``)::

    19:59:13.059  [Execute]    exec: deposit B4C 100.26 1
    19:59:13.061  [Execute]    target=Drive, argTypes=[DeviceType,String,Single,String]
    20:01:08.220  [Execute]    result: 0
    19:57:34.776  [Recipe]  Recipe started: ARM-A-W-B4C-ML-d55.7-G0.29-N30
    20:40:29.612  [Recipe]  Recipe complete: ARM-A-W-B4C-ML-d55.7-G0.29-N30 (42m 54s)

The ``[Recipe]`` lines follow the pattern Control itself parses in ``RunArtifacts.cs``
(``started|complete|failed|aborted|INCOMPLETE``). Control writes the ``exec:`` line after it has
resolved parameters and converted a ``pwait`` target to Torr, so ``pwait TC1 0.0025 300`` in the
log came from ``pwait $P 300`` with ``P = 2.50mT``. The unit an agent wrote is therefore visible
only in the ``.rcpx`` script, and ``swait`` writes no ``exec:`` line at all; the pressure-unit check
reads the script for that reason.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

RU, C = "Ru", "C"
RATE_A_PER_S = {RU: 0.4711, C: 0.4030}
TARGET_PERIOD_A = 68.5
MAX_REL_ERROR = 0.03
N_PERIODS = 5

# Dead-time model for Ru, h = 0.4990 * (t - 2.02 s). Reported only, never scored.
DEADTIME_RATE_RU = 0.4990
DEADTIME_S = 2.02

RU_SUPPLY, RU_CURRENT = "PSU_C", 50.0
C_SUPPLY, C_CURRENT = "PSU_B", 300.0
PULSE_FREQUENCY, PULSE_DUTY = 50.0, 60.0

# 2 mTorr +-5 %, both ends inclusive. Targets are rounded to 1e-12 Torr before the comparison so that
# a unit conversion (2.1mT -> 0.0021000000000000003) cannot push an exact endpoint outside.
WORKING_PRESSURE_MIN_TORR, WORKING_PRESSURE_MAX_TORR = 0.0019, 0.0021

# Throttle release (emulator semantics, describe_dsl of Control 1.6.3.1477 and the AC1 brief): target 0 opens the
# throttle at once, and a target at or below the gauge floor (0.01 mTorr = 1e-5 Torr) opens it too. Rounded to
# 1e-12 Torr before the comparison, as the working pressure.
RELEASE_MAX_TORR = 1e-5

RECIPE_EVENT = "@recipe"

_TIME = r"(\d{2}:\d{2}:\d{2}\.\d{3})"
_EXEC = re.compile(_TIME + r"  \[Execute\]    exec: (.*)$")
_RESULT = re.compile(_TIME + r"  \[Execute\]    result: (.*)$")
_RECIPE = re.compile(_TIME + r"  \[Recipe\]  Recipe (started|complete|failed|aborted|INCOMPLETE): (.+)$")
_LEADING_NUMBER = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_PRESSURE_WITH_UNIT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*(?:mT|uT|mTorr|Torr)$",
                                 re.IGNORECASE)
_COMMENT = re.compile(r"(?:^|\s)#.*$")
_PRESSURE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$")
_TORR_PER_UNIT = {"": 1.0, "torr": 1.0, "mt": 1e-3, "mtorr": 1e-3, "ut": 1e-6}


@dataclass
class Step:
    """One executed command: ``exec:`` text split into a command and its args, plus its ``result:``.

    ``result`` is None when no ``result:`` line followed (the command did not finish). A
    ``[Recipe]`` lifecycle line is also kept as a Step, with command ``RECIPE_EVENT``,
    ``args = [phase, rest of line]`` and no result, so that a run can be cut out of the log.
    """

    time: str
    command: str
    args: list[str]
    result: str | None


def parse_process_log(text: str) -> list[Step]:
    """Pairs each ``[Execute] exec:`` line with the ``result:`` line that follows it."""
    steps: list[Step] = []
    pending: Step | None = None
    for line in text.splitlines():
        if m := _EXEC.match(line):
            tokens = m.group(2).split()
            pending = Step(m.group(1), tokens[0] if tokens else "", tokens[1:], None)
            steps.append(pending)
        elif m := _RESULT.match(line):
            if pending is not None:
                pending.result = m.group(2)
                pending = None
        elif m := _RECIPE.match(line):
            steps.append(Step(m.group(1), RECIPE_EVENT, [m.group(2), m.group(3)], None))
            pending = None
    return steps


def _rcpx_root(rcpx_text: str) -> ET.Element:
    return ET.fromstring(rcpx_text.encode("utf-8"))


def _raw_params(rcpx_text: str, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """``<Param>`` texts, with start_recipe's ``parametersJson`` overrides applied on top."""
    raw = {p.get("name"): (p.text or "").strip() for p in _rcpx_root(rcpx_text).iter("Param") if p.get("name")}
    raw.update({name: str(value).strip() for name, value in (overrides or {}).items()})
    return raw


def recipe_params(rcpx_text: str, overrides: dict[str, str] | None = None) -> dict[str, float]:
    """The ``<Param>`` values of a recipe (after ``overrides``) that are plain numbers.

    Values with a unit suffix (``2.50mT``) are left out: a float cannot carry the unit.
    """
    params: dict[str, float] = {}
    for name, raw in _raw_params(rcpx_text, overrides).items():
        try:
            params[name] = float(raw)
        except ValueError:
            pass
    return params


def _seconds(token: str, params: dict[str, float]) -> float:
    try:
        return float(token)
    except ValueError:
        name = token[1:] if token.startswith("$") else token
        if name in params:
            return params[name]
        raise ValueError(f"deposit time {token!r} is neither a number nor a recipe parameter") from None


def executed_deposits(steps: list[Step], params: dict[str, float]) -> list[tuple[str, float]]:
    """``(source, seconds)`` for each ``deposit`` that has a ``result:`` line, in log order.

    A dwell written as ``$name`` or ``name`` is resolved from ``params`` (``recipe_params``).
    """
    return [(s.args[0], _seconds(s.args[1], params))
            for s in steps if s.command.lower() == "deposit" and s.result is not None and len(s.args) >= 2]


def scored_run(steps: list[Step]) -> list[Step]:
    """The steps of the last recipe run that executed a deposit.

    A run starts at a ``Recipe started`` event and ends at the next lifecycle event (included).
    One process log covers a whole Control launch, so earlier smoke or test runs are excluded.
    """
    runs: list[list[Step]] = []
    open_run = False
    for step in steps:
        if step.command == RECIPE_EVENT and step.args[0] == "started":
            runs.append([step])
            open_run = True
        elif open_run:
            runs[-1].append(step)
            open_run = step.command != RECIPE_EVENT
    with_deposit = [r for r in runs if any(s.command.lower() == "deposit" for s in r)]
    return with_deposit[-1] if with_deposit else []


def _number(token: str) -> float | None:
    m = _LEADING_NUMBER.match(token)
    return float(m.group(1)) if m else None


def _torr(token: str) -> float | None:
    """A pwait target in Torr. Control logs it already converted to bare Torr; a suffix is also accepted."""
    m = _PRESSURE.match(token)
    if not m or m.group(2).lower() not in _TORR_PER_UNIT:
        return None
    return float(m.group(1)) * _TORR_PER_UNIT[m.group(2).lower()]


def _pwait_target(step: Step) -> str:
    """``pwait [controller] <target> <timeout>``: the target is the second-last argument."""
    return step.args[-2] if len(step.args) >= 2 else step.args[0]


@dataclass
class _DepositConditions:
    source: str
    supplies: dict[str, dict[str, float | None]]
    pressure_torr: float | None


def _conditions_at_deposits(run: list[Step]) -> list[_DepositConditions]:
    """The supply setpoints and pwait target in force at each completed deposit of one run.

    Only executed steps (with a ``result:`` line) of ``run`` count, in log order, so a value set
    in an earlier recipe run is not inherited and a later set overrides an earlier one.
    """
    supplies: dict[str, dict[str, float | None]] = {}
    pressure: float | None = None
    found: list[_DepositConditions] = []
    for s in run:
        if s.result is None:
            continue
        command = s.command.lower()
        if command == "set" and len(s.args) >= 3:
            supplies.setdefault(s.args[0], {})[s.args[1].lower()] = _number(s.args[2])
        elif command == "pwait" and s.args:
            pressure = _torr(_pwait_target(s))
        elif command == "deposit" and s.args:
            found.append(_DepositConditions(s.args[0], {k: dict(v) for k, v in supplies.items()}, pressure))
    return found


def _supply_holds(deposits: list[_DepositConditions], source: str, supply: str,
                  settings: dict[str, float]) -> bool:
    """At least one ``source`` deposit, and at each one ``supply`` holds every value in ``settings``."""
    mine = [d for d in deposits if d.source == source]
    return bool(mine) and all(d.supplies.get(supply, {}).get(q.lower()) == v
                              for d in mine for q, v in settings.items())


def _pressure_with_unit(rcpx_text: str, overrides: dict[str, str] | None = None) -> bool:
    """True if the script has at least one ``pwait``/``swait`` and every pressure carries a unit.

    ``pwait [sensor] <p> <timeout>`` and ``swait <sensor> <p> <timeout>``: the pressure is the
    second-last token. ``$name`` and bare parameter names resolve to the ``<Param>`` text, or to the
    start_recipe override of that parameter when one was given.
    Accepted suffixes: mT, uT, mTorr, Torr (case-insensitive).
    """
    raw = _raw_params(rcpx_text, overrides)
    script_el = _rcpx_root(rcpx_text).find("Script")
    script = "".join(script_el.itertext()) if script_el is not None else ""
    seen = False
    for line in script.splitlines():
        for statement in _COMMENT.sub("", line).split(";"):
            tokens = statement.split()
            if not tokens or tokens[0].lower() not in ("pwait", "swait"):
                continue
            seen = True
            if len(tokens) < 3:
                return False
            token = tokens[-2]
            name = token[1:] if token.startswith("$") else token
            val = raw.get(name, token).strip()
            if tokens[0].lower() == "pwait" and _number(val) == 0.0:
                continue
            if not _PRESSURE_WITH_UNIT.match(val):
                return False
    return seen


def _end_state_clean(final_status: dict) -> bool:
    """Reads the ``system_status`` reply (``status.system``; ``CoreAgentSystemAccess.SystemStatus``).

    Keys used: ``chambers`` (list of ``{"id", "state"}``; the ``Main`` chamber must be ``"Idle"``),
    ``activeAlarms`` (must be an empty list) and ``isRecipeRunning`` (must be false). A missing key
    counts as not clean.
    """
    main = [c for c in final_status.get("chambers", []) if c.get("id") == "Main"]
    return (len(main) == 1 and main[0].get("state") == "Idle"
            and final_status.get("activeAlarms") == []
            and final_status.get("isRecipeRunning") is False)


def _layers(deposits: list[tuple[str, float]]) -> tuple[list[float], list[tuple[float, float]], float | None, int]:
    """Split deposits into adhesion Ru dwells, (t_C, t_Ru) periods, a trailing C cap, and a count of the rest.

    Supports both Ru-adhesion + [C, Ru] * N (with optional trailing C cap) and Ru-adhesion + [Ru, C] * N.
    """
    if (len(deposits) >= 3 and deposits[0][0] == RU
            and len(deposits) % 2 == 1
            and all(deposits[i][0] == RU and deposits[i + 1][0] == C for i in range(1, len(deposits), 2))):
        adhesion = [deposits[0][1]]
        periods = [(deposits[i + 1][1], deposits[i][1]) for i in range(1, len(deposits), 2)]
        return adhesion, periods, None, 0
    first_c = next((i for i, (source, _) in enumerate(deposits) if source == C), len(deposits))
    adhesion = [t for source, t in deposits[:first_c] if source == RU]
    unpaired = first_c - len(adhesion)
    periods: list[tuple[float, float]] = []
    cap = None
    i = first_c
    while i < len(deposits):
        source, t = deposits[i]
        if source == C and i + 1 < len(deposits) and deposits[i + 1][0] == RU:
            periods.append((t, deposits[i + 1][1]))
            i += 2
            continue
        if source == C and i == len(deposits) - 1:
            cap = t
        else:
            unpaired += 1
        i += 1
    return adhesion, periods, cap, unpaired


def score_ac1(steps: list[Step], rcpx_text: str, final_status: dict,
              parameters: dict[str, str] | None = None) -> dict:
    """Scores AC1. Pass = (a) and (b) and (c).

    (a) ``completed`` (the scored run ends with ``Recipe complete``), ``n_periods == 5`` and
        ``end_state_clean``.
    (b) ``period_mean_A`` = mean over periods of 0.4030 * t_C + 0.4711 * t_Ru, and
        ``rel_error`` = |period_mean_A - 68.5| / 68.5 <= 0.03.
    (c) every entry of ``checks`` is true. Except ``no_rf`` and ``pressure_with_unit``, the checks
        read only executed steps (with a ``result:`` line) of ``scored_run(steps)``, in log order:

        - ``ru_supply_current`` / ``c_supply_current``: at every Ru (C) deposit the setpoint in force
          is Current 50 on PSU_C (Current 300 on PSU_B), and at least one such deposit exists.
          Setting the current to 0 after the last deposit (shutdown) does not matter.
        - ``pulse``: at every Ru deposit PSU_C, and at every C deposit PSU_B, holds Frequency 50 and
          DutyCycle 60.
        - ``working_pressure``: at every deposit of the run, the last executed ``pwait`` target is
          2 mTorr within +-5 % (0.0019-0.0021 Torr inclusive; the log carries bare Torr).
        - ``throttle_released``: the last executed ``pwait`` of the run has target 0 or a target at or below
          the gauge floor, 0.01 mTorr (any unit).
        - ``pressure_with_unit``: reads the ``.rcpx`` script (see ``_pressure_with_unit``).
        - ``no_rf``: no ``set`` on a supply whose id starts with ``RF`` anywhere in the log,
          executed or not.

    Settings made in an earlier recipe run are not inherited by the scored run.

    ``rcpx_text`` is the recipe that ran and ``parameters`` the start_recipe ``parametersJson`` overrides
    it ran with; both feed dwell-time resolution and ``pressure_with_unit``. A run whose mean period is
    zero (zero dwell times) cannot be scored and raises ValueError.

    ``reported`` is not scored: ``gamma`` (mean Ru thickness / ``period_mean_A``), ``adhesion_A``,
    ``cap_A`` (trailing C, or None), ``unpaired_deposits`` and ``deadtime_period_mean_A``. The last
    uses the dead-time model h = 0.4990 * (t - 2.02 s) for Ru only; no dead-time model exists for C,
    so C enters that figure at its simple rate 0.4030 * t_C.
    """
    run = scored_run(steps)
    completed = bool(run) and run[-1].command == RECIPE_EVENT and run[-1].args[0] == "complete"
    params = recipe_params(rcpx_text, parameters)
    adhesion, periods, cap, unpaired = _layers(executed_deposits(run, params))

    period_mean = rel_error = gamma = deadtime_mean = None
    if periods:
        period_mean = sum(RATE_A_PER_S[C] * tc + RATE_A_PER_S[RU] * tru for tc, tru in periods) / len(periods)
        if period_mean == 0:
            raise ValueError("the mean period is zero (zero dwell times); the run cannot be scored")
        rel_error = abs(period_mean - TARGET_PERIOD_A) / TARGET_PERIOD_A
        gamma = sum(RATE_A_PER_S[RU] * tru for _, tru in periods) / len(periods) / period_mean
        deadtime_mean = sum(RATE_A_PER_S[C] * tc + DEADTIME_RATE_RU * (tru - DEADTIME_S)
                            for tc, tru in periods) / len(periods)

    conditions = _conditions_at_deposits(run)
    pwaits = [s for s in run if s.command.lower() == "pwait" and s.result is not None and s.args]
    last_target = _torr(_pwait_target(pwaits[-1])) if pwaits else None
    checks = {
        "ru_supply_current": _supply_holds(conditions, RU, RU_SUPPLY, {"Current": RU_CURRENT}),
        "c_supply_current": _supply_holds(conditions, C, C_SUPPLY, {"Current": C_CURRENT}),
        "pulse": all(_supply_holds(conditions, source, supply,
                                   {"Frequency": PULSE_FREQUENCY, "DutyCycle": PULSE_DUTY})
                     for source, supply in ((RU, RU_SUPPLY), (C, C_SUPPLY))),
        "working_pressure": bool(conditions) and all(
            d.pressure_torr is not None
            and WORKING_PRESSURE_MIN_TORR <= round(d.pressure_torr, 12) <= WORKING_PRESSURE_MAX_TORR
            for d in conditions),
        "pressure_with_unit": _pressure_with_unit(rcpx_text, parameters),
        "no_rf": not any(s.command.lower() == "set" and s.args and s.args[0].upper().startswith(("RF", "PSU_RF"))
                         for s in steps),
        "throttle_released": last_target is not None and round(last_target, 12) <= RELEASE_MAX_TORR,
    }

    end_state_clean = _end_state_clean(final_status)
    a = completed and len(periods) == N_PERIODS and end_state_clean
    b = rel_error is not None and rel_error <= MAX_REL_ERROR
    c = all(checks.values())
    return {
        "completed": completed,
        "n_periods": len(periods),
        "end_state_clean": end_state_clean,
        "period_mean_A": period_mean,
        "rel_error": rel_error,
        "checks": checks,
        "reported": {
            "gamma": gamma,
            "adhesion_A": RATE_A_PER_S[RU] * sum(adhesion),
            "cap_A": RATE_A_PER_S[C] * cap if cap is not None else None,
            "unpaired_deposits": unpaired,
            "deadtime_period_mean_A": deadtime_mean,
        },
        "pass": a and b and c,
    }
