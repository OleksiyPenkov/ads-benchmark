"""SC1 scorer: the Sputter Mo/SiO2 5-bilayer capstone, scored from executed dwell times."""
from __future__ import annotations

from score_ac1 import (
    RECIPE_EVENT,
    Step,
    _COMMENT,
    _conditions_at_deposits,
    _end_state_clean,
    _rcpx_root,
    _supply_holds,
    executed_deposits,
    parse_process_log,
    recipe_params,
    scored_run,
)

MO, SIO2 = "Mo", "SiO2"
RATE_A_PER_S = {MO: 0.800, SIO2: 0.400}
TARGET_PERIOD_A = 60.0
MAX_REL_ERROR = 0.03
N_PERIODS = 5

MO_SUPPLY, MO_CURRENT = "PSU1", 200.0
PULSE_FREQUENCY, PULSE_DUTY = 50.0, 60.0
SIO2_SUPPLY, SIO2_POWER = "PSU2", 150.0


def _sputter_2arg_deposit(rcpx_text: str) -> bool:
    script = _rcpx_root(rcpx_text).findtext("Script") or ""
    seen = False
    for raw_line in script.splitlines():
        line = _COMMENT.sub("", raw_line).strip()
        if not line or line.startswith(";"):
            continue
        tokens = line.split()
        if tokens[0].lower() == "deposit":
            seen = True
            if len(tokens) != 3:
                return False
    return seen


def _layers(deposits: list[tuple[str, float]]) -> tuple[list[tuple[float, float]], int]:
    periods: list[tuple[float, float]] = []
    unpaired = 0
    i = 0
    while i < len(deposits):
        if i + 1 < len(deposits):
            s1, t1 = deposits[i]
            s2, t2 = deposits[i + 1]
            if (s1, s2) == (MO, SIO2):
                periods.append((t1, t2))
                i += 2
                continue
            if (s1, s2) == (SIO2, MO):
                periods.append((t2, t1))
                i += 2
                continue
        unpaired += 1
        i += 1
    return periods, unpaired


def score_sc1(steps: list[Step], rcpx_text: str, final_status: dict,
              parameters: dict[str, str] | None = None) -> dict:
    run = scored_run(steps)
    completed = bool(run) and run[-1].command == RECIPE_EVENT and run[-1].args[0] == "complete"
    params = recipe_params(rcpx_text, parameters)
    periods, unpaired = _layers(executed_deposits(run, params))

    period_mean = rel_error = gamma = None
    if periods:
        period_mean = sum(RATE_A_PER_S[MO] * t_mo + RATE_A_PER_S[SIO2] * t_sio2
                          for t_mo, t_sio2 in periods) / len(periods)
        if period_mean == 0:
            raise ValueError("the mean period is zero (zero dwell times); the run cannot be scored")
        rel_error = abs(period_mean - TARGET_PERIOD_A) / TARGET_PERIOD_A
        gamma = sum(RATE_A_PER_S[MO] * t_mo for t_mo, _ in periods) / len(periods) / period_mean

    conditions = _conditions_at_deposits(run)
    checks = {
        "mo_supply_current": _supply_holds(conditions, MO, MO_SUPPLY, {"Current": MO_CURRENT}),
        "mo_pulse": _supply_holds(conditions, MO, MO_SUPPLY,
                                  {"Frequency": PULSE_FREQUENCY, "DutyCycle": PULSE_DUTY}),
        "sio2_supply_power": _supply_holds(conditions, SIO2, SIO2_SUPPLY, {"Power": SIO2_POWER}),
        "sputter_2arg_deposit": _sputter_2arg_deposit(rcpx_text),
        "no_pwait": not any(s.command.lower() == "pwait" for s in run),
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
            "unpaired_deposits": unpaired,
        },
        "pass": a and b and c,
    }
