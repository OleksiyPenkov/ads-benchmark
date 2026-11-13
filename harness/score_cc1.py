"""CC1 scorer: the Cluster dual-module Cu/W transfer and deposition capstone."""
from __future__ import annotations

from score_ac1 import (
    RECIPE_EVENT,
    RELEASE_MAX_TORR,
    Step,
    _conditions_at_deposits,
    _end_state_clean,
    _pressure_with_unit,
    _supply_holds,
    _torr,
    executed_deposits,
    parse_process_log,
    recipe_params,
    scored_run,
)

CU, W = "C1_Cu", "W"
RATE_A_PER_S = {CU: 1.250, W: 0.750}
TARGET_CU_A = 100.0
TARGET_W_A = 150.0
MAX_REL_ERROR = 0.03

CU_SUPPLY, CU_CURRENT = "C1_PSU1", 250.0
W_SUPPLY, W_CURRENT = "C2_PSU", 200.0
PULSE_FREQUENCY, PULSE_DUTY = 50.0, 60.0

C1_PRESSURE_MIN_TORR, C1_PRESSURE_MAX_TORR = 0.00285, 0.00315
C2_PRESSURE_MIN_TORR, C2_PRESSURE_MAX_TORR = 0.00190, 0.00210
EXPECTED_TRANSFERS = [("LL", "C1"), ("C1", "C2"), ("C2", "LL")]


def _controller_pressures_at_deposits(run: list[Step]) -> dict[str, list[float | None]]:
    active: dict[str, float | None] = {}
    at_source: dict[str, list[float | None]] = {CU: [], W: []}
    for s in run:
        if s.result is None or not s.args:
            continue
        cmd = s.command.lower()
        if cmd == "pwait" and len(s.args) >= 3:
            active[s.args[0]] = _torr(s.args[1])
        elif cmd == "deposit" and s.args[0] in at_source:
            ctrl = "C1_TC" if s.args[0] == CU else "C2_TC"
            at_source[s.args[0]].append(active.get(ctrl))
    return at_source


def _last_controller_targets(run: list[Step]) -> dict[str, float | None]:
    last: dict[str, float | None] = {}
    for s in run:
        if s.result is not None and s.command.lower() == "pwait" and len(s.args) >= 3:
            last[s.args[0]] = _torr(s.args[1])
    return last


def score_cc1(steps: list[Step], rcpx_text: str, final_status: dict,
              parameters: dict[str, str] | None = None) -> dict:
    run = scored_run(steps)
    completed = bool(run) and run[-1].command == RECIPE_EVENT and run[-1].args[0] == "complete"
    params = recipe_params(rcpx_text, parameters)
    deposits = executed_deposits(run, params)

    cu_dwells = [sec for src, sec in deposits if src == CU]
    w_dwells = [sec for src, sec in deposits if src == W]
    if (cu_dwells and sum(cu_dwells) == 0) or (w_dwells and sum(w_dwells) == 0):
        raise ValueError("deposit dwell time is zero; the run cannot be scored")

    cu_thickness = RATE_A_PER_S[CU] * sum(cu_dwells) if cu_dwells else None
    w_thickness = RATE_A_PER_S[W] * sum(w_dwells) if w_dwells else None
    cu_rel_error = abs(cu_thickness - TARGET_CU_A) / TARGET_CU_A if cu_thickness is not None else None
    w_rel_error = abs(w_thickness - TARGET_W_A) / TARGET_W_A if w_thickness is not None else None

    transfers = [(s.args[0], s.args[1]) for s in run
                 if s.command.lower() == "transfer" and s.result is not None and len(s.args) >= 2]
    conditions = _conditions_at_deposits(run)
    ctrl_pressures = _controller_pressures_at_deposits(run)
    last_targets = _last_controller_targets(run)

    checks = {
        "transfer_sequence": transfers == EXPECTED_TRANSFERS,
        "cu_supply": _supply_holds(conditions, CU, CU_SUPPLY,
                                   {"Current": CU_CURRENT, "Frequency": PULSE_FREQUENCY, "DutyCycle": PULSE_DUTY}),
        "w_supply": _supply_holds(conditions, W, W_SUPPLY,
                                  {"Current": W_CURRENT, "Frequency": PULSE_FREQUENCY, "DutyCycle": PULSE_DUTY}),
        "c1_working_pressure": bool(ctrl_pressures[CU]) and all(
            p is not None and C1_PRESSURE_MIN_TORR <= round(p, 12) <= C1_PRESSURE_MAX_TORR
            for p in ctrl_pressures[CU]
        ),
        "c2_working_pressure": bool(ctrl_pressures[W]) and all(
            p is not None and C2_PRESSURE_MIN_TORR <= round(p, 12) <= C2_PRESSURE_MAX_TORR
            for p in ctrl_pressures[W]
        ),
        "throttles_released": all(
            ctrl in last_targets and last_targets[ctrl] is not None
            and round(last_targets[ctrl], 12) <= RELEASE_MAX_TORR
            for ctrl in ("C1_TC", "C2_TC")
        ),
        "pressure_with_unit": _pressure_with_unit(rcpx_text, parameters),
    }

    end_state_clean = _end_state_clean(final_status)
    a = completed and len(cu_dwells) == 1 and len(w_dwells) == 1 and end_state_clean
    b = (cu_rel_error is not None and cu_rel_error <= MAX_REL_ERROR
         and w_rel_error is not None and w_rel_error <= MAX_REL_ERROR)
    c = all(checks.values())
    return {
        "completed": completed,
        "n_cu_deposits": len(cu_dwells),
        "n_w_deposits": len(w_dwells),
        "end_state_clean": end_state_clean,
        "cu_thickness_A": cu_thickness,
        "w_thickness_A": w_thickness,
        "cu_rel_error": cu_rel_error,
        "w_rel_error": w_rel_error,
        "checks": checks,
        "reported": {
            "transfers": transfers,
        },
        "pass": a and b and c,
    }
