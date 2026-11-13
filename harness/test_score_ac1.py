from pathlib import Path

import pytest

import score_ac1
from score_ac1 import Step, executed_deposits, parse_process_log, recipe_params, score_ac1 as score

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "test"
EMULATOR05_LOG = FIXTURES / "emulator-05-process-log.txt"
EMULATOR05_RCPX = FIXTURES / "emulator-05.rcpx"

CLEAN_STATUS = {
    "chambers": [{"id": "LL", "state": "Idle"}, {"id": "Main", "state": "Idle"}],
    "activeAlarms": [],
    "isRecipeRunning": False,
}


def _rcpx(pressure="2mT", release="0mT"):
    """A minimal Ru/C recipe: the scorer reads its <Param> values and the pwait/swait lines of its <Script>."""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Recipe>
  <Parameters>
    <Group name="Deposition Conditions">
      <Param name="P" caption="Pressure" unit="mTorr" paramType="0">{pressure}</Param>
      <Param name="PBase" caption="Release pressure" unit="mTorr" paramType="0">{release}</Param>
    </Group>
    <Group name="Layers">
      <Param name="timeRu" caption="Time" unit="sec" paramType="0">31.26</Param>
      <Param name="timeC" caption="Time" unit="sec" paramType="0">133.44</Param>
    </Group>
  </Parameters>
  <Script>
pwait $P 300
deposit Ru $timeRu 1
pwait $PBase 300
  </Script>
</Recipe>
"""


class _Log:
    """Writes a process log in the emulator-05 line format (two spaces after the time, four before exec:)."""

    def __init__(self):
        self.lines = ["--- Process log started 2026-10-09 10:00:00 ---", "------ 2026-10-09  Friday ------"]
        self.t = 36000.0

    def _stamp(self, dt=0.002):
        self.t += dt
        h, rem = divmod(self.t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    def recipe(self, phase, text):
        self.lines.append(f"{self._stamp()}  [Recipe]  Recipe {phase}: {text}")

    def step(self, command, result, *, target="Drive, argTypes=[DeviceType,String]", dt=1.0):
        self.lines.append(f"{self._stamp()}  [Execute]    exec: {command}")
        self.lines.append(f"{self._stamp()}  [Execute]    target={target}")
        if result is not None:
            self.lines.append(f"{self._stamp(dt)}  [Execute]    result: {result}")

    def text(self):
        return "\r\n".join(self.lines) + "\r\n"


def _ruc_log(*, n=5, t_c=133.44, t_ru=31.26, adhesion_ru=40.0, cap_c=50.0,
             pressure="0.002", release="0", extra=(), mid_run=(), setup=True, name="AC1-RuC"):
    log = _Log()
    log.recipe("started", name)
    log.step(f"pwait TC1 {pressure} 300", pressure)
    for supply, current in (("PSU_C", 50), ("PSU_B", 300)) if setup else ():
        log.step(f"set {supply} Frequency 50", 50)
        log.step(f"set {supply} DutyCycle 60", 60)
        log.step(f"set {supply} Current {current}", current)
        log.step(f"power {supply} true", "True")
    for command in extra:
        log.step(command, 0)
    log.step("holder Ru", 16096)
    log.step(f"deposit Ru {adhesion_ru} 1", 0, dt=adhesion_ru)
    for i in range(n):
        if i == n // 2:
            for command in mid_run:
                log.step(command, 0)
        log.step("holder C", 8048)
        log.step(f"deposit C {t_c} 1", 0, dt=t_c)
        log.step("holder Ru", 16096)
        log.step(f"deposit Ru {t_ru} 1", 0, dt=t_ru)
    if cap_c:
        log.step("holder C", 8048)
        log.step(f"deposit C {cap_c} 1", 0, dt=cap_c)
    log.step("set PSU_C Current 0", 0)
    log.step("set PSU_B Current 0", 0)
    log.step(f"pwait TC1 {release} 300", release)
    log.recipe("complete", f"{name} (01h 30m 00s)")
    return log.text()


def _score(text, rcpx=None, status=None):
    return score(parse_process_log(text), rcpx if rcpx is not None else _rcpx(),
                 status if status is not None else CLEAN_STATUS)


def test_emulator05_deposit_count():
    text = EMULATOR05_LOG.read_text(encoding="utf-8")
    steps = parse_process_log(text)
    params = recipe_params(EMULATOR05_RCPX.read_text(encoding="utf-8"))

    assert params["timeB4Cadh"] == pytest.approx(100.26)
    assert all(isinstance(step, Step) for step in steps)
    assert text.splitlines()[895].endswith(
        "[Recipe]  Recipe complete: ARM-A-W-B4C-ML-d55.7-G0.29-N30 (42m 54s)")

    run = score_ac1.scored_run(steps)
    assert run[0].command == score_ac1.RECIPE_EVENT
    assert run[0].args == ["started", "ARM-A-W-B4C-ML-d55.7-G0.29-N30"]
    assert run[-1].args == ["complete", "ARM-A-W-B4C-ML-d55.7-G0.29-N30 (42m 54s)"]

    deposits = executed_deposits(run, params)
    assert len(deposits) == 61
    assert deposits[0] == ("B4C", pytest.approx(100.26))
    assert deposits[1:3] == [("W", pytest.approx(11.95)), ("B4C", pytest.approx(33.89))]
    assert [source for source, _ in deposits[1::2]] == ["W"] * 30

    # The SMOKE run earlier in the same log is a separate recipe run.
    assert len(executed_deposits(steps, params)) == 62


def test_parameter_names_resolve_from_rcpx():
    log = _Log()
    log.step("deposit Ru $timeRu 1", 0)
    log.step("deposit C timeC 1", 0)
    log.step("deposit C 5 1", None)  # no result line: not completed
    deposits = executed_deposits(parse_process_log(log.text()), recipe_params(_rcpx()))
    assert deposits == [("Ru", pytest.approx(31.26)), ("C", pytest.approx(133.44))]


def test_period_from_dwell():
    result = _score(_ruc_log())
    assert result["completed"] is True
    assert result["n_periods"] == 5
    assert result["end_state_clean"] is True
    assert result["period_mean_A"] == pytest.approx(68.50, abs=0.01)
    assert result["rel_error"] <= 0.0002
    assert all(result["checks"].values()), result["checks"]
    assert result["reported"]["adhesion_A"] == pytest.approx(0.4711 * 40.0)
    assert result["reported"]["cap_A"] == pytest.approx(0.4030 * 50.0)
    assert result["reported"]["gamma"] == pytest.approx(0.4711 * 31.26 / result["period_mean_A"])
    assert result["pass"] is True


def test_wrong_n_fails():
    result = _score(_ruc_log(n=29))
    assert result["n_periods"] == 29
    assert result["rel_error"] <= 0.0002
    assert result["pass"] is False


def test_rf_set_fails():
    result = _score(_ruc_log(extra=["set RF601 Power 100"]))
    assert result["checks"]["no_rf"] is False
    assert result["pass"] is False


def test_bare_pressure_fails():
    rcpx = _rcpx().replace("pwait $P 300", "pwait 2 120")
    result = _score(_ruc_log(), rcpx=rcpx)
    assert result["checks"]["pressure_with_unit"] is False
    assert result["pass"] is False

    swait_bare = _rcpx().replace("pwait $PBase 300", "swait pm 1 300")
    assert _score(_ruc_log(), rcpx=swait_bare)["checks"]["pressure_with_unit"] is False
    param_bare = _rcpx(pressure="0.002")
    assert _score(_ruc_log(), rcpx=param_bare)["checks"]["pressure_with_unit"] is False

    pwait_zero = _rcpx().replace("pwait $PBase 300", "pwait 0 1")
    assert _score(_ruc_log(), rcpx=pwait_zero)["checks"]["pressure_with_unit"] is True

def test_deadtime_reported_not_scored(monkeypatch):
    baseline = _score(_ruc_log())
    monkeypatch.setattr(score_ac1, "DEADTIME_S", 10.0)
    shifted = _score(_ruc_log())

    assert baseline["reported"]["deadtime_period_mean_A"] == pytest.approx(0.4030 * 133.44 + 0.4990 * (31.26 - 2.02))
    assert shifted["reported"]["deadtime_period_mean_A"] == pytest.approx(0.4030 * 133.44 + 0.4990 * (31.26 - 10.0))
    assert shifted["period_mean_A"] == baseline["period_mean_A"]
    assert shifted["pass"] is baseline["pass"] is True


def test_process_checks_fail_independently():
    assert _score(_ruc_log(release="2E-05"))["checks"]["throttle_released"] is False
    wrong_current = _ruc_log().replace("exec: set PSU_C Current 50", "exec: set PSU_C Current 120")
    assert _score(wrong_current)["checks"]["ru_supply_current"] is False
    wrong_duty = _ruc_log().replace("exec: set PSU_B DutyCycle 60", "exec: set PSU_B DutyCycle 70")
    assert _score(wrong_duty)["checks"]["pulse"] is False


def test_throttle_release_at_zero_or_gauge_floor():
    # Emulator semantics (describe_dsl): target 0 opens at once; a target at/below 0.01 mTorr opens it too.
    assert _score(_ruc_log(release="0"))["checks"]["throttle_released"] is True
    assert _score(_ruc_log(release="1E-05"))["checks"]["throttle_released"] is True   # 0.01mT as logged
    assert _score(_ruc_log(release="0.01mT"))["checks"]["throttle_released"] is True
    assert _score(_ruc_log(release="0.02mT"))["checks"]["throttle_released"] is False
    assert _score(_ruc_log(release="2E-05"))["checks"]["throttle_released"] is False  # 0.02mT as logged


def test_incomplete_or_dirty_end_fails():
    aborted = _ruc_log().replace("Recipe complete: AC1-RuC", "Recipe aborted: AC1-RuC")
    assert _score(aborted)["completed"] is False
    assert _score(aborted)["pass"] is False

    alarm = dict(CLEAN_STATUS, activeAlarms=["Main: pressure high"])
    assert _score(_ruc_log(), status=alarm)["end_state_clean"] is False
    running = dict(CLEAN_STATUS, isRecipeRunning=True)
    assert _score(_ruc_log(), status=running)["end_state_clean"] is False
    processing = dict(CLEAN_STATUS, chambers=[{"id": "Main", "state": "Processing"}])
    assert _score(_ruc_log(), status=processing)["pass"] is False
    assert _score(_ruc_log(), status={})["end_state_clean"] is False


def test_wrong_working_pressure_fails():
    # Recipe and executed log agree on 3 mTorr: the unit is present, the value is wrong.
    result = _score(_ruc_log(pressure="0.003"), rcpx=_rcpx(pressure="3mT"))
    assert result["checks"]["pressure_with_unit"] is True
    assert result["checks"]["working_pressure"] is False
    assert result["pass"] is False

    # Within +-5 % of 2 mTorr passes; the final release to 0 does not count as the working pressure.
    assert _score(_ruc_log(pressure="0.00205"))["checks"]["working_pressure"] is True
    assert _score(_ruc_log(pressure="0.0022"))["checks"]["working_pressure"] is False
    # Both exact endpoints of +-5 % are inclusive; just outside them fails.
    for inside in ("0.0019", "0.0021", "1.9mT", "2.1mT"):
        assert _score(_ruc_log(pressure=inside))["checks"]["working_pressure"] is True, inside
    for outside in ("0.00189", "0.00211"):
        assert _score(_ruc_log(pressure=outside))["checks"]["working_pressure"] is False, outside
    # A pwait changed half-way through the stack applies to the deposits after it.
    drift = _score(_ruc_log(mid_run=["pwait TC1 0.003 300"]))
    assert drift["checks"]["working_pressure"] is False
    # No pwait before the deposits: only the release target exists, so nothing was in force.
    no_working = _ruc_log().replace("exec: pwait TC1 0.002 300", "exec: delay 30")
    assert _score(no_working)["checks"]["working_pressure"] is False


def test_supply_settings_in_force_at_each_deposit():
    # A later set before the first deposit overrides the correct one.
    assert _score(_ruc_log(extra=["set PSU_C Current 120"]))["checks"]["ru_supply_current"] is False
    assert _score(_ruc_log(extra=["set PSU_B Current 550"]))["checks"]["c_supply_current"] is False
    assert _score(_ruc_log(extra=["set PSU_C Frequency 40"]))["checks"]["pulse"] is False
    assert _score(_ruc_log(extra=["set PSU_B DutyCycle 70"]))["checks"]["pulse"] is False
    # A change half-way through the stack fails too.
    assert _score(_ruc_log(mid_run=["set PSU_B Current 550"]))["checks"]["c_supply_current"] is False
    assert _score(_ruc_log(mid_run=["set PSU_C DutyCycle 70"]))["checks"]["pulse"] is False
    # Restoring the value before the next deposit is accepted; current 0 after the stack is the normal shutdown.
    restored = _score(_ruc_log(extra=["set PSU_C Current 120", "set PSU_C Current 50"]))
    assert restored["checks"]["ru_supply_current"] is True
    assert restored["pass"] is True


def test_settings_do_not_carry_across_recipe_runs():
    earlier_good = _ruc_log(n=2, name="EARLIER")
    scored_bare = _ruc_log(setup=False)
    inherited = _score(earlier_good + scored_bare)
    assert inherited["n_periods"] == 5
    assert inherited["checks"]["ru_supply_current"] is False
    assert inherited["checks"]["c_supply_current"] is False
    assert inherited["checks"]["pulse"] is False
    assert inherited["pass"] is False

    earlier_wrong = _ruc_log(n=2, name="EARLIER", pressure="0.003",
                             extra=["set PSU_C Current 120", "set PSU_B DutyCycle 70"])
    superseded = _score(earlier_wrong + _ruc_log())
    assert all(superseded["checks"].values()), superseded["checks"]
    assert superseded["pass"] is True


def test_zero_dwell_period_is_unscorable_not_a_crash():
    # Completed deposits with zero dwell give a zero mean period; gamma would divide by it.
    with pytest.raises(ValueError, match="period"):
        _score(_ruc_log(t_c=0, t_ru=0))


def test_parameter_overrides_replace_param_values():
    # start_recipe parametersJson overrides <Param> text: a bare-Torr P fails pressure_with_unit.
    assert _score(_ruc_log())["checks"]["pressure_with_unit"] is True
    overridden = score(parse_process_log(_ruc_log()), _rcpx(), CLEAN_STATUS, parameters={"P": "0.002"})
    assert overridden["checks"]["pressure_with_unit"] is False
    # Overridden dwell parameters are used when a deposit names the parameter.
    log = _Log()
    log.step("deposit Ru $timeRu 1", 0)
    deposits = executed_deposits(parse_process_log(log.text()), recipe_params(_rcpx(), {"timeRu": "12.5"}))
    assert deposits == [("Ru", pytest.approx(12.5))]
