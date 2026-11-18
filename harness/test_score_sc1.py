import pytest
from score_ac1 import parse_process_log
from score_sc1 import score_sc1 as score
from test_score_ac1 import CLEAN_STATUS, _Log


def _rcpx(deposit_cmd="deposit Mo $tMo"):
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Recipe>
  <Parameters>
    <Group name="Layers">
      <Param name="tMo" caption="Mo Time" unit="sec" paramType="0">25.0</Param>
      <Param name="tSiO2" caption="SiO2 Time" unit="sec" paramType="0">100.0</Param>
    </Group>
  </Parameters>
  <Script>
{deposit_cmd}
deposit SiO2 $tSiO2
  </Script>
</Recipe>
"""


def _sputter_log(*, n=5, t_mo=25.0, t_sio2=100.0, extra=(), name="SC1-MoSiO2"):
    log = _Log()
    log.recipe("started", name)
    log.step("set PSU1 Current 200", 200)
    log.step("set PSU1 Frequency 50", 50)
    log.step("set PSU1 DutyCycle 60", 60)
    log.step("set PSU2 Power 150", 150)
    for step in extra:
        log.step(step, "ok")
    for _ in range(n):
        log.step(f"deposit Mo {t_mo}", f"ok ({t_mo}s)")
        log.step(f"deposit SiO2 {t_sio2}", f"ok ({t_sio2}s)")
    log.step("set PSU1 Current 0", 0)
    log.step("set PSU2 Power 0", 0)
    log.recipe("complete", name)
    return log.text()


def test_sc1_passes_nominal_run():
    res = score(parse_process_log(_sputter_log()), _rcpx(), CLEAN_STATUS)
    assert res["pass"] is True
    assert res["n_periods"] == 5
    assert res["period_mean_A"] == pytest.approx(60.0)
    assert res["rel_error"] == pytest.approx(0.0)
    assert all(res["checks"].values())


def test_sc1_fails_on_3arg_deposit_or_pwait_or_wrong_power():
    assert score(parse_process_log(_sputter_log()), _rcpx("deposit Mo $tMo 1"), CLEAN_STATUS)["checks"]["sputter_2arg_deposit"] is False
    assert score(parse_process_log(_sputter_log(extra=["pwait 0.003 120"])), _rcpx(), CLEAN_STATUS)["checks"]["no_pwait"] is False
    assert score(parse_process_log(_sputter_log(extra=["set PSU2 Power 100"])), _rcpx(), CLEAN_STATUS)["checks"]["sio2_supply_power"] is False


def test_sc1_ignores_prior_run_pwait():
    prior = _Log()
    prior.recipe("started", "Prior_ADS_Run")
    prior.step("pwait TC1 0.002 300", "0.002")
    prior.step("deposit Ru 30.0", "ok (30.0s)")
    prior.recipe("complete", "Prior_ADS_Run")
    combined = prior.text() + "\n" + _sputter_log()
    res = score(parse_process_log(combined), _rcpx(), CLEAN_STATUS)
    assert res["checks"]["no_pwait"] is True
    assert res["pass"] is True
