import pytest
from score_ac1 import parse_process_log
from score_cc1 import score_cc1 as score
from test_score_ac1 import CLEAN_STATUS, _Log


def _rcpx(p_c1="3mT", p_c2="2mT"):
    return f"""<?xml version="1.0" encoding="utf-8"?>
<Recipe>
  <Parameters>
    <Group name="Process">
      <Param name="pC1" caption="C1 Pressure" unit="mTorr" paramType="0">{p_c1}</Param>
      <Param name="pC2" caption="C2 Pressure" unit="mTorr" paramType="0">{p_c2}</Param>
    </Group>
  </Parameters>
  <Script>
transfer LL C1
pwait C1_TC $pC1 180
deposit C1_Cu 80.0
pwait C1_TC 0 60
transfer C1 C2
pwait C2_TC $pC2 180
deposit W 200.0
pwait C2_TC 0 60
transfer C2 LL
  </Script>
</Recipe>
"""


def _cluster_log(*, t_cu=80.0, t_w=200.0, p_c1="0.003", p_c2="0.002", release="0",
                 transfers=(("LL", "C1"), ("C1", "C2"), ("C2", "LL")), name="CC1-CuW"):
    log = _Log()
    log.recipe("started", name)
    log.step(f"transfer {transfers[0][0]} {transfers[0][1]}", "ok")
    log.step(f"pwait C1_TC {p_c1} 180", p_c1)
    log.step("set C1_PSU1 Current 250", 250)
    log.step("set C1_PSU1 Frequency 50", 50)
    log.step("set C1_PSU1 DutyCycle 60", 60)
    log.step(f"deposit C1_Cu {t_cu}", f"ok ({t_cu}s)")
    log.step("set C1_PSU1 Current 0", 0)
    log.step(f"pwait C1_TC {release} 60", release)
    log.step(f"transfer {transfers[1][0]} {transfers[1][1]}", "ok")
    log.step(f"pwait C2_TC {p_c2} 180", p_c2)
    log.step("set C2_PSU Current 200", 200)
    log.step("set C2_PSU Frequency 50", 50)
    log.step("set C2_PSU DutyCycle 60", 60)
    log.step(f"deposit W {t_w}", f"ok ({t_w}s)")
    log.step("set C2_PSU Current 0", 0)
    log.step(f"pwait C2_TC {release} 60", release)
    log.step(f"transfer {transfers[2][0]} {transfers[2][1]}", "ok")
    log.recipe("complete", name)
    return log.text()


def test_cc1_passes_nominal_run():
    res = score(parse_process_log(_cluster_log()), _rcpx(), CLEAN_STATUS)
    assert res["pass"] is True
    assert res["cu_thickness_A"] == pytest.approx(100.0)
    assert res["w_thickness_A"] == pytest.approx(150.0)
    assert all(res["checks"].values())


def test_cc1_fails_on_wrong_transfer_or_unreleased_throttle():
    bad_xfer = _cluster_log(transfers=(("LL", "C1"), ("C1", "C3"), ("C3", "LL")))
    assert score(parse_process_log(bad_xfer), _rcpx(), CLEAN_STATUS)["checks"]["transfer_sequence"] is False
    unreleased = _cluster_log(release="0.002")
    assert score(parse_process_log(unreleased), _rcpx(), CLEAN_STATUS)["checks"]["throttles_released"] is False


def test_cc1_accepts_sensor_names_and_six_chamber_status():
    cluster_status = {
        "chambers": [{"id": cid, "state": "Idle"} for cid in ("TM", "LL", "C1", "C2", "C3", "C4")],
        "activeAlarms": [],
        "isRecipeRunning": False,
    }
    log_text = _cluster_log().replace("pwait C1_TC", "pwait C1_pm").replace("pwait C2_TC", "pwait C2_pm")
    rcpx = _rcpx().replace("pwait C1_TC", "pwait C1_pm").replace("pwait C2_TC", "pwait C2_pm")
    res = score(parse_process_log(log_text), rcpx, cluster_status)
    assert res["pass"] is True
    assert res["end_state_clean"] is True
    assert res["checks"]["c1_working_pressure"] is True
    assert res["checks"]["c2_working_pressure"] is True
    assert res["checks"]["throttles_released"] is True
