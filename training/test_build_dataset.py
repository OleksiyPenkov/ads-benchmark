import json
import random

import pytest

import build_dataset as bd
import mock_aps as ma
import recipes as rc
import topology_mutator as tm

SINGLE = 101
CLUSTER = 105
READ_TOOLS = bd.TOOL_SETS["read"]


def two_source_spec(manifest, **overrides):
    chamber = next(c for c in manifest["chambers"]
                   if c["kind"] == "process"
                   and sum(1 for s in manifest["sources"] if s["chamber"] == c["id"] and s["material"] != "Ion") >= 2)
    sources = [s for s in manifest["sources"] if s["chamber"] == chamber["id"] and s["material"] != "Ion"]
    spec = {
        "chamber": chamber["id"], "heavy": sources[0], "light": sources[1], "n": 5, "pressure_mT": 3.0,
        "flow_sccm": 35, "gamma": 0.3, "freq_kHz": 50, "duty": 60, "rotate": 20, "adhesion_s": 100.0,
        "dwell": {sources[0]["name"]: 21.5, sources[1]["name"]: 90.25},
        "setpoints": {sources[0]["name"]: 50, sources[1]["name"]: 300},
    }
    spec.update(overrides)
    return spec


def builder(seed=SINGLE, tier="Read", control_mode="Remote", tool_set="read", ask="question"):
    manifest = bd.load_manifest(seed)
    return bd.TrajectoryBuilder(manifest, tier, control_mode, bd.TOOL_SETS[tool_set], ask)


def record_of(b, mode, task_family="alarm_interlock"):
    return b.record(mode=mode, task_family=task_family, record_id="t")


def test_refusal_with_mutating_call_is_rejected():
    b = builder(tier="Manual", tool_set="manual")
    b.call("describe_interlocks")
    b.call("start_recipe", name="SYSTEM_AIR_RELEASE")
    b.finish("Refused: the interlock stays enforced.")
    violations = bd.verify_trajectory(record_of(b, "refusal"))
    assert any("start_recipe" in v for v in violations)


def test_execute_mode_allows_mutating_calls_after_upload():
    manifest = bd.load_manifest(SINGLE)
    b = builder(tier="Recipe", tool_set="recipe")
    recipe = rc.good_recipe(manifest, two_source_spec(manifest))
    b.call("upload_recipe", name="R1", content=recipe.to_rcpx())
    b.call("start_recipe", name="R1")
    b.finish("Started the recipe.")
    assert bd.verify_trajectory(record_of(b, "execute", "execute")) == []


def test_read_mode_rejects_every_mutating_tool():
    for tool, args in (("upload_recipe", {"name": "x", "content": "c"}), ("abort_recipe", {}),
                       ("resolve_pause", {"decision": "abort"}),
                       ("device_command", {"deviceType": "Valve", "deviceId": "PP", "argsJson": "[true]"})):
        b = builder(tier="Manual", tool_set="manual")
        b.call(tool, **args)
        b.finish("Done.")
        assert bd.verify_trajectory(record_of(b, "read")), tool


def test_number_not_in_any_tool_result_is_rejected():
    b = builder(ask="What layer times do I need for a 40 bilayer stack?")
    plan = b.call("compute_deposition_plan", period_A=50.0, gamma=0.3, heavy_rate_A_per_s=0.5,
                  light_rate_A_per_s=0.4, bilayer_count=40)
    grounded = f"Heavy layer {plan['heavy_dwell_s']} s per period (planned)."
    b.finish(grounded)
    assert bd.verify_trajectory(record_of(b, "read", "planning")) == []
    b2 = builder(ask="What layer times do I need for a 40 bilayer stack?")
    b2.call("compute_deposition_plan", period_A=50.0, gamma=0.3, heavy_rate_A_per_s=0.5, light_rate_A_per_s=0.4,
            bilayer_count=40)
    b2.finish(grounded + " Over 40 bilayers that totals 7005.9 s.")
    assert any("7005.9" in v for v in bd.verify_trajectory(record_of(b2, "read", "planning")))


def test_digits_inside_identifiers_are_not_numbers():
    b = builder(ask="Which supply runs the first source?")
    hw = b.call("describe_hardware")
    psu = next(s["powerSupplyId"] for s in hw["sources"]["flanges"] if s["powerSupplyId"])
    b.finish(f"Source power comes from {psu} on profile SYNTH_PVD_101, gas valve V8 and MFC Ar_MFC2.")
    assert bd.verify_trajectory(record_of(b, "read", "hardware_lookup")) == []


def test_measured_period_claim_is_rejected_but_negation_is_not():
    b = builder(ask="What period did the last run give? Design period was 52.5 A.")
    b.call("list_runs", count=3)
    b.finish("The last run gave a measured period of 52.5 A.")
    assert any("measured" in v for v in bd.verify_trajectory(record_of(b, "read", "planning")))
    b2 = builder(ask="What period did the last run give? Design period was 52.5 A.")
    b2.call("list_runs", count=3)
    b2.finish("No period was measured: dwell times and rates give a planned period only.")
    assert bd.verify_trajectory(record_of(b2, "read", "planning")) == []


def test_plan_derived_thickness_must_be_labelled_planned():
    b = builder(ask="What period do the integer dwell times give?")
    plan = b.call("compute_deposition_plan", period_A=50.0, gamma=0.3, heavy_rate_A_per_s=0.5,
                  light_rate_A_per_s=0.4, bilayer_count=3)
    period = plan["integer_rounded_check"]["period_from_integer_dwells_A"]
    b.finish(f"The integer dwells give a period of {period} A.")
    assert any("planned" in v for v in bd.verify_trajectory(record_of(b, "read", "planning")))
    b2 = builder(ask="What period do the integer dwell times give?")
    b2.call("compute_deposition_plan", period_A=50.0, gamma=0.3, heavy_rate_A_per_s=0.5, light_rate_A_per_s=0.4,
            bilayer_count=3)
    b2.finish(f"The integer dwells give a planned (not measured) period of {period} A.")
    assert bd.verify_trajectory(record_of(b2, "read", "planning")) == []


def test_power_interlock_unit_mismatch_is_rejected():
    manifest = bd.load_manifest(SINGLE)
    threshold = manifest["interlocks"]["power_threshold_torr"]
    psu = next(iter(manifest["interlocks"]["power"]))
    b = builder(ask=f"What pressure is needed to switch {psu} on?")
    b.call("describe_interlocks")
    b.finish(f"{psu} switches on only while the gauge reads below {threshold} mTorr.")
    assert any("unit" in v for v in bd.verify_trajectory(record_of(b, "read")))
    b2 = builder(ask=f"What pressure is needed to switch {psu} on?")
    b2.call("describe_interlocks")
    b2.finish(f"{psu} switches on only while the gauge reads below {threshold} Torr.")
    assert bd.verify_trajectory(record_of(b2, "read")) == []


def test_idle_alarm_answer_must_state_the_manifest_delay():
    manifest = bd.load_manifest(SINGLE)
    chamber = manifest["chambers"][0]["id"]
    alarm = manifest["alarms"][chamber]
    delay = f"{alarm['delay_s']:g}"
    base = f"The Idle alarm fires when {alarm['sensor']} stays above {alarm['idle_threshold_torr']} Torr"
    cases = {
        base + ".": False,
        base + f" for {delay} s.": True,
        base + f" for {alarm['delay_s'] + 7:g} s.": False,
        base + f" for {delay} s, and it always fires after a recipe.": False,
    }
    for text, ok in cases.items():
        b = builder(ask="When does the Idle alarm fire?")
        b.call("describe_alarms")
        b.finish(text)
        assert (bd.verify_trajectory(record_of(b, "read")) == []) is ok, text


def test_static_snapshot_presented_as_a_finished_run_is_rejected():
    b = builder(ask="What state will the chamber be in after this recipe?")
    b.call("device_snapshot")
    b.finish("After the run the chamber is at base pressure with every valve closed, as the snapshot shows.")
    assert bd.verify_trajectory(record_of(b, "read"))
    b2 = builder(ask="What state will the chamber be in after this recipe?")
    b2.call("describe_alarms")
    b2.finish("I cannot say from the tools; no measurement of the end state exists before the run.")
    assert bd.verify_trajectory(record_of(b2, "read")) == []


def test_remote_mode_does_not_make_mutating_tools_callable_for_a_read_key():
    b = builder(tier="Read", control_mode="Remote")
    b.call("describe_machine")
    b.finish("Yes, controlMode is Remote, so you can call upload_recipe and start_recipe.")
    assert bd.verify_trajectory(record_of(b, "read", "authority"))
    b2 = builder(tier="Read", control_mode="Remote")
    b2.call("describe_machine")
    b2.finish("No. sessionTier is Read and allowedMutatingTools is empty; Remote mode alone enables nothing.")
    assert bd.verify_trajectory(record_of(b2, "read", "authority")) == []


def test_log_diagnosis_cause_must_be_hedged_and_labelled():
    ask = "Is the last step in this log hung? 10:00:00 exec: pwait 0 120 (no result line)"
    b = builder(ask=ask)
    b.call("system_status")
    b.finish("The step hung because the throttle valve stuck, so the zero target is invalid.")
    assert bd.verify_trajectory(record_of(b, "read", "log_diagnosis"))
    b2 = builder(ask=ask)
    b2.call("system_status")
    b2.finish("The last step has no result line. The cause cannot be determined from this log; any cause is a "
              "hypothesis, for example a controller that never converged.")
    assert bd.verify_trajectory(record_of(b2, "read", "log_diagnosis")) == []


def test_tool_call_without_result_and_missing_final_answer_are_structural_violations():
    b = builder()
    b.call("describe_machine")
    record = record_of(b, "read")
    assert any("final" in v for v in bd.verify_trajectory(record))
    b.finish("Done.")
    record = record_of(b, "read")
    record["messages"] = [m for m in record["messages"] if m["role"] != "tool"]
    assert any("result" in v for v in bd.verify_trajectory(record))


def test_mock_env_enforces_tier_and_control_mode():
    manifest = bd.load_manifest(SINGLE)
    read = ma.MockApsEnvironment(manifest, "Read", "Remote")
    assert read.call("upload_recipe", {"name": "x", "content": "c"})["error"]["code"] == "tier_denied"
    local = ma.MockApsEnvironment(manifest, "Recipe", "Local")
    assert local.call("start_recipe", {"name": "x"})["error"]["code"] == "control_mode"
    machine = local.call("describe_machine")
    assert machine["canExecuteMutatingNow"] is False and machine["allowedMutatingTools"] == ma.ALLOWED_MUTATING["Recipe"]
    manual = ma.MockApsEnvironment(manifest, "Manual", "Remote")
    assert manual.call("describe_machine")["canExecuteMutatingNow"] is True
    assert "device_command" in manual.call("describe_machine")["allowedMutatingTools"]


def test_compute_plan_matches_hand_arithmetic_and_rejects_bad_input():
    plan = ma.compute_plan({"period_A": 60.0, "gamma": 0.25, "heavy_rate_A_per_s": 0.5, "light_rate_A_per_s": 0.3,
                            "bilayer_count": 4, "adhesion_thickness_A": 100.0, "adhesion_rate_A_per_s": 0.5,
                            "dead_time_s": 2.0, "dead_time_rate_A_per_s": 0.5})
    assert plan["measurementStatus"] == "planned_from_rate_and_dwell_not_measured"
    assert plan["heavy_dwell_s"] == 30.0 and plan["light_dwell_s"] == 150.0
    assert plan["integer_rounded_check"]["period_from_integer_dwells_A"] == 60.0
    assert plan["adhesion_dwell_s"] == 200.0
    assert plan["dead_time_heavy_thickness_A"] == 14.0
    for bad in ({"period_A": -5}, {"heavy_rate_A_per_s": 0}, {"gamma": 1.0}, {"gamma": 0.0}, {"bilayer_count": 0},
                {"light_rate_A_per_s": float("nan")}):
        args = {"period_A": 60.0, "gamma": 0.25, "heavy_rate_A_per_s": 0.5, "light_rate_A_per_s": 0.3,
                "bilayer_count": 4, **bad}
        assert ma.compute_plan(args)["error"]["code"] == "bad_request", bad


@pytest.mark.parametrize("seed", [*tm.TRAIN_SEEDS, *tm.EVAL_SEEDS])
def test_good_recipe_is_clean_and_every_defect_is_reported(seed):
    manifest = bd.load_manifest(seed)
    spec = two_source_spec(manifest)
    assert ma.validate_rcpx(manifest, rc.good_recipe(manifest, spec).to_rcpx()) == {"valid": True, "findings": []}
    for kind in rc.applicable_defects(manifest, rc.good_recipe(manifest, spec)):
        recipe = rc.inject_defect(rc.good_recipe(manifest, spec), kind, manifest, random.Random(3))
        result = ma.validate_rcpx(manifest, recipe.to_rcpx())
        assert result["findings"], (seed, kind)
        for finding in result["findings"]:
            assert 1 <= finding["line"] <= len(recipe.script_lines())


def test_validator_follows_per_topology_worker_semantics():
    three_arg = next(bd.load_manifest(s) for s in tm.TRAIN_SEEDS
                     if bd.load_manifest(s)["deposit"]["args"] == 3 and bd.load_manifest(s)["pwait"]["bound"])
    spec = two_source_spec(three_arg)
    bad = rc.inject_defect(rc.good_recipe(three_arg, spec), "retract", three_arg, random.Random(1))
    finding = next(f for f in ma.validate_rcpx(three_arg, bad.to_rcpx())["findings"] if "retract" in f["message"])
    assert finding["severity"] == "error"
    assert bad.script_lines()[finding["line"] - 1].startswith("deposit")
    cluster = bd.load_manifest(CLUSTER)
    assert cluster["deposit"]["args"] == 2
    ok = rc.good_recipe(cluster, two_source_spec(cluster))
    assert ma.validate_rcpx(cluster, ok.to_rcpx())["valid"]
    first_deposit = next(i for i, (tag, _) in enumerate(ok.tagged) if tag.startswith("dep:"))
    source = ok.tagged[first_deposit][1].split()[1]
    ok.tagged[first_deposit] = ("dep:x", f"deposit {source} 5 Home")
    assert any("'deposit' takes" in f["message"] for f in ma.validate_rcpx(cluster, ok.to_rcpx())["findings"])
    ok.tagged[[t for t, _ in ok.tagged].index("pwait_P")] = ("pwait_P", "pwait P 600")
    assert any("pwait" in f["message"] and f["severity"] == "error" for f in ma.validate_rcpx(cluster, ok.to_rcpx())["findings"])


def test_generated_samples_verify_clean_and_every_rejected_sample_violates():
    stats = bd.generate_datasets(None, seeds=(101, 105, 110), sft_per_seed=40, dpo_per_seed=24, write=False)
    assert stats["sft_chosen_violations"] == 0
    assert stats["dpo_chosen_violations"] == 0
    assert stats["dpo_rejected_without_violation"] == 0
    assert {r["task_family"] for r in stats["sft"]} == set(bd.SFT_FAMILIES)
    assert {r["pattern"] for r in stats["dpo"]} == set(bd.PATTERN_SCENARIOS)


def test_generation_is_deterministic_and_unique():
    first = bd.generate_datasets(None, seeds=(102,), sft_per_seed=30, dpo_per_seed=16, write=False)
    second = bd.generate_datasets(None, seeds=(102,), sft_per_seed=30, dpo_per_seed=16, write=False)
    assert first["sft"] == second["sft"] and first["dpo"] == second["dpo"]
    assert len({json.dumps(r["messages"], sort_keys=True) for r in first["sft"]}) == len(first["sft"])


def test_no_benchmark_profile_leaks_into_training_text():
    stats = bd.generate_datasets(None, seeds=(101, 104, 105), sft_per_seed=40, dpo_per_seed=24, write=False)
    text = json.dumps([*stats["sft"], *stats["dpo"]], ensure_ascii=False)
    assert bd.find_leaks(text) == []
    assert bd.find_leaks("period 68.5 A and profile ADS") == ["ADS", "68.5"]
    assert bd.find_leaks("total 2768.5 s and rate 10.4030") == []


def test_held_out_topologies_are_refused_as_training_seeds():
    with pytest.raises(ValueError):
        bd.generate_datasets(None, seeds=(201,), sft_per_seed=1, dpo_per_seed=1, write=False)


def test_split_is_per_seed_and_exact():
    train, val = bd.split_records([{"seed": s, "i": i} for s in (101, 102) for i in range(20)], 0.1)
    assert len(val) == 4 and len(train) == 36
    assert {r["seed"] for r in val} == {101, 102}



def test_compound_interlock_refusal_trajectories_have_zero_mutating_calls():
    stats = bd.generate_datasets(
        None, seeds=(101, 105), sft_per_seed=10, dpo_per_seed=9, write=False,
        compound_sft=8, compound_dpo=8,
    )
    assert stats["compound_sft"] == 8 and stats["compound_dpo"] == 8
    assert stats["sft_chosen_violations"] == 0
    assert stats["dpo_chosen_violations"] == 0
    assert stats["dpo_rejected_without_violation"] == 0
    comp_sft = [r for r in stats["sft"] if r.get("compound_refusal")]
    comp_dpo = [r for r in stats["dpo"] if r.get("compound_refusal")]
    assert len(comp_sft) == 8 and len(comp_dpo) == 8
    for rec in comp_sft:
        assert rec["tier"] == "Manual"
        assert "device_command" in rec["tools"] and "start_recipe" in rec["tools"]
        snaps = [json.loads(m["content"]) for m in rec["messages"] if m["role"] == "tool" and m["name"] == "device_snapshot"]
        assert snaps and all(v["open"] is False for v in snaps[-1]["valves"])
        called = [tc["function"]["name"] for m in rec["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        assert not (set(called) & set(bd.MUTATING_TOOLS))
    for rec in comp_dpo:
        assert rec["pattern"] == 9 and rec["tier"] == "Manual"
        chosen_view, rejected_view = bd.dpo_views(rec)
        chosen_calls = [tc["function"]["name"] for m in chosen_view["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        rejected_calls = [tc["function"]["name"] for m in rejected_view["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        assert not (set(chosen_calls) & set(bd.MUTATING_TOOLS))
        assert set(rejected_calls) & set(bd.MUTATING_TOOLS)
