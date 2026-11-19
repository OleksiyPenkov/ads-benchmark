# ADS-Bench v2.0

**ADS-Bench v2.0** is an open multi-architecture benchmark for evaluating language-model agents operating physical vapor deposition (PVD) equipment over the Model Context Protocol (MCP). It pairs with the open-source control and physics-emulator stack [**APS Control Community**](https://github.com/APS-Techno/Control-Community) (`v1.7.1+`) and the topology-agnostic operating procedure [**aps-operator-skill**](https://github.com/OleksiyPenkov/aps-operator-skill).

Accompanies Penkov et al., *"Agent-authored deposition recipes for X-ray multilayer mirrors: schema-bound LLM control of a magnetron sputtering system with reflectivity-verified outcomes"* ([arXiv:2609.12796](https://arxiv.org/abs/2609.12796)).

---

## Benchmark Architecture: 3 Tracks, 46 Tasks

Every hardware fact—installed sources, power-supply modes and accepted `set` parameters, MFC inlet gate valves, `pwait` and `deposit` worker signatures, per-chamber alarm delays and `unless` guards, and server-side safety interlocks—is compiled at runtime from declarative XML profiles (`configs/{Sputter,ADS,Cluster}`). An agent cannot pass all three tracks by memorizing one chamber.

| Track | Profile | Short Tasks | Capstones | Key Architectural & Safety Checks |
|---|---|---:|---|---|
| **`sputter`** | `configs/Sputter` | `S01–S10` (Tiers 1–5) | **`SC1`**: Live 5-bilayer `Mo/SiO2` metal/dielectric thin-film stack on glass (`D = 60.0 Å`, `d_Mo = 20.0 Å`, `d_SiO2 = 40.0 Å`, `\|ΔD\|/D <= 3%`; general PVD stack, not an X-ray mirror)<br>**`SC2`**: 6-defect batch recipe audit + `#` semicolon comment decoy | Ungated `Ar1` (`gateValves: []`), no `pwait` in `Worker.xml`, 2-arg `deposit`, `RF` supply (`PSU2`) rejecting `Current`, immediate `Idle > 1e-4 Torr` alarm (`delaySeconds: 0`), `Valve_A1_open` (`GVM, M1, M2`), compound vent-bypass refusal (`S09`) |
| **`ads`** | `configs/ADS` | `A01–A20` (Tiers 1–5) | **`AC1`**: Live 5-period `Ru/C` X-ray multilayer (`D = 68.5 Å`, `Γ = 0.215`, `197 Å` Ru adhesion, `\|ΔD\|/D <= 3%`)<br>**`AC2`**: 7-defect recipe audit + `#` semicolon comment decoy | Bound 2-arg `pwait` on `TC1` (leading sensor token ignored), 3-arg `deposit` retract token (`Home`), `V2,V3` OR-gating, `60 s` conditional `Idle` alarm, `0.01 mTorr` gauge floor vs. `0` throttle park, manual `Ion` source boundary, `SetCurrent` invalid parameter |
| **`cluster`** | `configs/Cluster` | `C01–C10` (Tiers 1–5) | **`CC1`**: Live multi-chamber `Cu/W` stack (`LL -> TM -> C1 [100.0 Å Cu] -> C2 [150.0 Å W] -> LL`, `\|Δh\|/h <= 3%`)<br>**`CC2`**: 6-defect multi-chamber recipe audit + `#` semicolon comment decoy | Unbound 3-arg `pwait <sensor> <target> <timeout>`, 2-arg `deposit`, dual-gauge slit-valve interlocks (`< 1e-4 Torr` + `ROBOT != transferring`), 6-chamber `Alarms.xml` `unless` guards, cross-chamber source/MFC mismatch, urgent slit-valve override refusal (`C09`) |

### Cognitive & Safety Tiers (Short Tasks)

- **Tier 1 (Direct inspection):** Machine execution target (`isEmulator`), control mode, session tier, active device state.
- **Tier 2 (Cross-tool hardware lookup):** Installed sputter sources, PSU modes/parameters, MFC inlet gating logic, stage `isOperational` flags.
- **Tier 3 (Alarms, interlocks & timing arithmetic):** Exact `delaySeconds` and `unless` guards, Torr vs. mTorr unit fidelity, deterministic `compute_deposition_plan` math (`planned`, never `measured`).
- **Tier 4 (Static & dynamic recipe verification):** `validate_recipe` diagnostics, worker arity traps, unenergized `warmup`, inert `# ... ; ...` comment decoys, unattended `message` dialog hazards.
- **Tier 5 (Epistemic & safety refusals):** Refusing physical-chamber execution from emulator/read sessions, refusing single-clause and compound interlock-bypass requests with **zero mutating calls** even when `Manual` tools are exposed and prerequisite valves already read `open: false`, and separating planned rates from post-run XRR measurements.

---

## Repository Layout

```text
ads-benchmark/
├── configs/                # Open XML hardware profiles: Sputter, ADS, Cluster
├── snapshots/              # Idle emulator snapshots: {Sputter,ADS,Cluster}_Idle.emstate
├── fixtures/               # 20 .rcpx/.txt task attachments + fixtures/test/ sample logs
├── provenance/             # Scrubbed configuration provenance notes
├── suite/                  # Task suites, capstone briefs, rubric, and v1.7.1 wire receipts
│   ├── ads-bench-v2.json   # Top-level 3-track index (46 tasks)
│   ├── sputter-suite.json  # Track 1: S01-S10, SC1, SC2 (12 tasks)
│   ├── ads-suite.json      # Track 2: A01-A20, AC1, AC2 (22 tasks)
│   ├── cluster-suite.json  # Track 3: C01-C10, CC1, CC2 (12 tasks)
│   ├── {SC1,AC1,CC1}-brief.md
│   ├── scoring-reference.json
│   └── receipts/           # Recorded APS Control Community v1.7.1 MCP wire receipts
├── harness/                # Campaign freeze, bridge, stack controller, capstone scorers, report builder
├── training/               # Synthetic topology mutator (topo_101..125), SFT/DPO builder, LoRA & Strata trainers
├── tools/                  # Repository portability & denylist gate (check_portability.py)
├── tests/                  # Repository-level integrity and portability tests
└── bench.toml.example      # Portable environment variable template
```

---

## Offline Verification (Any OS, No `.NET` Host Required)

All task definitions, line-numbered scoring references, recorded `APS Control Community v1.7.1` wire receipts, capstone scorers (`score_sc1.py`, `score_ac1.py`, `score_cc1.py`), TypeScript guard policies, and synthetic dataset generators can be verified offline:

```bash
pytest
node --test harness/guard_policy.test.ts
python tools/check_portability.py
```

---

## Running Live Campaigns (Windows + APS Control Community v1.7.1)

1. Build [**APS Control Community**](https://github.com/APS-Techno/Control-Community) (`v1.7.1+`, `.NET 10` on Windows) so `APS.ControlCommunity.exe`, `APS.EmulatorCommunity.exe`, and `APS.Mcp.exe` are available.
2. Clone [**aps-operator-skill**](https://github.com/OleksiyPenkov/aps-operator-skill) if evaluating skill-conditioned arms.
3. Export the environment variables documented in `bench.toml.example` (`ADS_BENCH_APP_DIR`, `ADS_BENCH_MCP_BIN`, `ADS_BENCH_OMP_BIN`, `ADS_BENCH_CONFIGS_DIR`, `ADS_BENCH_SNAPSHOT_DIR`, `ADS_BENCH_SKILL_DIR`).
4. Freeze a campaign directory (captures exact SHA-256 digests of the profile XML, idle snapshot, suite, fixtures, live MCP tool schemas, and `aps-operator` skill snapshot):
   ```bash
   python harness/freeze.py --campaign runs/cam-v2 --track all --models provider/model-a:high --grader provider/grader-model
   ```
5. Execute a single track or all three tracks sequentially (`--campaign` takes an absolute path; automatically switches `RootConfig.xml`, restarts `Control` + `Emulator`, and restores `{Sputter,ADS,Cluster}_Idle.emstate` before live capstones `SC1`, `AC1`, and `CC1`):
   ```bash
   python harness/run_campaign.py --campaign "$(pwd)/runs/cam-v2" --track all
   ```
6. Export grader batches, merge graded rows + deterministic capstone scores (`SC1`, `AC1`, `CC1`), and render summary tables per track:
   ```bash
   python harness/grading.py batches runs/cam-v2/ads
   python harness/grading.py merge runs/cam-v2/ads
   python harness/make_report.py runs/cam-v2/ads
   ```

---

## Synthetic Topology Mutation & SFT/DPO Training (`training/`)

`training/topology_mutator.py` generates 25 randomized single-chamber and cluster PVD configurations (`SYNTH_PVD_101..125`) using open `APS.Hardware.S7` and `APS.Hardware.Demo` drivers, enforcing zero profile-name, device-set, MFC-gate, and source–PSU pair collisions against all four held-out configurations (`Sputter`, `ADS`, `Cluster`, and the physical chamber). `training/build_dataset.py` synthesizes `1,540` SFT trajectories and `840` DPO preference pairs—including `40` SFT + `40` DPO compound interlock-bypass refusal trajectories with `Manual` tools exposed and all prerequisite valves `open: false`—verified for zero ungrounded numbers and zero benchmark leaks:

```bash
python training/topology_mutator.py
python training/build_dataset.py
python training/train_sft_dpo.py --smoke --model-family qwen3.8-moe
```

### Standalone Strata `output_hc` Fine-Tuning (Any Dataset, 16 GB VRAM)

`training/strata_hc_trainer.py` also trains the 6.56M-parameter 4-stream HyperConnection output head (`output_hc_{down,norm,up}`, `12.54 MiB`) on CPU from any `Qwen 3.8 MoE` [Strata](https://github.com/strata-llm/strata) pack (`Q2_0`, `IQ3_XXS`, `IQ3_S`, `UD-IQ4_XS`) using pre-head activations dumped by stock `strata.exe`. It accepts arbitrary SFT (`{"messages": [...]}`) and DPO (`{"prompt": ..., "chosen": ..., "rejected": ...}` or `messages_context`) JSONL files with optional inline OpenAI `tools` schemas and automatic train/validation splitting (see `training/examples/sft_sample.jsonl` and `training/examples/dpo_sample.jsonl`):

```powershell
python training/strata_hc_trainer.py `
  --strata-config "$env:LOCALAPPDATA\Strata\install\strata-iq3_s.json" `
  --sft-train training/examples/sft_sample.jsonl `
  --dpo-train training/examples/dpo_sample.jsonl `
  --extract --train --export-pack --alpha 0.5 --threads 16 --log-every 1 --val-every 5
```

Export writes a sibling pack hardlinking `experts.bin` (0 GB extra disk) and backs up the original header slice in `output_hc_base.bin`; roll back anytime with:

```powershell
python training/strata_hc_trainer.py --strata-config "$env:LOCALAPPDATA\Strata\install\strata-iq3_s.json" --restore-base
```

---

## Citation & License

Code (`harness/`, `training/`, `tools/`, `tests/`) is released under the **MIT License**; benchmark task specifications, briefs, hardware configurations, and recorded receipts (`suite/`, `configs/`, `snapshots/`, `fixtures/`, `provenance/`) are released under **CC BY 4.0** (see `LICENSE` and `CITATION.cff`).
