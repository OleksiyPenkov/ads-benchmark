# Harness Origin

The campaign runner in this directory (`campaign_core.py`, `campaign_worker.py`,
`run_campaign.py`, `campaign_guard.ts`, `guard_policy.ts`) was adapted from the
accompanying laboratory evaluation harness with the following
changes for `ADS-Bench v2.0`:

1. **Manifest-driven configuration (`manifest.py`)**: Campaign parameters, tool
   declarations, input SHA-256 digests, and runner source SHA-256 digests are loaded
   and verified from `<campaign>/campaign.json` rather than module constants.
2. **Tiered MCP bridge (`bridge.py`)**: Per-key stdio configurations (`read`,
   `recipe`, `manual`) enforce the 3-tier `APS.Mcp` capability boundary and verify
   that bridge endpoints target `127.0.0.1` / `localhost` only.
3. **Host skill snapshot (`local_tools.py`)**: `skill_read` serves files from the
   frozen `<campaign>/inputs/skill/` snapshot with reparse-point and path-traversal
   refusal.
4. **Emulator stack lifecycle (`ads_stack.py`)**: Manages `APS.EmulatorCommunity.exe`
   and `APS.ControlCommunity.exe` by owned `(PID, CreationDate)` identity, refuses to
   touch foreign processes, gates trials on `describe_machine.profile` and chamber
   `Idle` state, and captures per-trial `Process_*.log`, `.rcpx`, and
   `final_status.json` artifacts.
5. **Capstone and rubric scoring (`grading.py`, `score_ac1.py`, `score_sc1.py`,
   `score_cc1.py`, `make_report.py`)**: Deterministic process-log scorers for `SC1`,
   `AC1`, and `CC1`, plus automated Honest/Correct overrides and 3-axis grader batch
   merging.
