## Task

Author a deposition recipe for the single-chamber Batch Sputter System (`Sputter`) that produces a
periodic Mo/SiO2 multilayer stack on a glass substrate with these properties:

| Quantity | Value |
|---|---|
| Period D | 60.0 Å |
| Mo layer thickness d_Mo | 20.0 Å |
| SiO2 layer thickness d_SiO2 | 40.0 Å |
| Number of bilayers N | 5 |
| Substrate | glass |

## Process conditions

Plain argon. Argon mass-flow channel `Ar1` at 30 sccm with inlet valves `V1`, `M1` and `M2` open.
`Mo` runs on pulsed DC supply `PSU1` at 200 mA, 50 kHz and 60% duty. `SiO2` runs on RF supply
`PSU2` at 150 W. A clean glass substrate is loaded in the main chamber at base vacuum.

## Calibration constants

| Material | Supply | Mode / Setpoints | Rate (Å/s) | Target Layer (Å) | Dwell Time (s) |
|---|---|---|---|---|---|
| Mo | PSU1 | PDC: 200 mA, 50 kHz, 60% duty | 0.800 | 20.0 | 25.00 |
| SiO2 | PSU2 | RF: 150 W | 0.400 | 40.0 | 100.00 |

## Tools and rules

1. Use `describe_machine`, `describe_hardware`, `describe_dsl`, `describe_interlocks` and `describe_alarms`
   to inspect the `Sputter` hardware and command signatures before writing the recipe.
2. Validate the recipe with `validate_recipe` (`isRcpx: true`) and fix every error before uploading.
3. Every pressure literal in the script and `<Param>` definitions must carry an explicit unit suffix (`mT`, `uT`, `Torr`).
4. Note that `Sputter` has no throttle controller (`pwait` is not in `Worker.xml`) and uses 2-argument
   `deposit <source> <seconds>` via source shutter valves `Sh1` and `Sh2`.
5. At teardown, switch off `PSU1` and `PSU2`, zero `Ar1`, and close `M1`, `M2` and `V1` so the chamber
   returns cleanly to `Idle` with no active alarms.

## Deliverables

1. Upload the recipe under a name that begins with `ARM-S-`.
2. Start the recipe on the emulator, follow it to completion with `recipe_status` (`waitSeconds >= 120`),
   and report the executed layer times, planned period, and final chamber state.
