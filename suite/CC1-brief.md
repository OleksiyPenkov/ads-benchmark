## Task

Author a multi-chamber deposition recipe for the Radial Cluster Tool (`Cluster`) that transfers a
wafer from the load lock (`LL`) through the transfer module (`TM`) into `C1` for a 100.0 Å `Cu` layer,
then into `C2` for a 150.0 Å `W` layer, and returns the wafer to `LL`:

| Layer | Chamber | Source | Target Thickness (Å) | Rate (Å/s) | Nominal Dwell (s) |
|---|---|---|---|---|---|
| 1 (bottom) | C1 | C1_Cu | 100.0 | 1.250 | 80.00 |
| 2 (top) | C2 | W | 150.0 | 0.750 | 200.00 |

## Process conditions

- **Chamber `C1` (`C1_Cu` on `C1_PSU1`):** Pulsed DC at 250 mA, 50 kHz, 60% duty; `C1_Ar = 40 sccm`
  with `C1_V` open; active pressure regulation `pwait C1_pm 3mT 300`.
- **Chamber `C2` (`W` on `C2_PSU`):** Pulsed DC at 200 mA, 50 kHz, 60% duty; `C2_Ar = 40 sccm`
  with `C2_V` open; active pressure regulation `pwait C2_pm 2mT 300`.
- **Transfers (`ROBOT`):** Wafer starts in `LL` at high vacuum. Use `transfer LL C1`, `transfer C1 C2`,
  and `transfer C2 LL` with slit valves (`SV_LL`, `SV_C1`, `SV_C2`) opened before each transfer and
  closed before process gas flow or plasma ignition.
- **Teardown per chamber:** Turn supply off, release the chamber throttle to `0` (`pwait C1_pm 0 1` /
  `pwait C2_pm 0 1`), zero the MFC, close the gas inlet valve (`C1_V` / `C2_V`), and wait for high
  vacuum (`< 1e-4 Torr`) before opening a slit valve into `TM`.

## Deliverables

1. Upload the recipe under a name that begins with `ARM-C-`.
2. Start the recipe on the emulator, follow it to completion with `recipe_status` (`waitSeconds >= 120`),
   and report the executed layer thicknesses (`planned`), transfer sequence, and final state across all chambers.
