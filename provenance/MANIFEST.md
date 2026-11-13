# Provenance — files separated from active benchmark profiles

| File here | Profile / origin | SHA-256 | Why |
|---|---|---|---|
| `Sources.xml.before-revert-2026-10-09` | `configs/ADS/Sources.xml` | `360eafb779fe24d2e8424c7ab266081ba67b91748e5e2b93e4fc73872525588c` | Sanitized `ADS` sources snapshot before reverting flange 3 (`Gr` / `Pyrolytic Graphite` -> `C` / `Graphite`); `liftMm` retained. |
| `ARM-A-RuC-ML-68p5A-G0215-N30.rcpx` | `configs/ADS/Recipes/` | `3f79e82bb703c7c461b37ab29265be6068e5fe4b14ed23a7dd3cc221a955dced` | 30-bilayer `ADS` reference solution from which `fixtures/control-RuC-68p5A.rcpx` derives. Kept outside `configs/ADS/Recipes/` so benchmark agents cannot copy it. |
| `ARM-A-probe-psu-setpoints.rcpx` | `configs/ADS/Recipes/` | `18b684888d1e8be1ab188e47a6de2dc73f1f47b05f8bb8e4d74035a0f4101cf0` | `ADS` PSU setpoint probe recipe. Kept outside `configs/ADS/Recipes/` for the same reason. |
| `ARM-S-MoSiO2-ML-60A-N20.rcpx` | `configs/Sputter/Recipes/` | `92bc705000b2981a4a58ff47f5ad5d6435c9dd3998e79f32b9d782656ce778a2` | 20-bilayer `Sputter` reference recipe from which `N=5` `fixtures/control-Sputter-MoSiO2-60A.rcpx` derives. |
| `ARM-C-CuW-Cluster-200A.rcpx` | `configs/Cluster/Recipes/` | `492589fdd19be8b6d8e3a36a6c9374f464948e449573ff9212148eb131d277df` | `200 A` Cu / `150 A` W `Cluster` reference recipe from which `100 A` Cu `fixtures/control-Cluster-CuW.rcpx` derives. |
