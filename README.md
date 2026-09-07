# Tesi Dirty FRLM

End-to-end demonstrator for the FVG EV charging thesis project.

## Status

**DEMONSTRATOR / NON CANONICAL**

This repository is intentionally separated from the canonical thesis
pipeline. It is used to build and validate a rapid end-to-end FRLM
proof of concept.

## Computational chain

Gravity dirty
→ provisional scaling
→ LIGHT OD demand
→ path flows
→ EV scenario
→ charging need
→ candidate sites
→ FRLM instance
→ IBM ILOG CPLEX
→ demonstrator outputs

## Data policy

Large input datasets are **not stored in GitHub**.

The runtime uses an external `DIRTY_FRLM_ROOT`, containing the verified
input snapshot and generated work/output directories.

The code must resolve input paths relative to that root and must not
depend on hard-coded `C:\Tesi\...` paths.

## Environment

- Python 3.12.10
- IBM ILOG CPLEX 22.2.0.0
- DOcplex 2.32.264

The complete Python environment is frozen in:

`requirements-runtime-v01.txt`

## Materialized Dirty demand

`LIGHT_DIRTY_OD_v01` preserves the Gravity v0 spatial structure with
`beta_dirty=0.045953794473` and applies the ratified demonstrator scale
`k_dirty=0.15`. The scale is a rounded engineering assumption anchored to the
M1 diagnostic `k*=0.154582128861`; it is not a scientific calibration.

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\run_light_dirty_od_v01.py `
  --dirty-root $env:DIRTY_FRLM_ROOT `
  --node C:\path\to\node.exe
```

## Scientific status

Observed, derived and scenario quantities must remain explicitly
distinguished.

The Dirty FRLM does not replace the canonical thesis methodology and
must not be presented as a final calibrated result.
