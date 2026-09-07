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

## Scientific status

Observed, derived and scenario quantities must remain explicitly
distinguished.

The Dirty FRLM does not replace the canonical thesis methodology and
must not be presented as a final calibrated result.
