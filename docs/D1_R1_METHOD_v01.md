# D1-R1 — Dirty Gravity joint beta + k fit

Status: **DIRTY / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL**.

This run preserves historical D1 (`NOT_READY`, `FIT_ELIGIBLE = []`) and uses a
new, independent D1-R1 contract with no eligibility filtering. Exactly these
ANAS sections are fitted once each: `920022`, `920028`, `920024`, `920026`,
`920042`, `920040`. Quality, external-exposure, and assignment-limitation
fields are diagnostics only. Section `920022` remains included with the note
**SECTION-SPECIFIC ASSIGNMENT REPRESENTATION LIMITATION**.

## Frozen inputs and model

The implementation resolves all operational files relative to
`DIRTY_FRLM_ROOT`. It reads `P_i`, `A_j`, ISTAT commuting, PRODUCT-LAMBDA OD
impedance/weights, and the frozen TIME_B5 path sequences from the verified
Dirty snapshot. It derives each section's path-incidence operator from the
already materialized paths; it does not recalculate routing, access nodes, OD
paths, or pair weights.

For ordered municipal OD pair `ij`:

`W_ij(beta) = P_i A_j exp(-beta c_ij)` and
`N_ij(beta,k) = k Q_v0 W_ij / sum(W)`, with
`Q_v0 = 1552629.518131 veh/day`.

For section `s`, `Yhat_s = C_s + k N_v0_s(beta)`, where `C_s` is the fixed
ISTAT contribution exposed through the same frozen path operator.

The objective is the sum of squared section-relative errors. For fixed beta,
the nonnegative scale is solved analytically:

`k*(beta) = max(0, sum(a_s b_s) / sum(a_s^2))`, with
`a_s = N_s(beta)/Y_s` and `b_s = (Y_s-C_s)/Y_s`.

The historical `0..0.2` domain is profiled on 5001 points. Beta zero is kept
only as an explicit lower-limit diagnostic because the fitted model requires
`beta > 0`. A deterministic bounded scalar refinement follows the grid scan,
and the lower and upper domain boundaries are evaluated explicitly.

## Required comparisons and diagnostics

- M0 uses historical beta `0.045953794473` and `k=1`.
- M1 holds historical beta fixed and estimates `k` analytically.
- M2 jointly profiles/refines beta and estimates `k` analytically.
- Six-fold joint and scale-only leave-one-out diagnostics never alter final M2.
- ANAS 2025 is a strict holdout and is not loaded or consulted.

The improvement label is descriptive, not a scientific gate. Its transparent
reporting heuristic is: negligible when the J_REL2 reduction rounds to 0.0%;
modest below 10% or when supporting MAPE/RMSE evidence is mixed; material at
or above 10% when both MAPE and RMSE improve.

## Execution

From the repository root, with `DIRTY_FRLM_ROOT` set:

```powershell
.venv\Scripts\python.exe scripts\run_d1_r1.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Numerical outputs are versioned under the external Dirty workspace and are
not committed to Git.
