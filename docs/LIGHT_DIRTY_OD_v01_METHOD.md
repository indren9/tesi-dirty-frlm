# LIGHT_DIRTY_OD_v01 — materialization method

Status: **ENGINEERING / DEMONSTRATOR / PROVISIONAL / NON-CANONICAL**.

This step implements the Chat Madre ratification without reopening D1 or D1-R1.
Both remain `NOT_READY / PRESERVED HISTORICAL`.

## Fixed assumptions

- `beta_dirty_demonstrator = 0.045953794473`
- `k_dirty_demonstrator = 0.15`
- `Q_v0 = 1,552,629.518131 veh/day`
- `Q_dirty_noncommuting = 232,894.42771965 veh/day`

`k=0.15` is a rounded engineering scaling assumption anchored to the M1
diagnostic `k*=0.154582128861`. It is **not scientifically calibrated**.

## Materialization

For each of the 46,010 ordered inter-municipal OD pairs:

`N_dirty_ij = 0.15 * N_v0_ij`

`T_dirty_ij = C_ISTAT_ij + N_dirty_ij`

The Gravity v0 vector is deterministically materialized from frozen `P_i`,
`A_j`, and the already stored `time_s_PRODUCT_LAMBDA`, using the fixed beta and
the frozen `Q_v0`. This evaluates the existing Gravity v0 definition; it does
not recalculate PRODUCT-LAMBDA.

No network, shortest path, access-set, TIME_B5, Gamma_OSM, or G_OSM_operativo
computation is executed by this step.

## Outputs

The external Dirty workspace receives append-only, versioned CSV and XLSX
datasets under `04_OUTPUT/light_dirty_od_v01`, plus QA evidence, manifest, log,
and the final gate report under `05_REPORTING/light_dirty_od_v01`.

The CSV is the deterministic machine-readable artifact. The XLSX is a readable
companion containing the same OD values and a compact metadata/QA sheet.

## Gate

The run fails closed unless cardinality, uniqueness, non-negativity, commuting
immutability, mass balance, row identities, determinism, snapshot integrity,
canonical-source stability, append-only publication, and Git cleanliness all
pass.
