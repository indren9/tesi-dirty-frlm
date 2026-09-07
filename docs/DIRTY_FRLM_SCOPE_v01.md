# Dirty FRLM v01 — Approved Scope

## Objective

Produce a complete, reproducible and presentable end-to-end FRLM
demonstrator before returning to the canonical thesis pipeline.

## D0 — Scenario contract

Centralize all non-observed assumptions and classify them as:
OBSERVED / DERIVED / SCENARIO / PLACEHOLDER.

## D1 — Updated Dirty Gravity

Preserve ISTAT commuting unchanged.

Preserve the spatial structure of Gravity v0.

Apply one provisional scaling coefficient k only to the synthetic
non-commuting component.

k must be estimated reproducibly and must not be invented.

## D2 — Dirty LIGHT demand

LIGHT_dirty =
ISTAT commuting + corrected Gravity component.

## D3 — Path flows

Use the frozen OSM access-pair path system and PRODUCT-LAMBDA weights.

Routing is not recalculated.

## D4 — EV scenario

Define a simple representative BEV scenario.

## D5 — Charging need

Convert EV path flows into path-level charging demand.

## D6 — Candidate sites

Define a finite and defensible candidate-site set.

## D7 — FRLM contract

Freeze variables, constraints and objective before implementation.

## D8 — Instance builder

Build the optimization instance for DOcplex / CPLEX.

## D9 — Validation

Required before the full run:

1. synthetic micro-instance;
2. small real FVG subset;
3. invariant checks.

## D10 — Full run

Execute the full Dirty FRLM v01 instance with CPLEX and produce:

- objective value;
- selected sites;
- capacity results where applicable;
- served demand;
- uncovered demand;
- solver gap;
- runtime;
- tables;
- maps.

## Explicitly outside Dirty FRLM v01

- Austria / Slovenia external demand;
- EE through traffic;
- canonical Gravity recalibration;
- ANAS 2025 holdout;
- stochastic SOC;
- detailed electrical-network optimization;
- definitive economic costs;
- extensive sensitivity analysis.

These remain future/canonical work.
