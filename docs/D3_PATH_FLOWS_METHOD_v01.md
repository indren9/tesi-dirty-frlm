# D3_PATH_FLOWS_METHOD_v01

## Status

D3 is an engineering demonstrator stage.

Scientific status:

`ENGINEERING / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL`

This stage consumes frozen upstream artifacts and does not recalibrate or
recompute demand, routing, Gamma_OSM, PRODUCT-LAMBDA, or TIME_B5.

## Frozen demand input

`LIGHT_DIRTY_OD_v01`

For municipal OD pair `(i,j)`:

`T_dirty_ij = C_ISTAT_ij + 0.15 * N_v0_ij`

Frozen total:

`437968.45921965 veh/day`

Ordered municipal OD count:

`46010`

## Frozen path system

Each municipality has three frozen access points.

Each ordered municipal OD therefore has exactly nine ordered access pairs.

Total access-pair paths:

`46010 * 9 = 414090`

For access pair `(a,b)`:

`lambda_product_ijab = lambda_i,a * lambda_j,b`

and

`path_flow_ijab = T_dirty_ij * lambda_product_ijab`

For each municipal OD:

`sum_ab lambda_product_ijab = 1`

therefore:

`sum_ab path_flow_ijab = T_dirty_ij`

The nine access pairs are a frozen zonal multi-access representation.
They are NOT route-choice alternatives.

Each access pair uses its single already frozen TIME_B5-optimal path.

## Sequence representation

The path system uses a CSR-like frozen representation.

For `path_idx = p`:

`transition_slots[offsets[p]:offsets[p+1]]`

returns the ordered transition slots of that path.

Each transition slot is a zero-based position in the frozen B5 arrays.

Directed physical edge identity is resolved by:

`edge_id = osm_turn_state_edgeid_v01.npz:data[slot]`

then:

`edge_id -> directed_edges.edge_id`

in:

`osm_directed_edges_v02.sqlite`

No shortest-path reconstruction is permitted in D3.

## Directed-edge aggregation

The frozen D3 contract is:

`F_e = sum_p path_flow_p * I(e in p)`

where `I(e in p)` is an indicator.

The implementation therefore validates that a frozen path does not contain
the same directed `edge_id` more than once before accepting the aggregation.

The sum of directed-edge flows is NOT expected to equal the municipal
OD-flow total. One vehicle contributes its flow to every directed edge
traversed along its frozen path.

## Mandatory QA

D3 requires:

- 46010 ordered municipal OD;
- 414090 access-pair paths;
- exactly 9 paths per OD;
- non-negative finite PRODUCT-LAMBDA weights;
- PRODUCT-LAMBDA sum per OD equal to 1 within numerical tolerance;
- non-negative finite access-path flows;
- access-path flow sum per OD equal to `T_dirty_ij`;
- global access-path-flow sum equal to `437968.45921965 veh/day`;
- zero unreachable frozen paths;
- frozen sequence offsets consistent with path metadata;
- all transition slots valid against the frozen B5 edge-id array;
- all referenced `edge_id` values present in `directed_edges`;
- deterministic rerun;
- frozen inputs verified by SHA256;
- canonical artifacts unchanged.

## Scope exclusions

D3 does not:

- recompute Gravity;
- modify LIGHT_DIRTY_OD_v01;
- recompute routing;
- modify Gamma_OSM;
- recompute PRODUCT-LAMBDA;
- recompute TIME_B5;
- use ANAS;
- start D4 or any downstream phase.