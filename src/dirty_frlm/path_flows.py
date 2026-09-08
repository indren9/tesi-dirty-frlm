"""D3 — deterministic dirty path-flow and directed-edge aggregation.

Consumes only the frozen Dirty_FRLM snapshot and LIGHT_DIRTY_OD_v01.
No routing, shortest-path, Gamma_OSM, PRODUCT-LAMBDA or TIME_B5
recalculation is performed.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v01"

EXPECTED_OD_COUNT = 46_010
EXPECTED_PATH_COUNT = 414_090
EXPECTED_PATHS_PER_OD = 9
EXPECTED_TOTAL_LIGHT_DIRTY = 437_968.45921965

LAMBDA_TOLERANCE = 1e-12
MASS_TOLERANCE = 1e-8

EXPECTED_INPUTS: dict[str, tuple[str, str]] = {
    "light_dirty_od": (
        r"04_OUTPUT\light_dirty_od_v01\LIGHT_DIRTY_OD_v01.csv",
        "c114d71cc8fbfd47167709899dc93e6dba823424fe9f1d25499795c6cbdd7769",
    ),
    "access_paths": (
        r"01_INPUT_SNAPSHOT\paths\OSM_OD_access_paths_v01.csv",
        "3c0a8786a05719db4ca2a4258250bde8937b8dd017d93fea0c8f4a8a101c8dd3",
    ),
    "path_offsets": (
        r"01_INPUT_SNAPSHOT\paths\OSM_OD_path_offsets_v01.npy",
        "478efd3a3f6eba6964db9f0a785dfd9405d5ae61af30e4f84538b0699a7a3a08",
    ),
    "transition_slots": (
        r"01_INPUT_SNAPSHOT\paths\OSM_OD_transition_slots_v01.npy",
        "2a6b06d21b6d3eea7132a0154bbb4c74d305a4ea582d780e07b5abeed24d2c1d",
    ),
    "sequence_contract": (
        r"01_INPUT_SNAPSHOT\paths\OSM_OD_sequence_contract_v01.json",
        "59ea1121d56c7cc1515e5aa5d40087a66fcf51ce6a1b1b763797baed70844364",
    ),
    "turn_state_edgeid": (
        r"01_INPUT_SNAPSHOT\network\osm_turn_state_edgeid_v01.npz",
        "dac4b68bb0f1aac363680202f1c0f7881c49d0af9257c727408f6e938ebee185",
    ),
    "directed_edges": (
        r"01_INPUT_SNAPSHOT\network\osm_directed_edges_v02.sqlite",
        "04809af9e13dc32de45e18a34a0917222e3ab9c0794406b1b242a8bab6c11859",
    ),
}


def resolve_dirty_root(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    raw = explicit or os.environ.get("DIRTY_FRLM_ROOT")
    if not raw:
        raise RuntimeError("DIRTY_FRLM_ROOT is required.")

    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DIRTY_FRLM_ROOT does not exist: {root}")

    return root


def sha256_file(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def input_paths(root: Path) -> dict[str, Path]:
    return {
        role: root / relative_path
        for role, (relative_path, _) in EXPECTED_INPUTS.items()
    }


def verify_frozen_inputs(root: Path) -> dict[str, object]:
    resolved = input_paths(root)
    checks: list[dict[str, object]] = []

    for role, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing D3 frozen input: {path}")

        actual = sha256_file(path)
        expected = EXPECTED_INPUTS[role][1]

        row = {
            "role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "match": actual == expected,
        }
        checks.append(row)

        if actual != expected:
            raise RuntimeError(
                f"Frozen input SHA256 mismatch for {role}: "
                f"{actual} != {expected}"
            )

    return {
        "status": "PASS",
        "files_checked": len(checks),
        "failures": 0,
        "checks": checks,
    }


def load_light_dirty(path: Path) -> pd.DataFrame:
    usecols = [
        "ORIGIN_PRO_COM",
        "ORIGIN_COMUNE",
        "DESTINATION_PRO_COM",
        "DESTINATION_COMUNE",
        "T_dirty_ij",
    ]

    frame = pd.read_csv(path, usecols=usecols)

    frame["ORIGIN_PRO_COM"] = frame["ORIGIN_PRO_COM"].astype(np.int64)
    frame["DESTINATION_PRO_COM"] = frame["DESTINATION_PRO_COM"].astype(np.int64)
    frame["T_dirty_ij"] = frame["T_dirty_ij"].astype(np.float64)

    return frame


def load_access_paths(path: Path) -> pd.DataFrame:
    usecols = [
        "path_idx",
        "path_id",
        "origin_PRO_COM",
        "origin_COMUNE",
        "destination_PRO_COM",
        "destination_COMUNE",
        "origin_access_index",
        "destination_access_index",
        "origin_access_order",
        "destination_access_order",
        "pair_weight",
        "n_B5_transitions",
        "status",
        "sequence_start",
        "sequence_end",
        "canonical_route_impedance",
        "distance_semantics",
    ]

    frame = pd.read_csv(path, usecols=usecols)

    integer_columns = [
        "path_idx",
        "origin_PRO_COM",
        "destination_PRO_COM",
        "origin_access_index",
        "destination_access_index",
        "origin_access_order",
        "destination_access_order",
        "n_B5_transitions",
        "sequence_start",
        "sequence_end",
    ]

    for column in integer_columns:
        frame[column] = frame[column].astype(np.int64)

    frame["pair_weight"] = frame["pair_weight"].astype(np.float64)

    return frame


def build_path_flow_dataframe(
    light: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    expected_od_count: int = EXPECTED_OD_COUNT,
    expected_path_count: int = EXPECTED_PATH_COUNT,
    expected_paths_per_od: int = EXPECTED_PATHS_PER_OD,
    expected_total_light_dirty: float = EXPECTED_TOTAL_LIGHT_DIRTY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    light = light.copy()
    paths = paths.copy()

    required_light = {
        "ORIGIN_PRO_COM",
        "ORIGIN_COMUNE",
        "DESTINATION_PRO_COM",
        "DESTINATION_COMUNE",
        "T_dirty_ij",
    }
    required_paths = {
        "path_idx",
        "path_id",
        "origin_PRO_COM",
        "origin_COMUNE",
        "destination_PRO_COM",
        "destination_COMUNE",
        "origin_access_index",
        "destination_access_index",
        "origin_access_order",
        "destination_access_order",
        "pair_weight",
        "n_B5_transitions",
        "status",
        "sequence_start",
        "sequence_end",
        "canonical_route_impedance",
        "distance_semantics",
    }

    if not required_light.issubset(light.columns):
        raise RuntimeError("LIGHT_DIRTY_OD_v01 schema is incompatible.")

    if not required_paths.issubset(paths.columns):
        raise RuntimeError("OSM access-path schema is incompatible.")

    if len(light) != expected_od_count:
        raise RuntimeError(
            f"Expected {expected_od_count} OD rows; found {len(light)}."
        )

    light_keys = ["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"]

    if light[light_keys].duplicated().any():
        raise RuntimeError("LIGHT_DIRTY_OD contains duplicate OD keys.")

    light_numeric = light["T_dirty_ij"].to_numpy(dtype=np.float64)

    if not np.isfinite(light_numeric).all():
        raise RuntimeError("LIGHT_DIRTY_OD contains non-finite flow.")

    if (light_numeric < 0).any():
        raise RuntimeError("LIGHT_DIRTY_OD contains negative flow.")

    total_light_dirty = float(light_numeric.sum())

    if not math.isclose(
        total_light_dirty,
        expected_total_light_dirty,
        rel_tol=0.0,
        abs_tol=MASS_TOLERANCE,
    ):
        raise RuntimeError(
            "LIGHT_DIRTY_OD global total differs from frozen contract: "
            f"{total_light_dirty} vs {expected_total_light_dirty}"
        )

    if len(paths) != expected_path_count:
        raise RuntimeError(
            f"Expected {expected_path_count} access paths; found {len(paths)}."
        )

    paths = paths.sort_values("path_idx", kind="stable").reset_index(drop=True)

    expected_idx = np.arange(expected_path_count, dtype=np.int64)
    actual_idx = paths["path_idx"].to_numpy(dtype=np.int64)

    if not np.array_equal(actual_idx, expected_idx):
        raise RuntimeError("path_idx is not canonical 0..N-1.")

    if paths["path_id"].duplicated().any():
        raise RuntimeError("Duplicate path_id detected.")

    if not paths["status"].eq("FINITE_RECONSTRUCTED").all():
        raise RuntimeError("Non-finite frozen path detected.")

    if not paths["canonical_route_impedance"].eq("TIME_B5").all():
        raise RuntimeError("Unexpected canonical route impedance.")

    if not paths["distance_semantics"].eq("PATH_ATTRIBUTE").all():
        raise RuntimeError("Unexpected distance semantics.")

    weights = paths["pair_weight"].to_numpy(dtype=np.float64)

    if not np.isfinite(weights).all():
        raise RuntimeError("Non-finite PRODUCT-LAMBDA weight detected.")

    if (weights < 0).any():
        raise RuntimeError("Negative PRODUCT-LAMBDA weight detected.")

    path_group_keys = ["origin_PRO_COM", "destination_PRO_COM"]

    path_counts = paths.groupby(
        path_group_keys,
        sort=False,
    ).size()

    if len(path_counts) != expected_od_count:
        raise RuntimeError(
            f"Expected {expected_od_count} path OD groups; found {len(path_counts)}."
        )

    if not path_counts.eq(expected_paths_per_od).all():
        raise RuntimeError(
            f"Expected exactly {expected_paths_per_od} paths for every OD."
        )

    lambda_sums = paths.groupby(
        path_group_keys,
        sort=False,
    )["pair_weight"].sum()

    lambda_residual = np.abs(
        lambda_sums.to_numpy(dtype=np.float64) - 1.0
    )
    max_lambda_residual = float(lambda_residual.max(initial=0.0))

    if max_lambda_residual > LAMBDA_TOLERANCE:
        raise RuntimeError(
            f"PRODUCT-LAMBDA OD mass failed: {max_lambda_residual}"
        )

    merged = paths.merge(
        light[
            [
                "ORIGIN_PRO_COM",
                "DESTINATION_PRO_COM",
                "T_dirty_ij",
            ]
        ],
        left_on=["origin_PRO_COM", "destination_PRO_COM"],
        right_on=["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if merged["T_dirty_ij"].isna().any():
        raise RuntimeError("At least one frozen path has no LIGHT_DIRTY_OD match.")

    merged["lambda_product"] = merged["pair_weight"].astype(np.float64)
    merged["path_flow_veh_day"] = (
        merged["T_dirty_ij"].astype(np.float64)
        * merged["lambda_product"]
    )

    output_columns = [
        "path_idx",
        "path_id",
        "origin_PRO_COM",
        "origin_COMUNE",
        "destination_PRO_COM",
        "destination_COMUNE",
        "origin_access_index",
        "destination_access_index",
        "origin_access_order",
        "destination_access_order",
        "sequence_start",
        "sequence_end",
        "n_B5_transitions",
        "lambda_product",
        "T_dirty_ij",
        "path_flow_veh_day",
    ]

    output = merged[output_columns].copy()

    flow_values = output["path_flow_veh_day"].to_numpy(dtype=np.float64)

    negative_flows = int(np.count_nonzero(flow_values < 0))
    nonfinite_flows = int(np.count_nonzero(~np.isfinite(flow_values)))

    if negative_flows:
        raise RuntimeError("Negative access-path flow detected.")

    if nonfinite_flows:
        raise RuntimeError("Non-finite access-path flow detected.")

    flow_by_od = output.groupby(
        path_group_keys,
        sort=False,
    )["path_flow_veh_day"].sum()

    dirty_by_od = output.groupby(
        path_group_keys,
        sort=False,
    )["T_dirty_ij"].first()

    od_residuals = (
        flow_by_od.to_numpy(dtype=np.float64)
        - dirty_by_od.to_numpy(dtype=np.float64)
    )

    max_od_mass_residual = float(
        np.max(np.abs(od_residuals), initial=0.0)
    )

    if max_od_mass_residual > MASS_TOLERANCE:
        raise RuntimeError(
            f"Per-OD path-flow mass failed: {max_od_mass_residual}"
        )

    global_path_flow = float(flow_values.sum())
    global_mass_residual = global_path_flow - total_light_dirty

    if abs(global_mass_residual) > MASS_TOLERANCE:
        raise RuntimeError(
            f"Global path-flow mass failed: {global_mass_residual}"
        )

    qa = {
        "status": "PASS",
        "od_count": int(len(path_counts)),
        "access_pair_path_count": int(len(output)),
        "paths_per_od": expected_paths_per_od,
        "total_light_dirty_veh_day": total_light_dirty,
        "sum_access_path_flows_veh_day": global_path_flow,
        "global_mass_residual": global_mass_residual,
        "max_od_mass_residual": max_od_mass_residual,
        "max_lambda_sum_residual": max_lambda_residual,
        "negative_flows": negative_flows,
        "nonfinite_flows": nonfinite_flows,
        "unreachable_paths": 0,
    }

    return output, qa


def validate_sequence_arrays(
    path_frame: pd.DataFrame,
    offsets: np.ndarray,
    transition_slot_count: int,
) -> dict[str, object]:
    path_count = len(path_frame)

    if offsets.ndim != 1:
        raise RuntimeError("Offsets array must be one-dimensional.")

    if len(offsets) != path_count + 1:
        raise RuntimeError("Offsets cardinality does not match path count.")

    if int(offsets[0]) != 0:
        raise RuntimeError("Offsets must start at zero.")

    if int(offsets[-1]) != transition_slot_count:
        raise RuntimeError("Offsets terminal value differs from slot count.")

    if np.any(np.diff(offsets) < 0):
        raise RuntimeError("Offsets are not monotonic.")

    starts = path_frame["sequence_start"].to_numpy(dtype=np.int64)
    ends = path_frame["sequence_end"].to_numpy(dtype=np.int64)
    transitions = path_frame["n_B5_transitions"].to_numpy(dtype=np.int64)

    if not np.array_equal(starts, np.asarray(offsets[:-1], dtype=np.int64)):
        raise RuntimeError("sequence_start differs from frozen offsets.")

    if not np.array_equal(ends, np.asarray(offsets[1:], dtype=np.int64)):
        raise RuntimeError("sequence_end differs from frozen offsets.")

    if not np.array_equal(
        transitions,
        np.diff(np.asarray(offsets, dtype=np.int64)),
    ):
        raise RuntimeError("n_B5_transitions differs from frozen offsets.")

    return {
        "status": "PASS",
        "path_count": path_count,
        "offset_count": int(len(offsets)),
        "transition_slot_count": int(transition_slot_count),
    }


def directed_edge_domain(
    sqlite_path: Path,
) -> dict[str, int]:
    connection = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro",
        uri=True,
    )

    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS n,
                MIN(edge_id) AS min_id,
                MAX(edge_id) AS max_id
            FROM directed_edges
            """
        ).fetchone()
    finally:
        connection.close()

    if row is None or row[0] <= 0:
        raise RuntimeError("directed_edges table is empty.")

    return {
        "count": int(row[0]),
        "min_edge_id": int(row[1]),
        "max_edge_id": int(row[2]),
    }


def aggregate_edge_flows_arrays(
    path_flows: np.ndarray,
    offsets: np.ndarray,
    transition_slots: np.ndarray,
    edgeid_data: np.ndarray,
    *,
    max_edge_id: int,
    target_transitions_per_chunk: int = 2_000_000,
    validate_indicator_semantics: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path_flows = np.asarray(path_flows, dtype=np.float64)
    offsets = np.asarray(offsets)

    if len(offsets) != len(path_flows) + 1:
        raise RuntimeError("Path-flow / offsets cardinality mismatch.")

    if int(offsets[-1]) != len(transition_slots):
        raise RuntimeError("Transition-slot count differs from offsets.")

    if target_transitions_per_chunk <= 0:
        raise ValueError("target_transitions_per_chunk must be positive.")

    if not np.isfinite(path_flows).all() or (path_flows < 0).any():
        raise RuntimeError("Invalid path flow supplied to edge aggregation.")

    edge_flow = np.zeros(max_edge_id + 1, dtype=np.float64)
    transition_occurrences = np.zeros(max_edge_id + 1, dtype=np.int64)

    n_paths = len(path_flows)
    n_slots = len(transition_slots)

    p = 0
    chunks = 0
    processed_slots = 0
    min_slot_seen: int | None = None
    max_slot_seen: int | None = None
    min_edge_seen: int | None = None
    max_edge_seen: int | None = None
    duplicate_path_edge_occurrences = 0

    while p < n_paths:
        start_slot = int(offsets[p])

        target_slot = min(
            start_slot + target_transitions_per_chunk,
            n_slots,
        )

        q = int(
            np.searchsorted(
                offsets,
                target_slot,
                side="right",
            )
            - 1
        )

        if q <= p:
            q = p + 1

        if q > n_paths:
            q = n_paths

        end_slot = int(offsets[q])

        lengths = np.diff(
            np.asarray(offsets[p : q + 1], dtype=np.int64)
        )

        chunk_weights = np.repeat(
            path_flows[p:q],
            lengths,
        )

        slot_chunk = np.asarray(
            transition_slots[start_slot:end_slot]
        )

        if len(slot_chunk) != len(chunk_weights):
            raise RuntimeError("Chunk path-weight expansion mismatch.")

        if len(slot_chunk):
            chunk_min_slot = int(slot_chunk.min())
            chunk_max_slot = int(slot_chunk.max())

            min_slot_seen = (
                chunk_min_slot
                if min_slot_seen is None
                else min(min_slot_seen, chunk_min_slot)
            )
            max_slot_seen = (
                chunk_max_slot
                if max_slot_seen is None
                else max(max_slot_seen, chunk_max_slot)
            )

            if chunk_min_slot < 0 or chunk_max_slot >= len(edgeid_data):
                raise RuntimeError(
                    "Transition slot references outside edgeid data array."
                )

            edge_ids = np.asarray(
                edgeid_data[slot_chunk],
                dtype=np.int64,
            )

            chunk_min_edge = int(edge_ids.min())
            chunk_max_edge = int(edge_ids.max())

            min_edge_seen = (
                chunk_min_edge
                if min_edge_seen is None
                else min(min_edge_seen, chunk_min_edge)
            )
            max_edge_seen = (
                chunk_max_edge
                if max_edge_seen is None
                else max(max_edge_seen, chunk_max_edge)
            )

            if chunk_min_edge < 0 or chunk_max_edge > max_edge_id:
                raise RuntimeError(
                    "Frozen path references edge_id outside SQLite domain."
                )

            if validate_indicator_semantics:
                path_ids = np.repeat(
                    np.arange(p, q, dtype=np.uint64),
                    lengths,
                )

                edge_ids_u64 = edge_ids.astype(
                    np.uint64,
                    copy=False,
                )

                pair_keys = (
                    (path_ids << np.uint64(32))
                    | edge_ids_u64
                )

                unique_pair_count = int(
                    np.unique(pair_keys).size
                )

                duplicate_count = (
                    len(pair_keys)
                    - unique_pair_count
                )

                duplicate_path_edge_occurrences += duplicate_count

                if duplicate_count:
                    raise RuntimeError(
                        "Frozen path violates indicator semantics: "
                        f"{duplicate_count} repeated (path_idx, edge_id) "
                        f"occurrences detected in path range [{p}, {q})."
                    )

            edge_flow += np.bincount(
                edge_ids,
                weights=chunk_weights,
                minlength=max_edge_id + 1,
            )

            transition_occurrences += np.bincount(
                edge_ids,
                minlength=max_edge_id + 1,
            ).astype(np.int64, copy=False)

        processed_slots += len(slot_chunk)
        chunks += 1
        p = q

    if processed_slots != n_slots:
        raise RuntimeError("Not all frozen transition slots were processed.")

    qa = {
        "status": "PASS",
        "chunks": chunks,
        "paths_processed": n_paths,
        "transition_slots_processed": processed_slots,
        "transition_slots_expected": n_slots,
        "min_slot_seen": min_slot_seen,
        "max_slot_seen": max_slot_seen,
        "min_edge_id_seen": min_edge_seen,
        "max_edge_id_seen": max_edge_seen,
        "positive_flow_edge_ids": int(np.count_nonzero(edge_flow > 0)),
        "sum_edge_flows_veh_day": float(edge_flow.sum()),
        "transition_occurrences": int(transition_occurrences.sum()),
        "indicator_semantics_validated": validate_indicator_semantics,
        "duplicate_path_edge_occurrences": duplicate_path_edge_occurrences,
    }

    return edge_flow, transition_occurrences, qa


def write_path_flow_csv(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    frame.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )


def write_edge_flow_csv(
    sqlite_path: Path,
    edge_flow: np.ndarray,
    transition_occurrences: np.ndarray,
    output_path: Path,
    *,
    sql_chunk_rows: int = 100_000,
) -> dict[str, object]:
    connection = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro",
        uri=True,
    )

    query = """
        SELECT
            edge_id,
            edge_uid,
            segment_uid,
            way_id,
            seq,
            u,
            v,
            way_direction,
            length_m,
            highway,
            edge_role
        FROM directed_edges
        ORDER BY edge_id
    """

    first = True
    rows_written = 0
    output_flow_sum = 0.0

    present = np.zeros(len(edge_flow), dtype=bool)

    try:
        for chunk in pd.read_sql_query(
            query,
            connection,
            chunksize=sql_chunk_rows,
        ):
            ids = chunk["edge_id"].to_numpy(dtype=np.int64)

            if ids.min() < 0 or ids.max() >= len(edge_flow):
                raise RuntimeError("SQLite edge_id outside aggregation array.")

            present[ids] = True

            flow = edge_flow[ids]
            occurrences = transition_occurrences[ids]

            keep = flow > 0

            if not np.any(keep):
                continue

            chunk = chunk.loc[keep].copy()
            chunk["dirty_flow_veh_day"] = flow[keep]
            chunk["transition_occurrences"] = occurrences[keep]

            chunk.to_csv(
                output_path,
                mode="w" if first else "a",
                header=first,
                index=False,
                lineterminator="\n",
                float_format="%.15g",
            )

            first = False
            rows_written += len(chunk)
            output_flow_sum += float(chunk["dirty_flow_veh_day"].sum())

    finally:
        connection.close()

    missing_referenced = int(
        np.count_nonzero(
            (edge_flow > 0)
            & ~present
        )
    )

    if missing_referenced:
        raise RuntimeError(
            f"{missing_referenced} referenced edge IDs are absent from SQLite."
        )

    expected_positive = int(np.count_nonzero(edge_flow > 0))

    if rows_written != expected_positive:
        raise RuntimeError(
            "Positive edge-flow output cardinality mismatch."
        )

    return {
        "status": "PASS",
        "rows_written": rows_written,
        "positive_edge_ids_expected": expected_positive,
        "referenced_edge_ids_missing_from_sqlite": missing_referenced,
        "output_sum_edge_flows_veh_day": output_flow_sum,
    }

def capture_frozen_input_states(root: Path) -> dict[str, dict[str, object]]:
    states: dict[str, dict[str, object]] = {}

    for role, path in input_paths(root).items():
        stat = path.stat()

        states[role] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    return states


def compute_d3_flow_state(
    root: Path,
    *,
    target_transitions_per_chunk: int = 2_000_000,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    paths = input_paths(root)

    light = load_light_dirty(
        paths["light_dirty_od"]
    )

    access_paths = load_access_paths(
        paths["access_paths"]
    )

    path_flows, path_qa = build_path_flow_dataframe(
        light,
        access_paths,
    )

    offsets = np.load(
        paths["path_offsets"],
        mmap_mode="r",
    )

    transition_slots = np.load(
        paths["transition_slots"],
        mmap_mode="r",
    )

    sequence_qa = validate_sequence_arrays(
        path_flows,
        offsets,
        transition_slot_count=len(transition_slots),
    )

    domain = directed_edge_domain(
        paths["directed_edges"]
    )

    with np.load(paths["turn_state_edgeid"]) as archive:
        if "data" not in archive.files:
            raise RuntimeError(
                "Frozen turn-state edge-id NPZ has no data array."
            )

        edgeid_data = archive["data"]

        if edgeid_data.ndim != 1:
            raise RuntimeError(
                "Frozen turn-state edge-id data must be one-dimensional."
            )

        edge_flow, occurrences, edge_qa = aggregate_edge_flows_arrays(
            path_flows[
                "path_flow_veh_day"
            ].to_numpy(dtype=np.float64),
            offsets,
            transition_slots,
            edgeid_data,
            max_edge_id=domain["max_edge_id"],
            target_transitions_per_chunk=target_transitions_per_chunk,
            validate_indicator_semantics=True,
        )

    if not edge_qa["indicator_semantics_validated"]:
        raise RuntimeError(
            "Indicator semantics were not validated."
        )

    if edge_qa["duplicate_path_edge_occurrences"] != 0:
        raise RuntimeError(
            "Repeated directed edge detected inside a frozen path."
        )

    edge_qa["directed_edge_domain"] = domain

    edge_qa["weighted_mean_directed_edges_per_vehicle"] = (
        edge_qa["sum_edge_flows_veh_day"]
        / path_qa["sum_access_path_flows_veh_day"]
    )

    edge_qa["mass_interpretation"] = (
        "Sum of edge flows is not OD mass. "
        "Each vehicle contributes to every directed edge on its path."
    )

    qa = {
        "path_flow": path_qa,
        "sequence_arrays": sequence_qa,
        "edge_aggregation": edge_qa,
    }

    return (
        path_flows,
        edge_flow,
        occurrences,
        qa,
    )


def materialize_run_payload(
    root: Path,
    run_dir: Path,
    *,
    target_transitions_per_chunk: int = 2_000_000,
) -> dict[str, object]:
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        path_frame,
        edge_flow,
        occurrences,
        qa,
    ) = compute_d3_flow_state(
        root,
        target_transitions_per_chunk=target_transitions_per_chunk,
    )

    path_output = (
        run_dir
        / "DIRTY_PATH_FLOWS_v01.csv"
    )

    edge_output = (
        run_dir
        / "DIRTY_EDGE_FLOWS_v01.csv"
    )

    write_path_flow_csv(
        path_frame,
        path_output,
    )

    edge_write_qa = write_edge_flow_csv(
        input_paths(root)["directed_edges"],
        edge_flow,
        occurrences,
        edge_output,
    )

    qa["edge_output"] = edge_write_qa

    return {
        "path_output": str(path_output),
        "edge_output": str(edge_output),
        "path_sha256": sha256_file(path_output),
        "edge_sha256": sha256_file(edge_output),
        "path_size_bytes": path_output.stat().st_size,
        "edge_size_bytes": edge_output.stat().st_size,
        "qa": qa,
    }

def git_info(repo_root: Path) -> dict[str, object]:
    import subprocess

    base = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
    ]

    branch = subprocess.check_output(
        base + ["branch", "--show-current"],
        text=True,
    ).strip()

    commit = subprocess.check_output(
        base + ["rev-parse", "HEAD"],
        text=True,
    ).strip()

    status = subprocess.check_output(
        base + ["status", "--porcelain=v1"],
        text=True,
    )

    return {
        "repo": str(repo_root),
        "branch": branch,
        "commit": commit,
        "status_porcelain": status,
    }


def run_d3_production(
    root: Path,
    repo_root: Path,
    *,
    target_transitions_per_chunk: int = 2_000_000,
) -> dict[str, object]:
    import json
    import shutil
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc)

    output_dir = root / "04_OUTPUT" / "path_flows_v01"
    reporting_dir = root / "05_REPORTING" / "path_flows_v01"
    work_root = root / "03_WORK" / "path_flows_v01"

    if output_dir.exists():
        raise FileExistsError(
            f"Append-only output target already exists: {output_dir}"
        )

    if reporting_dir.exists():
        raise FileExistsError(
            f"Append-only reporting target already exists: {reporting_dir}"
        )

    git_pre = git_info(repo_root)

    if git_pre["branch"] != "feat/d3-path-flows":
        raise RuntimeError(
            "D3 production must run on feat/d3-path-flows."
        )

    sha_verification = verify_frozen_inputs(root)

    input_states_before = capture_frozen_input_states(root)

    run_id = started.strftime("run_%Y%m%dT%H%M%SZ")

    run_a = work_root / f"{run_id}_A"
    run_b = work_root / f"{run_id}_B"

    result_a = materialize_run_payload(
        root,
        run_a,
        target_transitions_per_chunk=target_transitions_per_chunk,
    )

    result_b = materialize_run_payload(
        root,
        run_b,
        target_transitions_per_chunk=target_transitions_per_chunk,
    )

    path_deterministic = (
        result_a["path_sha256"]
        == result_b["path_sha256"]
    )

    edge_deterministic = (
        result_a["edge_sha256"]
        == result_b["edge_sha256"]
    )

    if not path_deterministic or not edge_deterministic:
        raise RuntimeError(
            "D3 deterministic rerun failed."
        )

    input_states_after = capture_frozen_input_states(root)

    if input_states_before != input_states_after:
        raise RuntimeError(
            "Frozen input state changed during D3."
        )

    path_qa = result_a["qa"]["path_flow"]
    edge_qa = result_a["qa"]["edge_aggregation"]
    edge_output_qa = result_a["qa"]["edge_output"]

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    reporting_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    final_path = output_dir / "DIRTY_PATH_FLOWS_v01.csv"
    final_edge = output_dir / "DIRTY_EDGE_FLOWS_v01.csv"

    shutil.copyfile(
        result_a["path_output"],
        final_path,
    )

    shutil.copyfile(
        result_a["edge_output"],
        final_edge,
    )

    final_path_sha = sha256_file(final_path)
    final_edge_sha = sha256_file(final_edge)

    if final_path_sha != result_a["path_sha256"]:
        raise RuntimeError(
            "Published path-flow SHA differs from staging."
        )

    if final_edge_sha != result_a["edge_sha256"]:
        raise RuntimeError(
            "Published edge-flow SHA differs from staging."
        )

    qa = {
        "status": "PASS",
        "scientific_status": (
            "ENGINEERING / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL"
        ),
        "sha_verification": sha_verification,
        "run_A": result_a["qa"],
        "run_B": result_b["qa"],
        "determinism": {
            "status": "PASS",
            "path_flow_bytes_equal": path_deterministic,
            "edge_flow_bytes_equal": edge_deterministic,
            "path_sha256_A": result_a["path_sha256"],
            "path_sha256_B": result_b["path_sha256"],
            "edge_sha256_A": result_a["edge_sha256"],
            "edge_sha256_B": result_b["edge_sha256"],
        },
        "frozen_input_states_unchanged": True,
        "input_states_before": input_states_before,
        "input_states_after": input_states_after,
        "routing_recomputed": False,
        "gravity_recomputed": False,
        "product_lambda_recomputed": False,
        "time_b5_recomputed": False,
        "gamma_osm_modified": False,
        "light_dirty_od_modified": False,
        "anas_used": False,
        "canonical_artifacts_modified": False,
        "D4_started": False,
    }

    qa_path = (
        reporting_dir
        / "DIRTY_PATH_FLOWS_v01_QA_evidence.json"
    )

    qa_path.write_text(
        json.dumps(
            qa,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    qa_sha = sha256_file(qa_path)

    report_path = (
        reporting_dir
        / "DIRTY_PATH_FLOWS_v01_GATE_REPORT.md"
    )

    report = f"""# D3_PATH_FLOWS_FINAL_GATE_REPORT

VERDICT = PASS

INPUT_LIGHT_DIRTY_OD = LIGHT_DIRTY_OD_v01 / FROZEN
OD_COUNT = {path_qa["od_count"]}
ACCESS_PAIR_PATH_COUNT = {path_qa["access_pair_path_count"]}
PATHS_PER_OD = {path_qa["paths_per_od"]}

TOTAL_LIGHT_DIRTY = {path_qa["total_light_dirty_veh_day"]:.15f}
SUM_ACCESS_PATH_FLOWS = {path_qa["sum_access_path_flows_veh_day"]:.15f}
GLOBAL_MASS_RESIDUAL = {path_qa["global_mass_residual"]:.17g}
MAX_OD_MASS_RESIDUAL = {path_qa["max_od_mass_residual"]:.17g}
MAX_LAMBDA_SUM_RESIDUAL = {path_qa["max_lambda_sum_residual"]:.17g}

UNREACHABLE_PATHS = {path_qa["unreachable_paths"]}
NEGATIVE_FLOWS = {path_qa["negative_flows"]}
NONFINITE_FLOWS = {path_qa["nonfinite_flows"]}

EDGE_FLOW_OUTPUT = YES
POSITIVE_FLOW_DIRECTED_EDGES = {edge_output_qa["rows_written"]}
SUM_EDGE_FLOWS_VEH_DAY = {edge_qa["sum_edge_flows_veh_day"]:.15f}
WEIGHTED_MEAN_DIRECTED_EDGES_PER_VEHICLE = {edge_qa["weighted_mean_directed_edges_per_vehicle"]:.15f}
DUPLICATE_PATH_EDGE_OCCURRENCES = {edge_qa["duplicate_path_edge_occurrences"]}

DETERMINISM = PASS
CANONICAL_ARTIFACTS_MODIFIED = NO

ROUTING_RECOMPUTED = NO
GRAVITY_RECOMPUTED = NO
PRODUCT_LAMBDA_RECOMPUTED = NO
TIME_B5_RECOMPUTED = NO
ANAS_USED = NO
D4_STARTED = NO

GIT_BRANCH = {git_pre["branch"]}
EXECUTION_BASE_COMMIT = {git_pre["commit"]}

PATH_FLOW_SHA256 = {final_path_sha}
EDGE_FLOW_SHA256 = {final_edge_sha}
QA_EVIDENCE_SHA256 = {qa_sha}

RECOMMENDATION = RATIFY_D3
"""

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    report_sha = sha256_file(report_path)

    manifest_path = (
        reporting_dir
        / "DIRTY_PATH_FLOWS_v01_manifest.json"
    )

    manifest = {
        "manifest": "DIRTY_PATH_FLOWS_v01_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scientific_status": (
            "ENGINEERING / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL"
        ),
        "git_execution": git_pre,
        "inputs": sha_verification,
        "determinism": qa["determinism"],
        "canonical_artifacts_modified": False,
        "D4_started": False,
        "artifacts": [
            {
                "path": str(final_path.relative_to(root)),
                "size_bytes": final_path.stat().st_size,
                "sha256": final_path_sha,
            },
            {
                "path": str(final_edge.relative_to(root)),
                "size_bytes": final_edge.stat().st_size,
                "sha256": final_edge_sha,
            },
            {
                "path": str(qa_path.relative_to(root)),
                "size_bytes": qa_path.stat().st_size,
                "sha256": qa_sha,
            },
            {
                "path": str(report_path.relative_to(root)),
                "size_bytes": report_path.stat().st_size,
                "sha256": report_sha,
            },
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "status": "PASS",
        "verdict": "PASS",
        "recommendation": "RATIFY_D3",
        "path_flow_csv": str(final_path),
        "path_flow_sha256": final_path_sha,
        "edge_flow_csv": str(final_edge),
        "edge_flow_sha256": final_edge_sha,
        "qa_evidence": str(qa_path),
        "qa_evidence_sha256": qa_sha,
        "gate_report": str(report_path),
        "gate_report_sha256": report_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "path_flow_qa": path_qa,
        "edge_aggregation_qa": edge_qa,
        "edge_output_qa": edge_output_qa,
        "determinism": qa["determinism"],
        "canonical_artifacts_modified": False,
        "D4_started": False,
    }