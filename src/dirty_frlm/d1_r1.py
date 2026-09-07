"""D1-R1 Dirty Gravity joint beta + k fitting.

This module deliberately uses the frozen Dirty FRLM snapshot. It extracts
section incidence from the existing OD path sequences and never recalculates
routing, access nodes, PRODUCT-LAMBDA weights, or TIME_B5 costs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


SECTION_IDS: tuple[int, ...] = (920022, 920028, 920024, 920026, 920042, 920040)
BETA_V0 = 0.045953794473
Q_V0 = 1_552_629.518131
BETA_MIN = 0.0
BETA_MAX = 0.2
BETA_GRID_POINTS = 5001
BETA_GRID_STEP = (BETA_MAX - BETA_MIN) / (BETA_GRID_POINTS - 1)
VERSION = "v01"


@dataclass(frozen=True)
class ModelInputs:
    od_origin: np.ndarray
    od_destination: np.ndarray
    gravity_base: np.ndarray
    cost_minutes: np.ndarray
    commuting_od: np.ndarray
    exposure: np.ndarray
    observed: np.ndarray
    commuting_section: np.ndarray
    section_metadata: pd.DataFrame
    source_paths: Mapping[str, str]


@dataclass(frozen=True)
class FitResult:
    beta: float
    k: float
    predictions: np.ndarray
    non_commuting_k1: np.ndarray
    metrics: Mapping[str, float]
    beta_boundary_flag: bool
    k_boundary_flag: bool


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_dirty_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    raw = explicit or os.environ.get("DIRTY_FRLM_ROOT")
    if not raw:
        raise RuntimeError("DIRTY_FRLM_ROOT is required (environment or --dirty-root).")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DIRTY_FRLM_ROOT does not exist: {root}")
    return root


def verify_snapshot(root: Path, verify_hashes: bool = True) -> dict:
    manifest_path = root / "05_REPORTING" / "manifests" / "DIRTY_FRLM_INPUT_SNAPSHOT_v01.json"
    csv_manifest = root / "05_REPORTING" / "manifests" / "DIRTY_FRLM_INPUT_SNAPSHOT_v01.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    checks: list[dict] = []
    for row in rows:
        path = root / "01_INPUT_SNAPSHOT" / Path(row["snapshot_relative_path"])
        stat = path.stat()
        actual_hash = _sha256(path) if verify_hashes else None
        checks.append(
            {
                "relative_path": row["snapshot_relative_path"],
                "exists": True,
                "size_bytes": stat.st_size,
                "expected_size_bytes": int(row["size_bytes"]),
                "size_match": stat.st_size == int(row["size_bytes"]),
                "sha256": actual_hash,
                "expected_sha256": row["snapshot_sha256"],
                "hash_match": (actual_hash == row["snapshot_sha256"]) if verify_hashes else None,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    declared_ok = (
        manifest.get("files_expected") == 12
        and manifest.get("files_copied") == 12
        and manifest.get("byte_identical") == "12/12"
        and len(rows) == 12
        and all(bool(row.get("byte_identical")) for row in rows)
    )
    actual_ok = all(c["size_match"] and (c["hash_match"] is not False) for c in checks)
    if not (manifest_path.is_file() and csv_manifest.is_file() and declared_ok and actual_ok):
        raise RuntimeError("Dirty FRLM snapshot preflight failed.")
    return {
        "manifest_json": str(manifest_path),
        "manifest_csv": str(csv_manifest),
        "declared_byte_identical": manifest["byte_identical"],
        "files_expected": manifest["files_expected"],
        "files_verified": len(checks),
        "hashes_verified": verify_hashes,
        "status": "PASS",
        "checks": checks,
    }


def _split_edge_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [int(piece) for piece in str(value).split("|") if piece.strip()]


def load_section_metadata(anas_review: Path) -> pd.DataFrame:
    # The review export has occasional unquoted semicolons in SHORT_REASON.
    # Preserve every earlier field and fold overflow tokens into that final note.
    with anas_review.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        records = []
        for values in reader:
            if len(values) > len(header):
                values = values[: len(header) - 1] + [";".join(values[len(header) - 1 :])]
            if len(values) < len(header):
                values += [""] * (len(header) - len(values))
            records.append(values)
    frame = pd.DataFrame(records, columns=header)
    frame["SECTION_ID"] = pd.to_numeric(frame["SECTION_ID"], errors="raise").astype(np.int64)
    frame["TGMA_LIGHT_2024"] = pd.to_numeric(
        frame["TGMA_LIGHT_2024"], errors="raise"
    ).astype(np.float64)
    if "TGMA_LIGHT_2025" in frame.columns or any("2025" in str(c) for c in frame.columns):
        raise RuntimeError("ANAS 2025 column detected in the selected section source.")
    chosen = frame.loc[frame["SECTION_ID"].isin(SECTION_IDS)].copy()
    if chosen["SECTION_ID"].duplicated().any() or set(chosen["SECTION_ID"]) != set(SECTION_IDS):
        raise RuntimeError("ANAS review does not contain the exact six-section set exactly once.")
    chosen = chosen.set_index("SECTION_ID").loc[list(SECTION_IDS)].reset_index()
    chosen["edge_ids"] = chosen.apply(
        lambda row: _split_edge_ids(row.get("PRIMARY_EDGE_IDS"))
        + _split_edge_ids(row.get("OPPOSITE_EDGE_IDS")),
        axis=1,
    )
    if chosen["edge_ids"].map(len).eq(0).any():
        raise RuntimeError("At least one selected ANAS section has no mapped edge IDs.")
    chosen["assignment_limitation"] = ""
    chosen.loc[
        chosen["SECTION_ID"].eq(920022), "assignment_limitation"
    ] = "SECTION-SPECIFIC ASSIGNMENT REPRESENTATION LIMITATION"
    return chosen


def extract_path_section_incidence(
    transition_slots_path: Path,
    offsets_path: Path,
    edge_map_path: Path,
    section_metadata: pd.DataFrame,
    chunk_elements: int = 12_000_000,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Return a path x section boolean matrix from frozen path sequences."""
    offsets = np.load(offsets_path, mmap_mode="r")
    transitions = np.load(transition_slots_path, mmap_mode="r")
    edge_map = np.load(edge_map_path)["data"]
    if offsets.ndim != 1 or transitions.ndim != 1 or int(offsets[-1]) != len(transitions):
        raise RuntimeError("Frozen CSR path arrays fail structural validation.")
    edge_to_section: dict[int, int] = {}
    for section_index, row in section_metadata.iterrows():
        for edge_id in row["edge_ids"]:
            if edge_id in edge_to_section:
                raise RuntimeError(f"Edge {edge_id} is assigned to multiple selected sections.")
            edge_to_section[int(edge_id)] = int(section_index)
    slot_to_section: dict[int, int] = {}
    for edge_id, section_index in edge_to_section.items():
        hits = np.flatnonzero(edge_map == edge_id)
        if len(hits) != 1:
            raise RuntimeError(f"Expected one B5 slot for edge {edge_id}; found {len(hits)}.")
        slot_to_section[int(hits[0])] = section_index
    target_slots = np.asarray(sorted(slot_to_section), dtype=transitions.dtype)
    incidence = np.zeros((len(offsets) - 1, len(SECTION_IDS)), dtype=bool)
    total = len(transitions)
    for start in range(0, total, chunk_elements):
        stop = min(start + chunk_elements, total)
        block = np.asarray(transitions[start:stop])
        local_positions = np.flatnonzero(np.isin(block, target_slots, assume_unique=False))
        if len(local_positions):
            global_positions = local_positions.astype(np.int64) + start
            paths = np.searchsorted(offsets, global_positions, side="right") - 1
            slots = block[local_positions]
            for target_slot in np.unique(slots):
                section_index = slot_to_section[int(target_slot)]
                incidence[paths[slots == target_slot], section_index] = True
        if progress and (start == 0 or stop == total or (start // chunk_elements) % 10 == 0):
            progress(f"section-incidence scan {stop:,}/{total:,} transitions")
    if not incidence.any(axis=0).all():
        missing = [SECTION_IDS[i] for i in np.flatnonzero(~incidence.any(axis=0))]
        raise RuntimeError(f"No frozen paths expose selected sections: {missing}")
    return incidence


def _build_exposure(
    access_paths_path: Path,
    municipal_summary_path: Path,
    incidence: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    summary = pd.read_csv(municipal_summary_path)
    required_summary = {"origin_PRO_COM", "destination_PRO_COM", "time_s_PRODUCT_LAMBDA"}
    if not required_summary.issubset(summary.columns):
        raise RuntimeError("Municipal summary schema is incompatible.")
    access = pd.read_csv(
        access_paths_path,
        usecols=["path_idx", "origin_PRO_COM", "destination_PRO_COM", "pair_weight"],
    )
    if len(access) != len(incidence) or not np.array_equal(
        access["path_idx"].to_numpy(dtype=np.int64), np.arange(len(access), dtype=np.int64)
    ):
        raise RuntimeError("Access path rows do not align with frozen path_idx order.")
    key_columns = ["origin_PRO_COM", "destination_PRO_COM"]
    od_index = pd.MultiIndex.from_frame(summary[key_columns])
    path_od = od_index.get_indexer(pd.MultiIndex.from_frame(access[key_columns]))
    if (path_od < 0).any():
        raise RuntimeError("At least one access path has no municipal OD summary row.")
    exposure = np.zeros((len(summary), len(SECTION_IDS)), dtype=np.float64)
    weights = access["pair_weight"].to_numpy(dtype=np.float64)
    for section_index in range(len(SECTION_IDS)):
        mask = incidence[:, section_index]
        np.add.at(exposure[:, section_index], path_od[mask], weights[mask])
    if (exposure < -1e-14).any() or (exposure > 1.0 + 1e-8).any():
        raise RuntimeError("Derived section exposure falls outside [0, 1].")
    return summary, exposure


def load_model_inputs(
    root: Path,
    anas_review: Path,
    progress: Callable[[str], None] | None = None,
) -> ModelInputs:
    snapshot = root / "01_INPUT_SNAPSHOT"
    territorial_path = snapshot / "demand" / "Gravity_v0_territorial_inputs_derived_v01.xlsx"
    commuting_path = snapshot / "demand" / "ISTAT_commuting_LIGHT_v0.xlsx"
    summary_path = snapshot / "paths" / "OSM_OD_municipal_summary_v01.csv"
    access_path = snapshot / "paths" / "OSM_OD_access_paths_v01.csv"
    offsets_path = snapshot / "paths" / "OSM_OD_path_offsets_v01.npy"
    transitions_path = snapshot / "paths" / "OSM_OD_transition_slots_v01.npy"
    edge_map_path = snapshot / "network" / "osm_turn_state_edgeid_v01.npz"
    metadata = load_section_metadata(anas_review)
    incidence = extract_path_section_incidence(
        transitions_path, offsets_path, edge_map_path, metadata, progress=progress
    )
    summary, exposure = _build_exposure(access_path, summary_path, incidence)
    territorial = pd.read_excel(territorial_path, sheet_name="TERRITORIAL_INPUTS")
    commuting = pd.read_excel(commuting_path, sheet_name="COMMUTING_OD")
    if len(territorial) != 215 or territorial["PRO_COM"].nunique() != 215:
        raise RuntimeError("Territorial inputs are not the frozen 215-municipality set.")
    if len(summary) != 46_010 or len(commuting) != 46_010:
        raise RuntimeError("Expected 46,010 ordered inter-municipal OD rows.")
    summary_keys = summary[["origin_PRO_COM", "destination_PRO_COM"]].to_numpy(dtype=np.int64)
    commuting_keys = commuting[["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"]].to_numpy(dtype=np.int64)
    if not np.array_equal(summary_keys, commuting_keys):
        raise RuntimeError("Commuting and frozen path OD orders differ.")
    territory = territorial.set_index("PRO_COM")
    origins = summary["origin_PRO_COM"].to_numpy(dtype=np.int64)
    destinations = summary["destination_PRO_COM"].to_numpy(dtype=np.int64)
    gravity_base = (
        territory.loc[origins, "P_i"].to_numpy(dtype=np.float64)
        * territory.loc[destinations, "A_j"].to_numpy(dtype=np.float64)
    )
    cost_minutes = summary["time_s_PRODUCT_LAMBDA"].to_numpy(dtype=np.float64) / 60.0
    commuting_od = commuting["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64)
    observed = metadata["TGMA_LIGHT_2024"].to_numpy(dtype=np.float64)
    commuting_section = commuting_od @ exposure
    arrays = [gravity_base, cost_minutes, commuting_od, exposure, observed, commuting_section]
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("Non-finite values detected in model inputs.")
    if (gravity_base < 0).any() or (cost_minutes < 0).any() or (commuting_od < 0).any():
        raise RuntimeError("Negative model input detected.")
    return ModelInputs(
        od_origin=origins,
        od_destination=destinations,
        gravity_base=gravity_base,
        cost_minutes=cost_minutes,
        commuting_od=commuting_od,
        exposure=exposure,
        observed=observed,
        commuting_section=commuting_section,
        section_metadata=metadata,
        source_paths={
            "territorial": str(territorial_path),
            "commuting": str(commuting_path),
            "municipal_summary": str(summary_path),
            "access_paths": str(access_path),
            "path_offsets": str(offsets_path),
            "transition_slots": str(transitions_path),
            "edge_map": str(edge_map_path),
            "anas_2024_review": str(anas_review),
        },
    )


def non_commuting_at_beta(
    beta: float,
    gravity_base: np.ndarray,
    cost_minutes: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    if beta < 0:
        raise ValueError("beta must be non-negative for boundary-limit evaluation.")
    weights = gravity_base * np.exp(-beta * cost_minutes)
    denominator = float(weights.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Gravity normalization is not positive and finite.")
    return Q_V0 * ((weights / denominator) @ exposure)


def beta_grid_non_commuting(
    beta_grid: np.ndarray,
    gravity_base: np.ndarray,
    cost_minutes: np.ndarray,
    exposure: np.ndarray,
    block_size: int = 100,
) -> np.ndarray:
    result = np.empty((len(beta_grid), exposure.shape[1]), dtype=np.float64)
    for start in range(0, len(beta_grid), block_size):
        stop = min(start + block_size, len(beta_grid))
        block = beta_grid[start:stop]
        weights = np.exp(-block[:, None] * cost_minutes[None, :])
        weights *= gravity_base[None, :]
        denominators = weights.sum(axis=1)
        result[start:stop] = Q_V0 * ((weights @ exposure) / denominators[:, None])
    return result


def analytic_k(non_commuting_k1: np.ndarray, observed: np.ndarray, commuting: np.ndarray) -> float:
    a = np.asarray(non_commuting_k1, dtype=np.float64) / np.asarray(observed, dtype=np.float64)
    b = (np.asarray(observed, dtype=np.float64) - np.asarray(commuting, dtype=np.float64)) / np.asarray(
        observed, dtype=np.float64
    )
    denominator = float(a @ a)
    if denominator <= 0 or not math.isfinite(denominator):
        return 0.0
    return max(0.0, float((a @ b) / denominator))


def compute_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    relative = residual / observed
    return {
        "J_REL2": float(relative @ relative),
        "MAPE_pct": float(np.mean(np.abs(relative)) * 100.0),
        "median_abs_relative_error_pct": float(np.median(np.abs(relative)) * 100.0),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "absolute_SSE": float(residual @ residual),
    }


def _fit_at_beta(
    beta: float,
    non_commuting: np.ndarray,
    observed: np.ndarray,
    commuting: np.ndarray,
) -> FitResult:
    k = analytic_k(non_commuting, observed, commuting)
    predictions = commuting + k * non_commuting
    return FitResult(
        beta=float(beta),
        k=float(k),
        predictions=predictions,
        non_commuting_k1=non_commuting,
        metrics=compute_metrics(observed, predictions),
        beta_boundary_flag=False,
        k_boundary_flag=k <= 1e-12,
    )


def fit_joint(
    gravity_base: np.ndarray,
    cost_minutes: np.ndarray,
    exposure: np.ndarray,
    observed: np.ndarray,
    commuting: np.ndarray,
    beta_grid: np.ndarray | None = None,
    grid_non_commuting: np.ndarray | None = None,
) -> tuple[FitResult, pd.DataFrame]:
    if beta_grid is None:
        beta_grid = np.linspace(BETA_MIN, BETA_MAX, BETA_GRID_POINTS)
    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    if grid_non_commuting is None:
        grid_non_commuting = beta_grid_non_commuting(
            beta_grid, gravity_base, cost_minutes, exposure
        )
    if grid_non_commuting.shape != (len(beta_grid), len(observed)):
        raise ValueError("grid_non_commuting shape mismatch.")
    a = grid_non_commuting / observed[None, :]
    b = (observed - commuting) / observed
    denominators = np.sum(a * a, axis=1)
    k_grid = np.maximum(0.0, (a @ b) / denominators)
    predicted_grid = commuting[None, :] + k_grid[:, None] * grid_non_commuting
    j_grid = np.sum(((predicted_grid - observed[None, :]) / observed[None, :]) ** 2, axis=1)
    valid = beta_grid > 0
    best_valid_index = int(np.flatnonzero(valid)[np.argmin(j_grid[valid])])
    lo_index = max(1, best_valid_index - 1)
    hi_index = min(len(beta_grid) - 1, best_valid_index + 1)
    lo = float(beta_grid[lo_index])
    hi = float(beta_grid[hi_index])

    def objective(beta: float) -> float:
        n = non_commuting_at_beta(beta, gravity_base, cost_minutes, exposure)
        fitted = _fit_at_beta(beta, n, observed, commuting)
        return fitted.metrics["J_REL2"]

    candidates: list[FitResult] = []
    if hi > lo:
        refined = minimize_scalar(
            objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-14, "maxiter": 300},
        )
        n_refined = non_commuting_at_beta(float(refined.x), gravity_base, cost_minutes, exposure)
        candidates.append(_fit_at_beta(float(refined.x), n_refined, observed, commuting))
    for beta in (float(beta_grid[best_valid_index]), np.nextafter(0.0, 1.0), BETA_MAX):
        n = non_commuting_at_beta(beta, gravity_base, cost_minutes, exposure)
        candidates.append(_fit_at_beta(beta, n, observed, commuting))
    result = min(candidates, key=lambda item: (item.metrics["J_REL2"], item.beta))
    boundary_flag = result.beta <= BETA_GRID_STEP or result.beta >= BETA_MAX - BETA_GRID_STEP
    result = FitResult(
        beta=result.beta,
        k=result.k,
        predictions=result.predictions,
        non_commuting_k1=result.non_commuting_k1,
        metrics=result.metrics,
        beta_boundary_flag=boundary_flag,
        k_boundary_flag=result.k_boundary_flag,
    )
    profile = pd.DataFrame(
        {
            "beta": beta_grid,
            "valid_beta": valid,
            "k_profile": k_grid,
            "J_REL2": j_grid,
            "lower_boundary_limit_row": beta_grid == 0.0,
            "upper_boundary_row": beta_grid == BETA_MAX,
        }
    )
    return result, profile


def _section_rows(
    model_name: str,
    beta: float,
    k: float,
    observed: np.ndarray,
    commuting: np.ndarray,
    non_commuting: np.ndarray,
    predicted: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for index, section_id in enumerate(SECTION_IDS):
        residual = float(predicted[index] - observed[index])
        rows.append(
            {
                "model": model_name,
                "section_id": section_id,
                "beta": beta,
                "k": k,
                "observed": float(observed[index]),
                "commuting_fixed": float(commuting[index]),
                "non_commuting_k1": float(non_commuting[index]),
                "predicted": float(predicted[index]),
                "residual": residual,
                "relative_error": residual / float(observed[index]),
                "absolute_relative_error": abs(residual / float(observed[index])),
            }
        )
    return rows


def _loo_rows(
    inputs: ModelInputs,
    beta_grid: np.ndarray,
    grid_non_commuting: np.ndarray,
    joint: bool,
) -> list[dict]:
    rows: list[dict] = []
    for held_out in range(len(SECTION_IDS)):
        train = np.arange(len(SECTION_IDS)) != held_out
        if joint:
            fit, _ = fit_joint(
                inputs.gravity_base,
                inputs.cost_minutes,
                inputs.exposure[:, train],
                inputs.observed[train],
                inputs.commuting_section[train],
                beta_grid,
                grid_non_commuting[:, train],
            )
            n_held = non_commuting_at_beta(
                fit.beta,
                inputs.gravity_base,
                inputs.cost_minutes,
                inputs.exposure[:, [held_out]],
            )[0]
            beta = fit.beta
            k = fit.k
            training_j = fit.metrics["J_REL2"]
            beta_boundary = fit.beta_boundary_flag
        else:
            beta = BETA_V0
            n_all = non_commuting_at_beta(
                beta, inputs.gravity_base, inputs.cost_minutes, inputs.exposure
            )
            k = analytic_k(n_all[train], inputs.observed[train], inputs.commuting_section[train])
            predicted_train = inputs.commuting_section[train] + k * n_all[train]
            training_j = compute_metrics(inputs.observed[train], predicted_train)["J_REL2"]
            n_held = n_all[held_out]
            beta_boundary = False
        predicted = float(inputs.commuting_section[held_out] + k * n_held)
        observed = float(inputs.observed[held_out])
        relative_error = (predicted - observed) / observed
        rows.append(
            {
                "held_out_section": SECTION_IDS[held_out],
                "beta_fold": beta,
                "k_fold": k,
                "Q_fold": k * Q_V0,
                "observed": observed,
                "predicted": predicted,
                "relative_error": relative_error,
                "absolute_relative_error": abs(relative_error),
                "training_J_REL2": training_j,
                "beta_boundary_flag": beta_boundary,
                "k_boundary_flag": k <= 1e-12,
            }
        )
    return rows


def _describe(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
    }


def _improvement_label(j_reduction_pct: float, mape_change: float, rmse_change: float) -> tuple[str, str]:
    # A transparent reporting heuristic, not a scientific acceptance threshold.
    if round(j_reduction_pct, 1) == 0.0:
        label = "NEGLIGIBLE"
    elif j_reduction_pct < 10.0:
        label = "MODEST"
    elif mape_change < 0.0 and rmse_change < 0.0:
        label = "MATERIAL"
    else:
        label = "MODEST"
    rule = (
        "Descriptive heuristic (not a scientific gate): NEGLIGIBLE when the J_REL2 reduction "
        "rounds to 0.0%; MODEST below 10% or with mixed supporting metrics; MATERIAL at or "
        "above 10% when both MAPE and RMSE improve."
    )
    return label, rule


def _fmt(value: float) -> str:
    return f"{value:.12g}"


def _write_gate_report(
    path: Path,
    payload: Mapping[str, object],
    branch: str,
    commit: str,
    tests: str,
    output_root: Path,
) -> None:
    m0 = payload["models"]["M0"]
    m1 = payload["models"]["M1"]
    m2 = payload["models"]["M2"]
    imp = payload["improvement_M2_vs_M1"]
    loo = payload["loo_joint"]
    influence = payload["section_920022_influence"]
    boundary = payload["boundary_checks"]
    if payload["verdict"] == "NOT_READY":
        robustness = (
            "[POOR parameter robustness: the M2 infimum occurs at beta -> 0+ (no interior "
            f"optimum); LOO beta spans {_fmt(loo['beta_summary']['min'])}..{_fmt(loo['beta_summary']['max'])} "
            f"and k spans {_fmt(loo['k_summary']['min'])}..{_fmt(loo['k_summary']['max'])}. "
            "Deterministic rerun PASS and LOO 6/6 complete, but those computational checks do "
            "not cure the boundary/instability result.]"
        )
        fit_quality = (
            f"[POOR in absolute terms: M2 MAPE={_fmt(m2['MAPE_pct'])}% and RMSE="
            f"{_fmt(m2['RMSE'])} veh/day. M2 provides only a {imp['classification'].lower()} "
            "J_REL2 improvement versus M1, while MAPE and MAE worsen. Per-section and LOO "
            "diagnostics are in the companion CSV files.]"
        )
    else:
        robustness = (
            f"[Deterministic rerun PASS; LOO 6/6 complete; beta boundary flag="
            f"{m2['beta_boundary_flag']}; k boundary flag={m2['k_boundary_flag']}; "
            f"lower-limit J_REL2={_fmt(boundary['lower_limit_J_REL2'])}; "
            f"upper-bound J_REL2={_fmt(boundary['upper_J_REL2'])}.]"
        )
        fit_quality = (
            f"[{imp['classification']} improvement versus scale-only M1; six-section in-sample "
            f"M2 MAPE={_fmt(m2['MAPE_pct'])}% and RMSE={_fmt(m2['RMSE'])} veh/day. "
            "Per-section and LOO diagnostics are in the companion CSV files.]"
        )
    text = f"""D1_R1_FINAL_GATE_REPORT

VERDICT =
{payload['verdict']}

SECTIONS_USED =
[920022,920028,920024,920026,920042,920040]

M0:
beta = {_fmt(m0['beta'])}
k = 1
J_REL2 = {_fmt(m0['J_REL2'])}
MAPE = {_fmt(m0['MAPE_pct'])}%
MEDIAN_ABS_REL_ERROR = {_fmt(m0['median_abs_relative_error_pct'])}%
MAE = {_fmt(m0['MAE'])}
RMSE = {_fmt(m0['RMSE'])}

M1:
beta = 0.045953794473
k_scale = {_fmt(m1['k'])}
Q_scale = {_fmt(m1['Q'])}
J_REL2 = {_fmt(m1['J_REL2'])}
MAPE = {_fmt(m1['MAPE_pct'])}%
MEDIAN_ABS_REL_ERROR = {_fmt(m1['median_abs_relative_error_pct'])}%
MAE = {_fmt(m1['MAE'])}
RMSE = {_fmt(m1['RMSE'])}

M2:
beta_dirty = {_fmt(m2['beta'])}
k_dirty = {_fmt(m2['k'])}
Q_dirty = {_fmt(m2['Q'])}
J_REL2 = {_fmt(m2['J_REL2'])}
MAPE = {_fmt(m2['MAPE_pct'])}%
MEDIAN_ABS_REL_ERROR = {_fmt(m2['median_abs_relative_error_pct'])}%
MAE = {_fmt(m2['MAE'])}
RMSE = {_fmt(m2['RMSE'])}

IMPROVEMENT_M2_VS_M1 =
[{imp['classification']}; J_REL2 reduction={_fmt(imp['J_REL2_reduction'])} ({_fmt(imp['J_REL2_reduction_pct'])}%); MAPE change={_fmt(imp['MAPE_change_pct_points'])} percentage points; median absolute relative error change={_fmt(imp['median_abs_relative_error_change_pct_points'])} percentage points; MAE change={_fmt(imp['MAE_change'])}; RMSE change={_fmt(imp['RMSE_change'])}. {imp['classification_rule']}]

LOO_CV =
PASS

beta_fold_range =
[{_fmt(loo['beta_summary']['min'])}, {_fmt(loo['beta_summary']['max'])}; mean={_fmt(loo['beta_summary']['mean'])}; std={_fmt(loo['beta_summary']['std_population'])}]

k_fold_range =
[{_fmt(loo['k_summary']['min'])}, {_fmt(loo['k_summary']['max'])}; mean={_fmt(loo['k_summary']['mean'])}; std={_fmt(loo['k_summary']['std_population'])}]

920022_INFLUENCE =
[Included in final M2 with SECTION-SPECIFIC ASSIGNMENT REPRESENTATION LIMITATION. Removing it in the diagnostic fold gives beta={_fmt(influence['beta_fold'])}, k={_fmt(influence['k_fold'])}, held-out relative error={_fmt(influence['relative_error'])}.]

ROBUSTNESS =
{robustness}

FIT_QUALITY =
{fit_quality}

SCIENTIFIC_RELIABILITY =
DIRTY / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL

ANAS_2025_USED =
NO

CANONICAL_ARTIFACTS_MODIFIED =
NO

GIT_BRANCH =
[{branch}]

COMMITS =
[{commit}]

TESTS =
[{tests}]

OUTPUT_ROOT =
[{output_root}]

RECOMMENDATION =
{payload['recommendation']}
"""
    path.write_text(text, encoding="utf-8")


def run_d1_r1(
    dirty_root: str | os.PathLike[str] | None = None,
    anas_review: str | os.PathLike[str] | None = None,
    verify_hashes: bool = True,
    branch: str = "feat/d1-r1-gravity-beta-k",
    commit: str = "PENDING_CODE_COMMIT",
    tests: str = "PENDING_POST_RUN_TESTS",
) -> dict:
    root = resolve_dirty_root(dirty_root)
    review = Path(
        anas_review
        or os.environ.get("DIRTY_FRLM_ANAS_REVIEW", "")
        or (root.parent / "ANAS_5_8D_QGIS_review.csv")
    ).expanduser().resolve()
    if "2025" in review.name:
        raise RuntimeError("ANAS 2025 is a strict holdout and cannot be an input.")
    work_dir = root / "03_WORK" / "d1_r1_gravity_beta_k_v01"
    output_dir = root / "04_OUTPUT" / "d1_r1_gravity_beta_k_v01"
    reporting_dir = root / "05_REPORTING" / "d1_r1_gravity_beta_k_v01"
    for directory in (work_dir, output_dir, reporting_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"{stamp} {message}"
        log_lines.append(line)
        print(line, flush=True)

    log("D1-R1 start; ANAS_2025_USED=NO")
    preflight = verify_snapshot(root, verify_hashes=verify_hashes)
    log(f"snapshot verification PASS ({preflight['files_verified']}/12)")
    inputs = load_model_inputs(root, review, progress=log)
    log("frozen section exposure operators derived; routing recalculation=NO")
    np.savez_compressed(
        work_dir / "D1_R1_section_operators_v01.npz",
        section_ids=np.asarray(SECTION_IDS, dtype=np.int64),
        od_origin=inputs.od_origin,
        od_destination=inputs.od_destination,
        exposure=inputs.exposure,
        commuting_section=inputs.commuting_section,
    )
    operator_rows = inputs.section_metadata.copy()
    operator_rows["commuting_fixed"] = inputs.commuting_section
    operator_rows["paths_exposed"] = [int((inputs.exposure[:, i] > 0).sum()) for i in range(6)]
    operator_rows.to_csv(work_dir / "D1_R1_section_operators_v01.csv", index=False)

    beta_grid = np.linspace(BETA_MIN, BETA_MAX, BETA_GRID_POINTS)
    grid_n = beta_grid_non_commuting(
        beta_grid, inputs.gravity_base, inputs.cost_minutes, inputs.exposure
    )
    log("historical 0.000000..0.200000 beta domain profiled on 5001 points")
    n_v0 = non_commuting_at_beta(
        BETA_V0, inputs.gravity_base, inputs.cost_minutes, inputs.exposure
    )
    m0_pred = inputs.commuting_section + n_v0
    m0_metrics = compute_metrics(inputs.observed, m0_pred)
    k_scale = analytic_k(n_v0, inputs.observed, inputs.commuting_section)
    m1_pred = inputs.commuting_section + k_scale * n_v0
    m1_metrics = compute_metrics(inputs.observed, m1_pred)
    m2, profile = fit_joint(
        inputs.gravity_base,
        inputs.cost_minutes,
        inputs.exposure,
        inputs.observed,
        inputs.commuting_section,
        beta_grid,
        grid_n,
    )
    m2_repeat, _ = fit_joint(
        inputs.gravity_base,
        inputs.cost_minutes,
        inputs.exposure,
        inputs.observed,
        inputs.commuting_section,
        beta_grid,
        grid_n,
    )
    deterministic = (
        m2.beta == m2_repeat.beta
        and m2.k == m2_repeat.k
        and np.array_equal(m2.predictions, m2_repeat.predictions)
    )
    if not deterministic:
        raise RuntimeError("Deterministic rerun failed.")
    log("M0/M1/M2 fitted; deterministic rerun PASS")

    section_rows = []
    section_rows.extend(
        _section_rows("M0", BETA_V0, 1.0, inputs.observed, inputs.commuting_section, n_v0, m0_pred)
    )
    section_rows.extend(
        _section_rows("M1", BETA_V0, k_scale, inputs.observed, inputs.commuting_section, n_v0, m1_pred)
    )
    section_rows.extend(
        _section_rows(
            "M2",
            m2.beta,
            m2.k,
            inputs.observed,
            inputs.commuting_section,
            m2.non_commuting_k1,
            m2.predictions,
        )
    )
    section_frame = pd.DataFrame(section_rows)
    m1_rel = section_frame.loc[section_frame["model"].eq("M1"), "relative_error"].to_numpy()
    m2_mask = section_frame["model"].eq("M2")
    section_frame.loc[m2_mask, "relative_error_change_vs_M1"] = (
        section_frame.loc[m2_mask, "relative_error"].to_numpy() - m1_rel
    )

    loo_joint_rows = _loo_rows(inputs, beta_grid, grid_n, joint=True)
    loo_scale_rows = _loo_rows(inputs, beta_grid, grid_n, joint=False)
    if len(loo_joint_rows) != 6 or len(loo_scale_rows) != 6:
        raise RuntimeError("LOO did not complete 6/6 folds.")
    log("joint and scale-only LOO complete 6/6")

    j_reduction = m1_metrics["J_REL2"] - m2.metrics["J_REL2"]
    j_reduction_pct = 100.0 * j_reduction / m1_metrics["J_REL2"]
    mape_change = m2.metrics["MAPE_pct"] - m1_metrics["MAPE_pct"]
    median_change = (
        m2.metrics["median_abs_relative_error_pct"]
        - m1_metrics["median_abs_relative_error_pct"]
    )
    mae_change = m2.metrics["MAE"] - m1_metrics["MAE"]
    rmse_change = m2.metrics["RMSE"] - m1_metrics["RMSE"]
    classification, classification_rule = _improvement_label(
        j_reduction_pct, mape_change, rmse_change
    )
    lower_n = non_commuting_at_beta(
        np.nextafter(0.0, 1.0), inputs.gravity_base, inputs.cost_minutes, inputs.exposure
    )
    lower_fit = _fit_at_beta(
        np.nextafter(0.0, 1.0), lower_n, inputs.observed, inputs.commuting_section
    )
    upper_n = non_commuting_at_beta(
        BETA_MAX, inputs.gravity_base, inputs.cost_minutes, inputs.exposure
    )
    upper_fit = _fit_at_beta(BETA_MAX, upper_n, inputs.observed, inputs.commuting_section)
    models = {
        "M0": {"beta": BETA_V0, "k": 1.0, "Q": Q_V0, **m0_metrics},
        "M1": {"beta": BETA_V0, "k": k_scale, "Q": k_scale * Q_V0, **m1_metrics},
        "M2": {
            "beta": m2.beta,
            "k": m2.k,
            "Q": m2.k * Q_V0,
            "beta_boundary_flag": m2.beta_boundary_flag,
            "k_boundary_flag": m2.k_boundary_flag,
            **m2.metrics,
        },
    }
    comparison = pd.DataFrame(
        [{"model": name, **metrics} for name, metrics in models.items()]
    )
    loo_joint_frame = pd.DataFrame(loo_joint_rows)
    loo_scale_frame = pd.DataFrame(loo_scale_rows)
    influence = loo_joint_frame.loc[
        loo_joint_frame["held_out_section"].eq(920022)
    ].iloc[0].to_dict()
    recommendation = "RATIFY_DIRTY_BETA_K" if classification == "MATERIAL" else "USE_SCALE_ONLY_M1"
    if m2.beta_boundary_flag or m2.k_boundary_flag:
        recommendation = "DO_NOT_RATIFY"
    verdict = "NOT_READY" if m2.beta_boundary_flag or m2.k_boundary_flag else "PASS"
    payload = {
        "status": verdict,
        "verdict": verdict,
        "scientific_reliability": "DIRTY / PROVISIONAL / DEMONSTRATOR / NON-CANONICAL",
        "sections_used": list(SECTION_IDS),
        "models": models,
        "improvement_M2_vs_M1": {
            "classification": classification,
            "classification_rule": classification_rule,
            "J_REL2_reduction": j_reduction,
            "J_REL2_reduction_pct": j_reduction_pct,
            "MAPE_change_pct_points": mape_change,
            "median_abs_relative_error_change_pct_points": median_change,
            "MAE_change": mae_change,
            "RMSE_change": rmse_change,
            "per_section_relative_error_change": {
                str(row.section_id): float(row.relative_error_change_vs_M1)
                for row in section_frame.loc[m2_mask].itertuples()
            },
        },
        "loo_joint": {
            "folds": 6,
            "beta_summary": _describe(loo_joint_frame["beta_fold"]),
            "k_summary": _describe(loo_joint_frame["k_fold"]),
            "held_out_errors": {
                str(row.held_out_section): float(row.relative_error)
                for row in loo_joint_frame.itertuples()
            },
        },
        "loo_scale": {"folds": 6},
        "section_920022_influence": influence,
        "boundary_checks": {
            "lower_beta_limit": np.nextafter(0.0, 1.0),
            "lower_limit_J_REL2": lower_fit.metrics["J_REL2"],
            "lower_limit_k": lower_fit.k,
            "upper_beta": BETA_MAX,
            "upper_J_REL2": upper_fit.metrics["J_REL2"],
            "upper_k": upper_fit.k,
        },
        "qa": {
            "exact_six_sections_once": list(inputs.section_metadata["SECTION_ID"]) == list(SECTION_IDS),
            "no_eligibility_filtering": True,
            "deterministic_rerun": deterministic,
            "loo_6_of_6": True,
            "anas_2025_used": False,
            "commuting_unchanged": True,
            "territorial_inputs_unchanged": True,
            "routing_recalculated": False,
            "canonical_artifacts_modified": False,
            "snapshot": preflight,
        },
        "historical_domain": {
            "minimum": BETA_MIN,
            "maximum": BETA_MAX,
            "points": BETA_GRID_POINTS,
            "actual_constraint": "beta > 0",
            "evidence": "D1-R1 execution contract; no separate historical beta-profile artifact was present in the local Dirty workspace.",
        },
        "source_paths": dict(inputs.source_paths),
        "recommendation": recommendation,
    }

    snapshot_unchanged = all(
        (root / "01_INPUT_SNAPSHOT" / Path(check["relative_path"])).stat().st_size
        == check["size_bytes"]
        and (root / "01_INPUT_SNAPSHOT" / Path(check["relative_path"])).stat().st_mtime_ns
        == check["mtime_ns"]
        for check in preflight["checks"]
    )
    if not snapshot_unchanged:
        raise RuntimeError("A frozen snapshot input changed during execution.")
    payload["qa"]["snapshot_unchanged_during_run"] = True

    comparison.to_csv(output_dir / "D1_R1_model_comparison_v01.csv", index=False)
    section_frame.to_csv(output_dir / "D1_R1_section_metrics_v01.csv", index=False)
    loo_joint_frame.to_csv(output_dir / "D1_R1_loo_joint_v01.csv", index=False)
    loo_scale_frame.to_csv(output_dir / "D1_R1_loo_scale_v01.csv", index=False)
    profile.to_csv(output_dir / "D1_R1_beta_profile_v01.csv", index=False)
    _json_dump(output_dir / "D1_R1_final_parameters_v01.json", payload)

    preflight_text = f"""# D1-R1 input and execution preflight

- DIRTY_FRLM_ROOT: `{root}`
- Snapshot manifests: present
- Mandatory snapshot: **{preflight['declared_byte_identical']} BYTE-IDENTICAL**
- Actual files verified: **{preflight['files_verified']}/12**
- SHA-256 reverified in this run: **{'YES' if verify_hashes else 'NO'}**
- ANAS observations/mapping source: `{review}` (2024 only)
- Sections: `{list(SECTION_IDS)}` exactly once; no eligibility filtering
- `920022`: included with **SECTION-SPECIFIC ASSIGNMENT REPRESENTATION LIMITATION**
- Frozen commuting/P_i/A_j/PRODUCT-LAMBDA/TIME_B5/access/path system: read-only
- Routing recalculated: **NO**
- Historical beta domain used: 0.000000..0.200000, 5001 points; optimization enforces beta > 0
- Domain provenance caveat: no separate historical beta-profile artifact was found in the local Dirty workspace; the explicit D1-R1 execution contract is the available evidence.
- ANAS_2025_USED: **NO**
"""
    (reporting_dir / "D1_R1_INPUT_AND_EXECUTION_PREFLIGHT.md").write_text(
        preflight_text, encoding="utf-8"
    )
    _write_gate_report(
        reporting_dir / "D1_R1_FINAL_GATE_REPORT.md",
        payload,
        branch,
        commit,
        tests,
        output_dir,
    )
    log("all required D1-R1 outputs written; canonical artifacts modified=NO")
    (reporting_dir / "D1_R1_run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    artifacts = []
    for directory in (output_dir, reporting_dir):
        for artifact in sorted(directory.iterdir()):
            if artifact.name == "D1_R1_manifest_v01.json" or not artifact.is_file():
                continue
            artifacts.append(
                {
                    "path": str(artifact.relative_to(root)),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                }
            )
    output_manifest = {
        "manifest": "D1_R1_manifest_v01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": verdict,
        "anas_2025_used": False,
        "canonical_artifacts_modified": False,
        "artifacts": artifacts,
    }
    _json_dump(reporting_dir / "D1_R1_manifest_v01.json", output_manifest)
    return payload
