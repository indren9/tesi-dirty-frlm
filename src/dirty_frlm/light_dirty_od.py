"""Materialize the ratified LIGHT_DIRTY_OD_v01 demonstrator demand.

The implementation consumes only the verified Dirty FRLM snapshot. It uses
the already stored PRODUCT-LAMBDA municipal impedance and never recalculates
routing, shortest paths, access sets, Gamma_OSM, or G_OSM_operativo.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


BETA_DIRTY = 0.045953794473
K_DIRTY = 0.15
M1_K_ANCHOR = 0.154582128861
Q_V0 = 1_552_629.518131
Q_DIRTY_EXPECTED = 232_894.42771965
EXPECTED_OD_ROWS = 46_010
EXPECTED_MUNICIPALITIES = 215
SCALE_LABEL = "DIRTY_GRAVITY_SCALE_V01"
DATASET_LABEL = "LIGHT_DIRTY_OD_v01"
VERSION = "v01"
NUMERIC_TOLERANCE = 1e-8


@dataclass(frozen=True)
class MaterializationInputs:
    territorial: pd.DataFrame
    commuting: pd.DataFrame
    municipal_summary: pd.DataFrame
    paths: Mapping[str, Path]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def resolve_dirty_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    raw = explicit or os.environ.get("DIRTY_FRLM_ROOT")
    if not raw:
        raise RuntimeError("DIRTY_FRLM_ROOT is required.")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DIRTY_FRLM_ROOT does not exist: {root}")
    return root


def snapshot_manifest_path(root: Path) -> Path:
    return root / "05_REPORTING" / "manifests" / "DIRTY_FRLM_INPUT_SNAPSHOT_v01.json"


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def optional_file_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": None, "mtime_ns": None}
    return file_state(path)


def verify_snapshot(root: Path) -> dict[str, object]:
    manifest_path = snapshot_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    checks: list[dict[str, object]] = []
    for row in rows:
        path = root / "01_INPUT_SNAPSHOT" / Path(row["snapshot_relative_path"])
        actual_hash = sha256_file(path)
        state = file_state(path)
        checks.append(
            {
                **state,
                "relative_path": row["snapshot_relative_path"],
                "expected_size_bytes": int(row["size_bytes"]),
                "expected_sha256": row["snapshot_sha256"],
                "sha256": actual_hash,
                "size_match": state["size_bytes"] == int(row["size_bytes"]),
                "hash_match": actual_hash == row["snapshot_sha256"],
            }
        )
    declared_ok = (
        manifest.get("files_expected") == 12
        and manifest.get("files_copied") == 12
        and manifest.get("byte_identical") == "12/12"
        and len(rows) == 12
    )
    if not declared_ok or not all(c["size_match"] and c["hash_match"] for c in checks):
        raise RuntimeError("Dirty FRLM snapshot verification failed.")
    canonical_states: list[dict[str, object]] = []
    for row in rows:
        source = Path(row["source_path"])
        canonical_states.append(
            {
                **optional_file_state(source),
                "expected_sha256": row["source_sha256"],
                "source_status": row.get("source_status", ""),
                "source_original_status": row.get("source_original_status", ""),
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files_verified": len(checks),
        "declared_byte_identical": manifest["byte_identical"],
        "status": "PASS",
        "snapshot_checks": checks,
        "canonical_states": canonical_states,
    }


def load_inputs(root: Path) -> MaterializationInputs:
    demand = root / "01_INPUT_SNAPSHOT" / "demand"
    paths = root / "01_INPUT_SNAPSHOT" / "paths"
    input_paths = {
        "territorial": demand / "Gravity_v0_territorial_inputs_derived_v01.xlsx",
        "commuting": demand / "ISTAT_commuting_LIGHT_v0.xlsx",
        "municipal_summary": paths / "OSM_OD_municipal_summary_v01.csv",
    }
    territorial = pd.read_excel(input_paths["territorial"], sheet_name="TERRITORIAL_INPUTS")
    commuting = pd.read_excel(input_paths["commuting"], sheet_name="COMMUTING_OD")
    municipal_summary = pd.read_csv(input_paths["municipal_summary"])
    return MaterializationInputs(
        territorial=territorial,
        commuting=commuting,
        municipal_summary=municipal_summary,
        paths=input_paths,
    )


def validate_inputs(inputs: MaterializationInputs) -> dict[str, object]:
    territorial = inputs.territorial
    commuting = inputs.commuting
    summary = inputs.municipal_summary
    territorial_required = {"PRO_COM", "COMUNE", "P_i", "A_j"}
    commuting_required = {"ORIGIN_PRO_COM", "DESTINATION_PRO_COM", "C_ij_ISTAT_VEH_DAY"}
    summary_required = {
        "origin_PRO_COM",
        "origin_COMUNE",
        "destination_PRO_COM",
        "destination_COMUNE",
        "time_s_PRODUCT_LAMBDA",
    }
    if not territorial_required.issubset(territorial.columns):
        raise RuntimeError("Territorial input schema is incompatible.")
    if not commuting_required.issubset(commuting.columns):
        raise RuntimeError("Commuting input schema is incompatible.")
    if not summary_required.issubset(summary.columns):
        raise RuntimeError("Municipal summary schema is incompatible.")
    if len(territorial) != EXPECTED_MUNICIPALITIES or territorial["PRO_COM"].nunique() != EXPECTED_MUNICIPALITIES:
        raise RuntimeError("Expected the frozen 215-municipality territorial set.")
    if len(commuting) != EXPECTED_OD_ROWS or len(summary) != EXPECTED_OD_ROWS:
        raise RuntimeError("Expected exactly 46,010 ordered inter-municipal OD rows.")
    summary_keys = summary[["origin_PRO_COM", "destination_PRO_COM"]].to_numpy(dtype=np.int64)
    commuting_keys = commuting[["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"]].to_numpy(dtype=np.int64)
    if not np.array_equal(summary_keys, commuting_keys):
        raise RuntimeError("Commuting and PRODUCT-LAMBDA OD ordering differ.")
    if len(np.unique(summary_keys, axis=0)) != EXPECTED_OD_ROWS:
        raise RuntimeError("OD keys are not unique.")
    if np.any(summary_keys[:, 0] == summary_keys[:, 1]):
        raise RuntimeError("Intrazonal pairs are present in the inter-municipal contract.")
    numeric_arrays = [
        territorial["P_i"].to_numpy(dtype=np.float64),
        territorial["A_j"].to_numpy(dtype=np.float64),
        commuting["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64),
        summary["time_s_PRODUCT_LAMBDA"].to_numpy(dtype=np.float64),
    ]
    if any(not np.isfinite(values).all() for values in numeric_arrays):
        raise RuntimeError("Non-finite input value detected.")
    if any((values < 0).any() for values in numeric_arrays):
        raise RuntimeError("Negative input value detected.")
    return {
        "municipalities": EXPECTED_MUNICIPALITIES,
        "od_rows": EXPECTED_OD_ROWS,
        "unique_od_keys": EXPECTED_OD_ROWS,
        "intrazonal_rows": 0,
        "od_order_matches_commuting": True,
        "status": "PASS",
    }


def materialize_dataframe(inputs: MaterializationInputs) -> pd.DataFrame:
    validate_inputs(inputs)
    territorial = inputs.territorial.set_index("PRO_COM")
    summary = inputs.municipal_summary
    commuting = inputs.commuting
    origins = summary["origin_PRO_COM"].to_numpy(dtype=np.int64)
    destinations = summary["destination_PRO_COM"].to_numpy(dtype=np.int64)
    gravity_base = (
        territorial.loc[origins, "P_i"].to_numpy(dtype=np.float64)
        * territorial.loc[destinations, "A_j"].to_numpy(dtype=np.float64)
    )
    cost_minutes = summary["time_s_PRODUCT_LAMBDA"].to_numpy(dtype=np.float64) / 60.0
    weights = gravity_base * np.exp(-BETA_DIRTY * cost_minutes)
    denominator = float(weights.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Gravity v0 normalization denominator is invalid.")
    n_v0 = Q_V0 * weights / denominator
    n_dirty = K_DIRTY * n_v0
    c_istat = commuting["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64).copy()
    t_dirty = c_istat + n_dirty
    return pd.DataFrame(
        {
            "ORIGIN_PRO_COM": origins,
            "ORIGIN_COMUNE": summary["origin_COMUNE"].astype(str).to_numpy(),
            "DESTINATION_PRO_COM": destinations,
            "DESTINATION_COMUNE": summary["destination_COMUNE"].astype(str).to_numpy(),
            "C_ISTAT_ij": c_istat,
            "N_v0_ij": n_v0,
            "N_dirty_ij": n_dirty,
            "T_dirty_ij": t_dirty,
        }
    )


def csv_payload(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", float_format="%.15g")
    return buffer.getvalue().encode("utf-8")


def check_materialization(
    inputs: MaterializationInputs,
    first: pd.DataFrame,
    second: pd.DataFrame,
    csv_first: bytes,
    csv_second: bytes,
) -> dict[str, object]:
    numeric = first[["C_ISTAT_ij", "N_v0_ij", "N_dirty_ij", "T_dirty_ij"]].to_numpy(
        dtype=np.float64
    )
    source_commuting = inputs.commuting["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64)
    n_v0_sum = float(first["N_v0_ij"].sum())
    n_dirty_sum = float(first["N_dirty_ij"].sum())
    c_sum = float(first["C_ISTAT_ij"].sum())
    t_sum = float(first["T_dirty_ij"].sum())
    scale_residual = n_dirty_sum - K_DIRTY * n_v0_sum
    q_residual = n_dirty_sum - Q_DIRTY_EXPECTED
    identity_max_abs = float(
        np.max(np.abs(first["T_dirty_ij"].to_numpy() - source_commuting - first["N_dirty_ij"].to_numpy()))
    )
    checks = {
        "cardinality_od": len(first) == EXPECTED_OD_ROWS,
        "unique_od_keys": not first[["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"]].duplicated().any(),
        "finite_values": bool(np.isfinite(numeric).all()),
        "no_negative_values": bool((numeric >= 0).all()),
        "commuting_logically_unchanged": bool(
            np.array_equal(first["C_ISTAT_ij"].to_numpy(dtype=np.float64), source_commuting)
        ),
        "n_dirty_scale_identity": abs(scale_residual) <= NUMERIC_TOLERANCE,
        "q_dirty_target": abs(q_residual) <= NUMERIC_TOLERANCE,
        "t_dirty_identity": identity_max_abs <= 1e-12,
        "deterministic_arrays": first.equals(second),
        "deterministic_csv_bytes": csv_first == csv_second,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Materialization QA failed: {failed}")
    return {
        "status": "PASS",
        "checks": checks,
        "actuals": {
            "od_rows": len(first),
            "unique_od_keys": int(
                first[["ORIGIN_PRO_COM", "DESTINATION_PRO_COM"]].drop_duplicates().shape[0]
            ),
            "negative_value_count": int((numeric < 0).sum()),
            "nonfinite_value_count": int((~np.isfinite(numeric)).sum()),
            "sum_C_ISTAT_ij": c_sum,
            "sum_N_v0_ij": n_v0_sum,
            "sum_N_dirty_ij": n_dirty_sum,
            "sum_T_dirty_ij": t_sum,
            "expected_k_times_sum_N_v0": K_DIRTY * n_v0_sum,
            "n_dirty_scale_residual": scale_residual,
            "expected_Q_dirty": Q_DIRTY_EXPECTED,
            "q_dirty_residual": q_residual,
            "t_dirty_identity_max_abs_error": identity_max_abs,
            "csv_sha256_first": sha256_bytes(csv_first),
            "csv_sha256_second": sha256_bytes(csv_second),
        },
        "tolerances": {
            "aggregate_absolute": NUMERIC_TOLERANCE,
            "row_identity_absolute": 1e-12,
        },
    }


def compare_states(before: list[dict[str, object]], after: list[dict[str, object]]) -> bool:
    before_keyed = {
        str(item["path"]): (item.get("exists", True), item["size_bytes"], item["mtime_ns"])
        for item in before
    }
    after_keyed = {
        str(item["path"]): (item.get("exists", True), item["size_bytes"], item["mtime_ns"])
        for item in after
    }
    return before_keyed == after_keyed


def git_info(repo_root: Path) -> dict[str, object]:
    base = ["git", "-c", f"safe.directory={repo_root.as_posix()}", "-C", str(repo_root)]
    branch = subprocess.check_output(base + ["branch", "--show-current"], text=True).strip()
    commit = subprocess.check_output(base + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(base + ["status", "--porcelain"], text=True)
    return {
        "repo": str(repo_root),
        "branch": branch,
        "commit": commit,
        "status_porcelain": status,
        "scope_clean": status == "",
    }


def run_materialization(
    root: Path,
    repo_root: Path,
    node_executable: Path,
    xlsx_builder: Path,
) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    output_dir = root / "04_OUTPUT" / "light_dirty_od_v01"
    reporting_dir = root / "05_REPORTING" / "light_dirty_od_v01"
    work_dir = root / "03_WORK" / "light_dirty_od_v01" / started.strftime("run_%Y%m%dT%H%M%SZ")
    for target in (output_dir, reporting_dir, work_dir):
        if target.exists():
            raise FileExistsError(f"Append-only target already exists: {target}")

    git_pre = git_info(repo_root)
    if not git_pre["scope_clean"]:
        raise RuntimeError("Git scope is not clean before materialization.")
    snapshot_pre = verify_snapshot(root)
    commuting_path = root / "01_INPUT_SNAPSHOT" / "demand" / "ISTAT_commuting_LIGHT_v0.xlsx"
    commuting_hash_pre = sha256_file(commuting_path)
    canonical_pre = snapshot_pre["canonical_states"]

    inputs = load_inputs(root)
    input_validation = validate_inputs(inputs)
    first = materialize_dataframe(inputs)
    second = materialize_dataframe(inputs)
    csv_first = csv_payload(first)
    csv_second = csv_payload(second)
    qa = check_materialization(inputs, first, second, csv_first, csv_second)

    work_dir.mkdir(parents=True, exist_ok=False)
    staged_csv = work_dir / f"{DATASET_LABEL}.csv"
    staged_xlsx = work_dir / f"{DATASET_LABEL}.xlsx"
    staged_csv.write_bytes(csv_first)

    metadata = {
        "dataset_label": DATASET_LABEL,
        "scale_label": SCALE_LABEL,
        "status": "ENGINEERING / DEMONSTRATOR / PROVISIONAL / NON-CANONICAL",
        "beta_dirty_demonstrator": BETA_DIRTY,
        "k_dirty_demonstrator": K_DIRTY,
        "m1_k_anchor": M1_K_ANCHOR,
        "q_v0_veh_day": Q_V0,
        "q_dirty_noncommuting_veh_day": Q_DIRTY_EXPECTED,
        "method": "N_dirty_ij = 0.15 * N_v0_ij; T_dirty_ij = C_ISTAT_ij + N_dirty_ij",
        "assumption": "k=0.15 is a rounded engineering scaling assumption anchored to M1 k*=0.154582128861; it is not scientifically calibrated.",
        "gravity_v0_reconstruction": "N_v0 uses frozen P_i, A_j and stored time_s_PRODUCT_LAMBDA at beta_dirty; PRODUCT-LAMBDA is read, not recalculated.",
        "historical_status": {
            "D1_original": "NOT_READY / PRESERVED HISTORICAL",
            "D1_R1": "NOT_READY / PRESERVED HISTORICAL",
        },
        "qa": qa,
        "provenance": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in inputs.paths.items()
        },
    }
    metadata_path = work_dir / "LIGHT_DIRTY_OD_v01_metadata_for_xlsx.json"
    metadata_path.write_bytes(json_bytes(metadata))
    preview_data = work_dir / "LIGHT_DIRTY_OD_v01_preview_data.png"
    preview_meta = work_dir / "LIGHT_DIRTY_OD_v01_preview_metadata.png"
    inspect_path = work_dir / "LIGHT_DIRTY_OD_v01_artifact_tool_inspect.json"
    subprocess.run(
        [
            str(node_executable),
            str(xlsx_builder),
            str(staged_csv),
            str(metadata_path),
            str(staged_xlsx),
            str(preview_data),
            str(preview_meta),
            str(inspect_path),
        ],
        cwd=repo_root,
        check=True,
    )

    reloaded = pd.read_csv(staged_csv)
    csv_reload_scale_residual = float(
        reloaded["N_dirty_ij"].sum() - K_DIRTY * reloaded["N_v0_ij"].sum()
    )
    csv_reload_q_residual = float(reloaded["N_dirty_ij"].sum() - Q_DIRTY_EXPECTED)
    csv_reload_identity_max_abs = float(
        np.max(
            np.abs(
                reloaded["T_dirty_ij"].to_numpy(dtype=np.float64)
                - reloaded["C_ISTAT_ij"].to_numpy(dtype=np.float64)
                - reloaded["N_dirty_ij"].to_numpy(dtype=np.float64)
            )
        )
    )
    if abs(csv_reload_scale_residual) > NUMERIC_TOLERANCE:
        raise RuntimeError("Serialized CSV fails N_dirty scale identity.")
    if abs(csv_reload_q_residual) > NUMERIC_TOLERANCE:
        raise RuntimeError("Serialized CSV fails Q_dirty target.")
    if csv_reload_identity_max_abs > NUMERIC_TOLERANCE:
        raise RuntimeError("Serialized CSV fails T=C+N identity.")
    qa["serialized_csv"] = {
        "status": "PASS",
        "scale_residual": csv_reload_scale_residual,
        "q_dirty_residual": csv_reload_q_residual,
        "t_dirty_identity_max_abs_error": csv_reload_identity_max_abs,
    }

    commuting_after = pd.read_excel(commuting_path, sheet_name="COMMUTING_OD")
    commuting_hash_post = sha256_file(commuting_path)
    commuting_logical_post = np.array_equal(
        commuting_after["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64),
        inputs.commuting["C_ij_ISTAT_VEH_DAY"].to_numpy(dtype=np.float64),
    )
    manifest_rows = json.loads(snapshot_manifest_path(root).read_text(encoding="utf-8"))["rows"]
    snapshot_after_states = [
        file_state(root / "01_INPUT_SNAPSHOT" / Path(row["snapshot_relative_path"]))
        for row in manifest_rows
    ]
    snapshot_before_states = [
        {"path": item["path"], "size_bytes": item["size_bytes"], "mtime_ns": item["mtime_ns"]}
        for item in snapshot_pre["snapshot_checks"]
    ]
    canonical_after = [optional_file_state(Path(item["path"])) for item in canonical_pre]
    canonical_present_count = sum(bool(item["exists"]) for item in canonical_pre)
    canonical_missing_count = len(canonical_pre) - canonical_present_count
    snapshot_unchanged = compare_states(snapshot_before_states, snapshot_after_states)
    canonical_unchanged = compare_states(canonical_pre, canonical_after)
    if commuting_hash_pre != commuting_hash_post or not commuting_logical_post:
        raise RuntimeError("ISTAT commuting input changed during the run.")
    if not snapshot_unchanged or not canonical_unchanged:
        raise RuntimeError("Frozen/canonical artifact state changed during the run.")

    output_dir.mkdir(parents=True, exist_ok=False)
    reporting_dir.mkdir(parents=True, exist_ok=False)
    final_csv = output_dir / staged_csv.name
    final_xlsx = output_dir / staged_xlsx.name
    shutil.copyfile(staged_csv, final_csv)
    shutil.copyfile(staged_xlsx, final_xlsx)

    parameters_path = output_dir / "DIRTY_GRAVITY_SCALE_V01_parameters.json"
    parameters = {
        "label": SCALE_LABEL,
        "dataset": DATASET_LABEL,
        "status": "ENGINEERING / DEMONSTRATOR / PROVISIONAL / NON-CANONICAL",
        "beta_dirty_demonstrator": BETA_DIRTY,
        "k_dirty_demonstrator": K_DIRTY,
        "m1_k_anchor": M1_K_ANCHOR,
        "q_v0_veh_day": Q_V0,
        "q_dirty_noncommuting_veh_day": qa["actuals"]["sum_N_dirty_ij"],
        "method": {
            "N_dirty_ij": "0.15 * N_v0_ij",
            "T_dirty_ij": "C_ISTAT_ij + N_dirty_ij",
        },
        "interpretation": "Rounded engineering scaling assumption anchored to M1 k*=0.154582128861; NOT scientifically calibrated.",
        "D1_original": "NOT_READY / PRESERVED HISTORICAL",
        "D1_R1": "NOT_READY / PRESERVED HISTORICAL",
    }
    parameters_path.write_bytes(json_bytes(parameters))

    qa.update(
        {
            "input_validation": input_validation,
            "commuting_byte_unchanged": commuting_hash_pre == commuting_hash_post,
            "commuting_logically_unchanged_after_run": commuting_logical_post,
            "commuting_sha256_before": commuting_hash_pre,
            "commuting_sha256_after": commuting_hash_post,
            "snapshot_unchanged_during_run": snapshot_unchanged,
            "canonical_artifacts_unchanged_during_run": canonical_unchanged,
            "canonical_source_paths_present": canonical_present_count,
            "canonical_source_paths_unavailable": canonical_missing_count,
            "routing_recalculated": False,
            "shortest_paths_recalculated": False,
            "product_lambda_recalculated": False,
            "time_b5_recalculated": False,
            "gamma_osm_recalculated": False,
            "g_osm_operativo_recalculated": False,
            "artifact_tool_inspect": str(inspect_path),
            "visual_previews": [str(preview_data), str(preview_meta)],
        }
    )
    qa_path = reporting_dir / "LIGHT_DIRTY_OD_v01_QA_evidence.json"
    qa_path.write_bytes(json_bytes(qa))

    git_post = git_info(repo_root)
    if not git_post["scope_clean"]:
        raise RuntimeError("Git scope is not clean after materialization.")

    log_path = reporting_dir / "LIGHT_DIRTY_OD_v01_run_log.txt"
    log_lines = [
        f"started_utc={started.isoformat()}",
        f"finished_utc={datetime.now(timezone.utc).isoformat()}",
        f"dirty_root={root}",
        f"repo={repo_root}",
        f"git_branch={git_post['branch']}",
        f"git_commit={git_post['commit']}",
        "status=PASS",
        f"od_rows={len(first)}",
        f"sum_N_v0_ij={qa['actuals']['sum_N_v0_ij']:.15f}",
        f"sum_N_dirty_ij={qa['actuals']['sum_N_dirty_ij']:.15f}",
        f"sum_T_dirty_ij={qa['actuals']['sum_T_dirty_ij']:.15f}",
        "stop_after_gate_report=true",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    payload_artifacts = [final_csv, final_xlsx, parameters_path, qa_path, log_path]
    payload_records = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_artifacts
    ]
    report_path = reporting_dir / "DIRTY_DEMAND_MATERIALIZATION_GATE_REPORT.md"
    report_lines = [
        "# DIRTY_DEMAND_MATERIALIZATION_GATE_REPORT",
        "",
        "## Gate verdict",
        "",
        "```text",
        "STATUS = PASS",
        f"DIRTY_GRAVITY_METHOD = PRESERVE_GRAVITY_V0_SPATIAL_STRUCTURE",
        f"SCALE_LABEL = {SCALE_LABEL}",
        f"DATASET_LABEL = {DATASET_LABEL}",
        "SCIENTIFIC_STATUS = ENGINEERING / DEMONSTRATOR / PROVISIONAL / NON-CANONICAL",
        "D1 ORIGINAL = NOT_READY / PRESERVED HISTORICAL",
        "D1-R1 = NOT_READY / PRESERVED HISTORICAL",
        "NEXT_STEP_STARTED = NO",
        "```",
        "",
        "## Ratified parameters and interpretation",
        "",
        f"- `beta_dirty_demonstrator = {BETA_DIRTY}`",
        f"- `k_dirty_demonstrator = {K_DIRTY}`",
        f"- `Q_dirty_noncommuting = {qa['actuals']['sum_N_dirty_ij']:.14f} veh/day`",
        f"- `N_dirty_ij = {K_DIRTY} * N_v0_ij`",
        "- `T_dirty_ij = C_ISTAT_ij + N_dirty_ij`",
        f"- `k=0.15` is a rounded engineering scaling assumption anchored to M1 `k*={M1_K_ANCHOR}`. It is **NOT scientifically calibrated**.",
        "",
        "`N_v0_ij` was materialized from the frozen Gravity v0 territorial inputs and the already stored `time_s_PRODUCT_LAMBDA` at the ratified historical beta. PRODUCT-LAMBDA was read unchanged; it was not recalculated.",
        "",
        "## Mandatory QA",
        "",
        "| Check | Result | Evidence |",
        "|---|---:|---|",
        f"| OD cardinality | PASS | {len(first):,} rows; {EXPECTED_OD_ROWS:,} expected; unique keys {qa['actuals']['unique_od_keys']:,} |",
        f"| No negative / non-finite demand | PASS | negatives {qa['actuals']['negative_value_count']}; non-finite {qa['actuals']['nonfinite_value_count']} |",
        f"| Commuting byte unchanged | PASS | SHA256 before/after `{commuting_hash_pre}` |",
        "| Commuting logically unchanged | PASS | output `C_ISTAT_ij` equals source vector exactly; source reload exact |",
        f"| `sum(N_dirty)=0.15*sum(N_v0)` | PASS | residual `{qa['actuals']['n_dirty_scale_residual']:.3e}`; tolerance `{NUMERIC_TOLERANCE:.1e}` |",
        f"| `Q_dirty=232894.42771965` | PASS | actual `{qa['actuals']['sum_N_dirty_ij']:.14f}`; residual `{qa['actuals']['q_dirty_residual']:.3e}` |",
        f"| `T_dirty=C+N_dirty` | PASS | max row error `{qa['actuals']['t_dirty_identity_max_abs_error']:.3e}` |",
        "| Determinism | PASS | two independent in-memory materializations equal; CSV bytes and SHA256 equal |",
        "| Provenance | PASS | exact source paths and SHA256 recorded in QA evidence and manifest |",
        "| Frozen snapshot unchanged | PASS | all 12 snapshot files hash-verified before run and size/mtime stable after run |",
        f"| Canonical artifacts unchanged | PASS | all 12 declared source-path states stable ({canonical_present_count} present; {canonical_missing_count} unavailable); only verified snapshot copies were read |",
        "| Forbidden recalculation | PASS | network, shortest paths, PRODUCT-LAMBDA, TIME_B5, Gamma_OSM and G_OSM_operativo not recalculated |",
        f"| Git scope clean | PASS | branch `{git_post['branch']}`; commit `{git_post['commit']}`; porcelain empty before and after |",
        "| Append-only publication | PASS | versioned output/report directories were absent and created once; no existing output overwritten |",
        "",
        "## Aggregate controls",
        "",
        f"- `sum(C_ISTAT_ij) = {qa['actuals']['sum_C_ISTAT_ij']:.14f} veh/day`",
        f"- `sum(N_v0_ij) = {qa['actuals']['sum_N_v0_ij']:.14f} veh/day`",
        f"- `sum(N_dirty_ij) = {qa['actuals']['sum_N_dirty_ij']:.14f} veh/day`",
        f"- `sum(T_dirty_ij) = {qa['actuals']['sum_T_dirty_ij']:.14f} veh/day`",
        "",
        "## Output artifacts and SHA256",
        "",
        "| Artifact | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    report_lines.extend(
        f"| `{record['path']}` | {record['size_bytes']} | `{record['sha256']}` |"
        for record in payload_records
    )
    report_lines.extend(
        [
            "",
            "## Provenance and stop condition",
            "",
            f"- Snapshot manifest: `{snapshot_pre['manifest_path']}` (`{snapshot_pre['manifest_sha256']}`)",
            f"- Repo: `{repo_root}`",
            f"- Branch: `{git_post['branch']}`",
            f"- Commit: `{git_post['commit']}`",
            "- Canonical artifacts modified: `false`",
            "- ANAS 2025 used: `false`",
            "- Next step started: `false`",
            "- Operational stop: **STOP after this gate report.**",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    manifest_path = reporting_dir / "LIGHT_DIRTY_OD_v01_manifest.json"
    all_payloads = payload_artifacts + [report_path]
    manifest = {
        "manifest": "LIGHT_DIRTY_OD_v01_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scale_label": SCALE_LABEL,
        "dataset_label": DATASET_LABEL,
        "scientific_status": "ENGINEERING / DEMONSTRATOR / PROVISIONAL / NON-CANONICAL",
        "git": git_post,
        "inputs": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in inputs.paths.items()
        },
        "snapshot_manifest": {
            "path": snapshot_pre["manifest_path"],
            "sha256": snapshot_pre["manifest_sha256"],
            "files_verified": snapshot_pre["files_verified"],
        },
        "qa": qa,
        "canonical_artifacts_modified": False,
        "forbidden_recalculations_performed": False,
        "anas_2025_used": False,
        "next_step_started": False,
        "artifacts": [
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in all_payloads
        ],
    }
    manifest_path.write_bytes(json_bytes(manifest))

    return {
        "status": "PASS",
        "report": str(report_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_csv": str(final_csv),
        "dataset_xlsx": str(final_xlsx),
        "qa_evidence": str(qa_path),
        "work_dir": str(work_dir),
        "git": git_post,
        "aggregates": qa["actuals"],
    }
