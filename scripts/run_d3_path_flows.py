from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.path_flows import (  # noqa: E402
    EXPECTED_PATH_COUNT,
    EXPECTED_TOTAL_LIGHT_DIRTY,
    input_paths,
    load_access_paths,
    load_light_dirty,
    resolve_dirty_root,
    run_d3_production,
    build_path_flow_dataframe,
    validate_sequence_arrays,
    verify_frozen_inputs,
)


EXPECTED_TRANSITION_SLOTS = 598_707_601


def validate_sequence_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    checks = {
        "version_v01": payload.get("version") == "v01",
        "cost_contract_TIME_B5": payload.get("cost_contract") == "TIME_B5",
        "distance_PATH_ATTRIBUTE": payload.get("distance_semantics") == "PATH_ATTRIBUTE",
        "weight_PRODUCT_LAMBDA": (
            payload.get("weight_contract") == "PRODUCT_LAMBDA / EXP_REL_300"
        ),
        "path_count": payload.get("counts", {}).get("paths") == EXPECTED_PATH_COUNT,
        "transition_slot_count": (
            payload.get("counts", {}).get("transition_slots")
            == EXPECTED_TRANSITION_SLOTS
        ),
    }

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"Frozen sequence contract validation failed: {failed}"
        )

    return {
        "status": "PASS",
        "checks": checks,
        "representation": payload.get("representation"),
        "path_order": payload.get("path_order"),
    }


def validate_paths_only(
    root: Path,
    *,
    verify_sha: bool,
) -> dict[str, object]:
    paths = input_paths(root)

    if verify_sha:
        sha_qa = verify_frozen_inputs(root)
    else:
        sha_qa = {
            "status": "PREVALIDATED_EXTERNALLY",
            "files_checked_this_run": 0,
            "note": (
                "Full 7-file SHA gate was completed immediately before this "
                "development dry validation."
            ),
        }

    contract_qa = validate_sequence_contract(
        paths["sequence_contract"]
    )

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

    result = {
        "mode": "VALIDATE_PATHS_ONLY",
        "status": "PASS",
        "sha_verification": sha_qa,
        "sequence_contract": contract_qa,
        "path_flow_qa": path_qa,
        "sequence_array_qa": sequence_qa,
        "expected_total_light_dirty_veh_day": EXPECTED_TOTAL_LIGHT_DIRTY,
        "edge_aggregation_executed": False,
        "output_materialized": False,
        "routing_recomputed": False,
        "product_lambda_recomputed": False,
        "time_b5_recomputed": False,
        "canonical_artifacts_modified": False,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D3 — materialize Dirty FRLM path and edge flows."
    )

    parser.add_argument(
        "--dirty-root",
        help="Dirty_FRLM operational root.",
    )

    parser.add_argument(
        "--validate-paths-only",
        action="store_true",
        help=(
            "Read-only development validation of the 414,090 access-pair "
            "path flows. Does not perform edge aggregation or publication."
        ),
    )

    parser.add_argument(
        "--skip-sha-for-prevalidated-dry-run",
        action="store_true",
        help=(
            "Allowed only with --validate-paths-only after an externally "
            "completed frozen-input SHA gate."
        ),
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Execute the full D3 production run.",
    )

    parser.add_argument(
        "--chunk-transitions",
        type=int,
        default=2_000_000,
        help="Target number of transition slots per aggregation chunk.",
    )

    args = parser.parse_args()

    if args.validate_paths_only == args.production:
        raise RuntimeError(
            "Choose exactly one mode: "
            "--validate-paths-only or --production."
        )

    root = resolve_dirty_root(args.dirty_root)

    if args.validate_paths_only:
        result = validate_paths_only(
            root,
            verify_sha=not args.skip_sha_for_prevalidated_dry_run,
        )
    else:
        if args.skip_sha_for_prevalidated_dry_run:
            raise RuntimeError(
                "Production mode may not skip SHA verification."
            )

        result = run_d3_production(
            root,
            REPO_ROOT,
            target_transitions_per_chunk=args.chunk_transitions,
        )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()