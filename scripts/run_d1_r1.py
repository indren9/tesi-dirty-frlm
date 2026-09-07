#!/usr/bin/env python
"""Execute the D1-R1 Dirty Gravity beta+k fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.d1_r1 import run_d1_r1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirty-root", help="Overrides DIRTY_FRLM_ROOT")
    parser.add_argument("--anas-review", help="2024 ANAS section review CSV")
    parser.add_argument(
        "--skip-full-snapshot-hash",
        action="store_true",
        help="Validate declared snapshot and sizes without rehashing all 3.25 GiB (not for final QA).",
    )
    parser.add_argument("--branch", default="feat/d1-r1-gravity-beta-k")
    parser.add_argument("--commit", default="PENDING_CODE_COMMIT")
    parser.add_argument("--tests", default="PENDING_POST_RUN_TESTS")
    args = parser.parse_args()
    result = run_d1_r1(
        args.dirty_root,
        args.anas_review,
        verify_hashes=not args.skip_full_snapshot_hash,
        branch=args.branch,
        commit=args.commit,
        tests=args.tests,
    )
    m2 = result["models"]["M2"]
    print(
        f"D1-R1 {result['verdict']} beta={m2['beta']:.12g} k={m2['k']:.12g} "
        f"J_REL2={m2['J_REL2']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
