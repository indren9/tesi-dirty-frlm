from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.light_dirty_od import resolve_dirty_root, run_materialization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize LIGHT_DIRTY_OD_v01.")
    parser.add_argument("--dirty-root")
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    result = run_materialization(
        resolve_dirty_root(args.dirty_root),
        REPO_ROOT,
        Path(args.node).resolve(),
        REPO_ROOT / "scripts" / "build_light_dirty_od_xlsx.mjs",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
