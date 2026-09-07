from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.light_dirty_od import (  # noqa: E402
    BETA_DIRTY,
    DATASET_LABEL,
    EXPECTED_OD_ROWS,
    K_DIRTY,
    M1_K_ANCHOR,
    Q_DIRTY_EXPECTED,
    Q_V0,
    SCALE_LABEL,
    MaterializationInputs,
    check_materialization,
    csv_payload,
    materialize_dataframe,
)


class LightDirtyODTests(unittest.TestCase):
    def fixture(self) -> MaterializationInputs:
        territorial = pd.DataFrame(
            {
                "PRO_COM": [1, 2, 3],
                "COMUNE": ["A", "B", "C"],
                "P_i": [0.2, 0.3, 0.5],
                "A_j": [0.4, 0.35, 0.25],
            }
        )
        keys = [(o, d) for o in (1, 2, 3) for d in (1, 2, 3) if o != d]
        summary = pd.DataFrame(
            {
                "origin_PRO_COM": [o for o, _ in keys],
                "origin_COMUNE": [{1: "A", 2: "B", 3: "C"}[o] for o, _ in keys],
                "destination_PRO_COM": [d for _, d in keys],
                "destination_COMUNE": [{1: "A", 2: "B", 3: "C"}[d] for _, d in keys],
                "time_s_PRODUCT_LAMBDA": [600, 900, 1200, 1500, 1800, 2100],
            }
        )
        commuting = pd.DataFrame(
            {
                "ORIGIN_PRO_COM": [o for o, _ in keys],
                "DESTINATION_PRO_COM": [d for _, d in keys],
                "C_ij_ISTAT_VEH_DAY": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        return MaterializationInputs(territorial, commuting, summary, {})

    def test_ratified_constants(self) -> None:
        self.assertEqual(BETA_DIRTY, 0.045953794473)
        self.assertEqual(K_DIRTY, 0.15)
        self.assertEqual(M1_K_ANCHOR, 0.154582128861)
        self.assertAlmostEqual(Q_V0 * K_DIRTY, Q_DIRTY_EXPECTED, places=10)
        self.assertEqual(SCALE_LABEL, "DIRTY_GRAVITY_SCALE_V01")
        self.assertEqual(DATASET_LABEL, "LIGHT_DIRTY_OD_v01")
        self.assertEqual(EXPECTED_OD_ROWS, 46_010)

    def test_materialization_formula_and_immutability(self) -> None:
        inputs = self.fixture()
        territorial_before = inputs.territorial.copy(deep=True)
        commuting_before = inputs.commuting.copy(deep=True)
        summary_before = inputs.municipal_summary.copy(deep=True)
        # The production cardinality guard is deliberately bypassed for this tiny unit fixture.
        import dirty_frlm.light_dirty_od as module

        original_rows = module.EXPECTED_OD_ROWS
        original_municipalities = module.EXPECTED_MUNICIPALITIES
        module.EXPECTED_OD_ROWS = 6
        module.EXPECTED_MUNICIPALITIES = 3
        try:
            first = materialize_dataframe(inputs)
            second = materialize_dataframe(inputs)
        finally:
            module.EXPECTED_OD_ROWS = original_rows
            module.EXPECTED_MUNICIPALITIES = original_municipalities
        self.assertTrue(first.equals(second))
        self.assertAlmostEqual(float(first["N_v0_ij"].sum()), Q_V0, places=8)
        self.assertAlmostEqual(float(first["N_dirty_ij"].sum()), Q_DIRTY_EXPECTED, places=8)
        np.testing.assert_array_equal(
            first["N_dirty_ij"].to_numpy(), K_DIRTY * first["N_v0_ij"].to_numpy()
        )
        np.testing.assert_array_equal(
            first["T_dirty_ij"].to_numpy(),
            first["C_ISTAT_ij"].to_numpy() + first["N_dirty_ij"].to_numpy(),
        )
        pd.testing.assert_frame_equal(inputs.territorial, territorial_before)
        pd.testing.assert_frame_equal(inputs.commuting, commuting_before)
        pd.testing.assert_frame_equal(inputs.municipal_summary, summary_before)
        self.assertEqual(csv_payload(first), csv_payload(second))

    def test_qa_rejects_negative(self) -> None:
        inputs = self.fixture()
        frame = pd.DataFrame(
            {
                "ORIGIN_PRO_COM": [1],
                "ORIGIN_COMUNE": ["A"],
                "DESTINATION_PRO_COM": [2],
                "DESTINATION_COMUNE": ["B"],
                "C_ISTAT_ij": [-1.0],
                "N_v0_ij": [Q_V0],
                "N_dirty_ij": [Q_DIRTY_EXPECTED],
                "T_dirty_ij": [Q_DIRTY_EXPECTED - 1.0],
            }
        )
        with self.assertRaises(RuntimeError):
            check_materialization(inputs, frame, frame.copy(), csv_payload(frame), csv_payload(frame))


if __name__ == "__main__":
    unittest.main()
