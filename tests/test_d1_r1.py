from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.d1_r1 import (  # noqa: E402
    BETA_MAX,
    Q_V0,
    SECTION_IDS,
    analytic_k,
    compute_metrics,
    fit_joint,
)


class D1R1Tests(unittest.TestCase):
    def test_exact_section_contract(self) -> None:
        self.assertEqual(SECTION_IDS, (920022, 920028, 920024, 920026, 920042, 920040))
        self.assertEqual(len(SECTION_IDS), 6)
        self.assertEqual(len(set(SECTION_IDS)), 6)

    def test_analytic_k_matches_direct_solution_and_constraint(self) -> None:
        n = np.array([2.0, 3.0, 7.0])
        y = np.array([10.0, 20.0, 30.0])
        c = np.array([1.0, 2.0, 3.0])
        a = n / y
        b = (y - c) / y
        expected = max(0.0, float((a @ b) / (a @ a)))
        self.assertAlmostEqual(analytic_k(n, y, c), expected, places=14)
        self.assertEqual(analytic_k(n, y, np.array([100.0, 200.0, 300.0])), 0.0)

    def test_metrics(self) -> None:
        metrics = compute_metrics(np.array([10.0, 20.0]), np.array([11.0, 18.0]))
        self.assertAlmostEqual(metrics["J_REL2"], 0.02)
        self.assertAlmostEqual(metrics["MAPE_pct"], 10.0)
        self.assertAlmostEqual(metrics["MAE"], 1.5)

    def test_joint_fit_constraints_determinism_and_input_immutability(self) -> None:
        gravity_base = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
        cost = np.array([5.0, 15.0, 35.0, 60.0], dtype=float)
        exposure = np.array(
            [
                [1.0, 0.0, 0.2],
                [0.3, 0.5, 0.0],
                [0.0, 0.8, 0.4],
                [0.1, 0.0, 1.0],
            ]
        )
        commuting = np.array([100.0, 150.0, 80.0])
        observed = np.array([900_000.0, 500_000.0, 350_000.0])
        originals = [array.copy() for array in (gravity_base, cost, exposure, commuting, observed)]
        first, profile = fit_joint(gravity_base, cost, exposure, observed, commuting)
        second, _ = fit_joint(gravity_base, cost, exposure, observed, commuting)
        self.assertGreater(first.beta, 0.0)
        self.assertLessEqual(first.beta, BETA_MAX)
        self.assertGreaterEqual(first.k, 0.0)
        self.assertEqual(first.beta, second.beta)
        self.assertEqual(first.k, second.k)
        np.testing.assert_array_equal(first.predictions, second.predictions)
        self.assertEqual(len(profile), 5001)
        self.assertFalse(bool(profile.iloc[0]["valid_beta"]))
        for before, after in zip(originals, (gravity_base, cost, exposure, commuting, observed)):
            np.testing.assert_array_equal(before, after)

    def test_m1_regression_fixture(self) -> None:
        # Frozen real-run M1 section operators guard the scale-only regression.
        commuting = np.array(
            [
                7792.609180536387,
                537.4016999999989,
                707.9645999999989,
                742.2485999999989,
                461.74707433901017,
                4478.183870302174,
            ]
        )
        non_commuting = np.array(
            [
                161937.64312971575,
                3935.8035107844576,
                18515.349103411947,
                18622.59210669397,
                5438.912310978278,
                37961.29456283229,
            ]
        )
        observed = np.array([23143, 6088, 21028, 22770, 23127, 14659], dtype=float)
        k = analytic_k(non_commuting, observed, commuting)
        self.assertAlmostEqual(k, 0.15458212886141692, places=14)
        predicted = commuting + k * non_commuting
        self.assertAlmostEqual(compute_metrics(observed, predicted)["J_REL2"], 3.207633238213867, places=13)

    def test_anas_2025_contract_literal_absent(self) -> None:
        source = (REPO_ROOT / "src" / "dirty_frlm" / "d1_r1.py").read_text(encoding="utf-8")
        self.assertIn("anas_2025_used", source.lower())
        self.assertNotIn("TGMA_LIGHT_2025\"]", source)

    def test_q_constant(self) -> None:
        self.assertAlmostEqual(Q_V0, 1552629.518131, places=6)


if __name__ == "__main__":
    unittest.main()
