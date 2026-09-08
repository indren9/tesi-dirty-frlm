from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dirty_frlm.path_flows import (  # noqa: E402
    aggregate_edge_flows_arrays,
    build_path_flow_dataframe,
    validate_sequence_arrays,
    write_edge_flow_csv,
)


class PathFlowTests(unittest.TestCase):
    def fixture(self):
        light = pd.DataFrame(
            {
                "ORIGIN_PRO_COM": [1, 2],
                "ORIGIN_COMUNE": ["A", "B"],
                "DESTINATION_PRO_COM": [2, 1],
                "DESTINATION_COMUNE": ["B", "A"],
                "T_dirty_ij": [90.0, 45.0],
            }
        )

        rows = []
        path_idx = 0
        sequence_start = 0

        for origin, destination, origin_name, destination_name in (
            (1, 2, "A", "B"),
            (2, 1, "B", "A"),
        ):
            for a in range(1, 4):
                for b in range(1, 4):
                    sequence_end = sequence_start + 2

                    rows.append(
                        {
                            "path_idx": path_idx,
                            "path_id": f"P{path_idx}",
                            "origin_PRO_COM": origin,
                            "origin_COMUNE": origin_name,
                            "destination_PRO_COM": destination,
                            "destination_COMUNE": destination_name,
                            "origin_access_index": a - 1,
                            "destination_access_index": b - 1,
                            "origin_access_order": a,
                            "destination_access_order": b,
                            "pair_weight": 1.0 / 9.0,
                            "n_B5_transitions": 2,
                            "status": "FINITE_RECONSTRUCTED",
                            "sequence_start": sequence_start,
                            "sequence_end": sequence_end,
                            "canonical_route_impedance": "TIME_B5",
                            "distance_semantics": "PATH_ATTRIBUTE",
                        }
                    )

                    path_idx += 1
                    sequence_start = sequence_end

        paths = pd.DataFrame(rows)

        return light, paths


    def test_path_flow_mass_conservation(self):
        light, paths = self.fixture()

        output, qa = build_path_flow_dataframe(
            light,
            paths,
            expected_od_count=2,
            expected_path_count=18,
            expected_paths_per_od=9,
            expected_total_light_dirty=135.0,
        )

        self.assertEqual(len(output), 18)
        self.assertEqual(qa["od_count"], 2)
        self.assertEqual(qa["paths_per_od"], 9)
        self.assertAlmostEqual(
            qa["sum_access_path_flows_veh_day"],
            135.0,
            places=12,
        )
        self.assertLessEqual(
            abs(qa["global_mass_residual"]),
            1e-12,
        )
        self.assertLessEqual(
            qa["max_od_mass_residual"],
            1e-12,
        )
        self.assertLessEqual(
            qa["max_lambda_sum_residual"],
            1e-12,
        )


    def test_path_flow_rejects_bad_lambda_mass(self):
        light, paths = self.fixture()
        paths.loc[0, "pair_weight"] = 0.25

        with self.assertRaises(RuntimeError):
            build_path_flow_dataframe(
                light,
                paths,
                expected_od_count=2,
                expected_path_count=18,
                expected_paths_per_od=9,
                expected_total_light_dirty=135.0,
            )


    def test_sequence_contract(self):
        light, paths = self.fixture()

        output, _ = build_path_flow_dataframe(
            light,
            paths,
            expected_od_count=2,
            expected_path_count=18,
            expected_paths_per_od=9,
            expected_total_light_dirty=135.0,
        )

        offsets = np.arange(
            0,
            2 * len(output) + 1,
            2,
            dtype=np.int64,
        )

        qa = validate_sequence_arrays(
            output,
            offsets,
            transition_slot_count=int(offsets[-1]),
        )

        self.assertEqual(qa["status"], "PASS")
        self.assertEqual(qa["path_count"], 18)


    def test_edge_aggregation_rejects_repeated_edge_with_indicator_guard(self):
        path_flows = np.array(
            [10.0],
            dtype=np.float64,
        )

        offsets = np.array(
            [0, 2],
            dtype=np.int64,
        )

        transition_slots = np.array(
            [0, 1],
            dtype=np.int32,
        )

        edgeid_data = np.array(
            [1, 1],
            dtype=np.int64,
        )

        with self.assertRaises(RuntimeError):
            aggregate_edge_flows_arrays(
                path_flows,
                offsets,
                transition_slots,
                edgeid_data,
                max_edge_id=1,
                target_transitions_per_chunk=2,
                validate_indicator_semantics=True,
            )


    def test_edge_aggregation(self):
        path_flows = np.array(
            [10.0, 20.0],
            dtype=np.float64,
        )

        offsets = np.array(
            [0, 2, 3],
            dtype=np.int64,
        )

        transition_slots = np.array(
            [0, 1, 2],
            dtype=np.int32,
        )

        edgeid_data = np.array(
            [1, 2, 1],
            dtype=np.int64,
        )

        edge_flow, occurrences, qa = aggregate_edge_flows_arrays(
            path_flows,
            offsets,
            transition_slots,
            edgeid_data,
            max_edge_id=2,
            target_transitions_per_chunk=2,
        )

        self.assertAlmostEqual(edge_flow[1], 30.0)
        self.assertAlmostEqual(edge_flow[2], 10.0)

        self.assertEqual(occurrences[1], 2)
        self.assertEqual(occurrences[2], 1)

        self.assertEqual(
            qa["transition_slots_processed"],
            3,
        )
        self.assertEqual(
            qa["transition_occurrences"],
            3,
        )



    def test_sparse_edge_writer(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)

            db = temp / "edges.sqlite"
            output = temp / "edge_flows.csv"

            con = sqlite3.connect(db)

            try:
                con.execute(
                    """
                    CREATE TABLE directed_edges (
                        edge_id INTEGER PRIMARY KEY,
                        edge_uid TEXT NOT NULL,
                        segment_uid TEXT NOT NULL,
                        way_id INTEGER NOT NULL,
                        seq INTEGER NOT NULL,
                        u INTEGER NOT NULL,
                        v INTEGER NOT NULL,
                        way_direction TEXT NOT NULL,
                        length_m REAL NOT NULL,
                        highway TEXT,
                        edge_role TEXT NOT NULL
                    )
                    """
                )

                con.executemany(
                    "INSERT INTO directed_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            1, "E1", "S1", 10, 0,
                            1, 2, "FWD", 10.0,
                            "primary", "CORE"
                        ),
                        (
                            2, "E2", "S2", 10, 1,
                            2, 3, "FWD", 20.0,
                            "primary", "CORE"
                        ),
                        (
                            3, "E3", "S3", 10, 2,
                            3, 4, "FWD", 30.0,
                            "primary", "CORE"
                        ),
                    ],
                )

                con.commit()

            finally:
                con.close()

            edge_flow = np.array(
                [0.0, 5.0, 0.0, 7.0],
                dtype=np.float64,
            )

            occurrences = np.array(
                [0, 2, 0, 3],
                dtype=np.int64,
            )

            qa = write_edge_flow_csv(
                db,
                edge_flow,
                occurrences,
                output,
                sql_chunk_rows=2,
            )

            frame = pd.read_csv(output)

            self.assertEqual(
                frame["edge_id"].tolist(),
                [1, 3],
            )

            self.assertAlmostEqual(
                float(frame["dirty_flow_veh_day"].sum()),
                12.0,
            )

            self.assertEqual(
                qa["rows_written"],
                2,
            )

            self.assertEqual(
                qa["referenced_edge_ids_missing_from_sqlite"],
                0,
            )


if __name__ == "__main__":
    unittest.main()