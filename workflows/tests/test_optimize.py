import unittest

import numpy as np

from workflows.optimize import cumulative_production, feasible_candidates


class OptimizeTests(unittest.TestCase):
    def test_feasible_candidates_respect_occupancy_and_spacing(self):
        centroids = [[x * 100.0, y * 100.0, 36.0] for x in range(5) for y in range(5)]
        geometry = {
            "centroids": centroids,
            "wells": {
                "I1": {"xyz": [0.0, 0.0, 36.0], "cell": 0},
                "P1": {"xyz": [400.0, 400.0, 36.0], "cell": 24},
                "P2": {"xyz": [200.0, 200.0, 36.0], "cell": 12},
            },
        }
        candidates = feasible_candidates(geometry, "I1", min_spacing_m=150.0)
        cells = {c["cell"] for c in candidates}
        self.assertNotIn(24, cells)
        self.assertNotIn(12, cells)
        self.assertNotIn(13, cells, "within 100 m of P2")
        self.assertIn(0, cells, "the well's own current cell stays a candidate")
        self.assertTrue(
            all(
                np.linalg.norm(np.array(c["xyz"][:2]) - [200.0, 200.0]) >= 150.0
                for c in candidates
            )
        )
        self.assertEqual(
            len(feasible_candidates(geometry, "I1", min_spacing_m=0.0)), 23
        )

    def test_cumulative_production_from_negative_producer_rates(self):
        observed = {
            "time": [10.0, 20.0, 30.0],
            "values": {
                "P1:oil_rate": [-100.0, -100.0, -100.0],
                "P2:oil_rate": [-50.0, -50.0, -50.0],
            },
        }
        total = cumulative_production(observed, ["P1", "P2"], "oil_rate")
        self.assertAlmostEqual(total, 150.0 * 20.0)
        single = cumulative_production(
            {"time": [10.0], "values": {"P1:oil_rate": [-100.0]}}, ["P1"], "oil_rate"
        )
        self.assertAlmostEqual(single, 1000.0)


if __name__ == "__main__":
    unittest.main()
