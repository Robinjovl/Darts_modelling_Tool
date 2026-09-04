import unittest

from workflows.cost import (
    esmda_runs,
    estimate,
    fd_gradient_runs,
    morris_runs,
    planned_simulation_range,
    planned_simulations,
    robust_runs,
    saltelli_runs,
    usable_workers,
)
from workflows.spec import ModelRef, ParameterSpec, StudySpec


def spec(workflow: str, design: dict, n_params: int = 8) -> StudySpec:
    return StudySpec(
        name="c",
        workflow=workflow,
        model=ModelRef(model_dir="m", adapter="a"),
        parameters=[
            ParameterSpec(name=f"p{i}", family="ScalarParam") for i in range(n_params)
        ],
        seed_root=0,
        design=design,
    )


class FormulaTests(unittest.TestCase):
    def test_documented_formulas(self):
        self.assertEqual(saltelli_runs(512, 8), 5120)
        self.assertEqual(saltelli_runs(512, 8, second_order=True), 9216)
        with self.assertRaises(ValueError):
            saltelli_runs(500, 8)
        self.assertEqual(morris_runs(10, 14), 150)
        self.assertEqual(fd_gradient_runs(40), 80)
        self.assertEqual(fd_gradient_runs(40, central=False), 40)
        self.assertEqual(esmda_runs(100, 4), 500)
        self.assertEqual(robust_runs(431, 20), 8620)

    def test_planned_simulations_by_workflow(self):
        self.assertEqual(
            planned_simulations(spec("ensemble", {"method": "lhs", "n": 100})), 100
        )
        morris = spec("ensemble", {"method": "morris", "n_trajectories": 10}, 14)
        self.assertEqual(planned_simulations(morris), 150)
        self.assertEqual(
            planned_simulations(
                spec("ensemble", {"method": "saltelli", "n_base": 512})
            ),
            5120,
        )
        self.assertEqual(
            planned_simulations(spec("hm-esmda", {"ne": 100, "n_steps": 4})), 500
        )
        self.assertEqual(
            planned_simulations(spec("hm-adjoint", {"max_iterations": 30})), 60
        )
        exhaustive = spec("optimize", {"driver": "exhaustive", "n_candidates": 431})
        self.assertEqual(planned_simulations(exhaustive), 432)
        robust = spec("optimize", {"driver": "robust", "n_candidates": 431, "ne": 20})
        self.assertEqual(planned_simulations(robust), 8640)  # + baseline per member
        self.assertEqual(planned_simulation_range(robust), (8640, 8640))
        fd = spec(
            "optimize",
            {"driver": "fd_controls", "n_controls": 40, "max_iterations": 20},
        )
        self.assertEqual(planned_simulation_range(fd), (1800, 2500))
        self.assertEqual(planned_simulations(fd), 2500)
        with self.assertRaises(ValueError):
            planned_simulations(spec("ensemble", {"method": "magic"}))

    def test_estimate_ranges_validation_and_workers(self):
        est = estimate(
            simulations=1000,
            member_seconds=(8.0, 12.0),
            workers=50,
            startup_seconds=(0.5, 1.0),
        )
        self.assertEqual(est.simulations, 1000)
        self.assertLess(est.wall_s[0], est.wall_s[1])
        self.assertLess(est.cpu_hours[0], est.cpu_hours[1])
        self.assertGreater(est.disk_gb[1], est.disk_gb[0])
        self.assertIn("simulations", est.to_dict())
        with self.assertRaises(ValueError):
            estimate(simulations=-1, member_seconds=(1.0, 2.0), workers=1)
        with self.assertRaises(ValueError):
            estimate(simulations=1, member_seconds=(1.0, 2.0), workers=0)
        with self.assertRaises(ValueError):
            estimate(
                simulations=1, member_seconds=(1.0, 2.0), workers=1, failure_rate=1.5
            )
        self.assertGreaterEqual(usable_workers(), 1)
        self.assertEqual(usable_workers(max_workers=3), min(3, usable_workers()))
        self.assertGreaterEqual(usable_workers(memory_per_member_gb=1e6), 1)


class ExhaustiveEstimateTests(unittest.TestCase):
    def _spec(self, design):
        from workflows.spec import ModelRef, ParameterSpec, StudySpec

        return StudySpec(
            name="x",
            workflow="optimize",
            model=ModelRef(model_dir=".", adapter="a"),
            parameters=[
                ParameterSpec(
                    name="I1", family="ScalarParam", args={"target": "well_xyz"}
                )
            ],
            seed_root=1,
            design=design,
        )

    def test_max_candidates_counts_baseline(self):
        from workflows.cost import planned_simulations

        spec = self._spec({"driver": "exhaustive", "max_candidates": 4})
        self.assertEqual(planned_simulations(spec), 5)

    def test_missing_candidate_count_is_an_error(self):
        from workflows.cost import planned_simulations

        with self.assertRaises(ValueError):
            planned_simulations(self._spec({"driver": "exhaustive"}))


if __name__ == "__main__":
    unittest.main()
