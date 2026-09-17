import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows import cli
from workflows.ensemble import (
    lhs_design,
    make_design,
    morris_design,
    morris_indices,
    saltelli_design,
    sobol_indices,
)
from workflows.members import (
    LogPermField,
    LogScalarParam,
    Parameterization,
)
from workflows.spec import (
    ComputeSpec,
    ModelRef,
    ObservationSpec,
    ParameterSpec,
    StudySpec,
)


def ishigami(x, a=7.0, b=0.1):
    x = -np.pi + 2 * np.pi * np.asarray(x)
    return (
        np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])
    )


class MembersTests(unittest.TestCase):
    def test_families_map_from_unit_cube(self):
        param = Parameterization.from_specs(
            [
                ParameterSpec(
                    name="tm", family="LogScalarParam", args={"low": 0.1, "high": 10.0}
                ),
                ParameterSpec(
                    name="wi",
                    family="MultiplierField",
                    args={"n": 3, "sigma_log10": 0.2, "target": "wi_multiplier"},
                ),
            ]
        )
        self.assertEqual(param.dim, 4)
        self.assertEqual(param.labels(), ["tm", "wi[0]", "wi[1]", "wi[2]"])
        realization = param.to_realization([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(realization["tran_multiplier"], 1.0)
        np.testing.assert_allclose(realization["wi_multiplier"], 1.0)
        with self.assertRaises(ValueError):
            param.to_realization([0.5, 0.5])
        with self.assertRaises(KeyError):
            Parameterization.from_specs([ParameterSpec(name="x", family="Unknown")])

    def test_logperm_field_needs_geometry_and_keeps_energy(self):
        with self.assertRaises(ValueError):
            Parameterization.from_specs(
                [ParameterSpec(name="k", family="LogPermField")]
            )
        grid = np.array([[i * 100.0, j * 100.0] for i in range(10) for j in range(10)])
        field = LogPermField(
            name="k",
            target="permx",
            mean_log10=np.log10(500),
            sigma_log10=0.3,
            range_m=300.0,
            n_components=8,
        )
        field.set_centroids(grid)
        self.assertGreater(field.energy_fraction, 0.5)
        k = field.from_unit(np.full(8, 0.5))
        np.testing.assert_allclose(k, 500.0)
        k2 = field.from_unit(np.full(8, 0.9))
        self.assertEqual(k2.shape, (100,))
        self.assertTrue((k2 > 0).all())
        self.assertIsInstance(
            LogScalarParam(name="a", target="t", low=1, high=2), LogScalarParam
        )
        with self.assertRaises(ValueError):
            LogScalarParam(name="a", target="t", low=0, high=1)


class DesignTests(unittest.TestCase):
    def test_design_shapes_and_seeds(self):
        rng = np.random.default_rng(0)
        self.assertEqual(lhs_design(20, 3, rng).shape, (20, 3))
        morris = morris_design(5, 4, np.random.default_rng(1))
        self.assertEqual(morris["points"].shape, (25, 4))
        self.assertEqual(len(morris["steps"]), 20)
        self.assertTrue(((morris["points"] >= 0) & (morris["points"] <= 1)).all())
        saltelli = saltelli_design(64, 3, np.random.default_rng(2))
        self.assertEqual(saltelli["points"].shape, (64 * 5, 3))
        with self.assertRaises(ValueError):
            saltelli_design(60, 3, np.random.default_rng(2))
        spec = StudySpec(
            name="d",
            workflow="ensemble",
            model=ModelRef(model_dir="m", adapter="a"),
            parameters=[ParameterSpec(name="p", family="ScalarParam")],
            seed_root=3,
            design={"method": "lhs", "n": 4},
        )
        first = make_design(spec, 2)["points"]
        np.testing.assert_array_equal(first, make_design(spec, 2)["points"])

    def test_sobol_and_morris_estimators_on_ishigami(self):
        design = saltelli_design(2048, 3, np.random.default_rng(5))
        indices = sobol_indices(ishigami(design["points"]), design, ["x1", "x2", "x3"])
        self.assertAlmostEqual(indices["x1"]["S1"], 0.3139, delta=0.05)
        self.assertAlmostEqual(indices["x2"]["S1"], 0.4424, delta=0.05)
        self.assertAlmostEqual(indices["x3"]["S1"], 0.0, delta=0.05)
        self.assertAlmostEqual(indices["x3"]["ST"], 0.2437, delta=0.06)
        morris = morris_design(50, 3, np.random.default_rng(6))
        effects = morris_indices(ishigami(morris["points"]), morris, ["x1", "x2", "x3"])
        self.assertGreater(effects["x2"]["mu_star"], effects["x3"]["mu_star"])
        self.assertGreater(effects["x3"]["sigma"], 0.0)


class CliTests(unittest.TestCase):
    def test_estimate_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = StudySpec(
                name="e",
                workflow="ensemble",
                model=ModelRef(
                    model_dir=str(
                        Path(__file__).resolve().parents[2]
                        / "models"
                        / "Uniform_Brugge"
                    ),
                    adapter="workflows.adapters.brugge_proxy",
                ),
                parameters=[
                    ParameterSpec(
                        name="tm",
                        family="LogScalarParam",
                        args={"low": 0.5, "high": 2.0},
                    )
                ],
                seed_root=1,
                observations=ObservationSpec(
                    wells=["P1"], quantities=["oil_rate"], report_times=[10.0]
                ),
                design={"method": "lhs", "n": 12},
                compute=ComputeSpec(max_workers=4),
            )
            path = Path(tmp) / "study.json"
            spec.dump(path)
            import io
            from contextlib import redirect_stdout

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(["estimate", "--spec", str(path)])
            self.assertEqual(code, 0)
            out = json.loads(buffer.getvalue())
            self.assertEqual(out["planned_range"], [12, 12])
            self.assertEqual(out["simulations"], 12)
            self.assertEqual(out["workers"], 4)
            self.assertGreater(out["wall_s"][1], out["wall_s"][0])


if __name__ == "__main__":
    unittest.main()
