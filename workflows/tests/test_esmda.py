import unittest

import numpy as np

from workflows import gates
from workflows.esmda import ObservationLayout, gaspari_cohn
from workflows.spec import ObservationSpec


class GateTests(unittest.TestCase):
    def test_gates_on_synthetic_data(self):
        rng = np.random.default_rng(0)
        obs = np.zeros(50)
        sigma = np.ones(50)
        ens = rng.standard_normal((200, 50))
        self.assertAlmostEqual(gates.chi2(ens, obs, sigma), 1.0, delta=0.15)
        self.assertTrue(gates.chi2_band(1.0))
        self.assertFalse(gates.chi2_band(5.0))
        self.assertLess(gates.held_out_rmse(ens, obs, sigma), 0.3)
        self.assertAlmostEqual(gates.coverage(ens, obs, 0.8), 1.0)
        self.assertLess(gates.spread_ratio(0.5 * ens, ens), 0.6)
        self.assertAlmostEqual(gates.gradient_angle([1, 0], [0, 1]), 90.0)
        self.assertAlmostEqual(gates.regret(95.0, 100.0, 50.0), 0.1)
        self.assertIsNone(gates.regret(95.0, 100.0, 100.0))
        self.assertAlmostEqual(gates.regret(-95.0, -100.0, -50.0, maximize=False), 0.1)
        self.assertTrue(gates.budget_exceeded(11, 10))

    def test_gaspari_cohn_taper(self):
        r = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        t = gaspari_cohn(r, 1.0)
        self.assertAlmostEqual(t[0], 1.0)
        self.assertTrue(np.all(np.diff(t) <= 1e-12))
        self.assertAlmostEqual(t[4], 0.0, places=9)
        self.assertEqual(t[5], 0.0)
        self.assertTrue(np.all((t >= 0) & (t <= 1)))


class LayoutTests(unittest.TestCase):
    def test_layout_order_train_mask_and_sigma(self):
        obs = ObservationSpec(
            wells=["P1", "P2"],
            quantities=["oil_rate"],
            report_times=[10, 20, 30, 40, 50],
            held_out_fraction=0.4,
            sigma_rel=0.1,
            sigma_floor=1e-3,
        )
        layout = ObservationLayout(obs)
        self.assertEqual(len(layout.entries), 10)
        self.assertEqual(layout.train.sum(), 6)
        self.assertTrue(layout.train[:3].all() and not layout.train[3:5].any())
        observed = {
            "values": {
                "P1:oil_rate": [1, 2, 3, 4, 5],
                "P2:oil_rate": [10, 20, 30, 40, 50],
            }
        }
        vec = layout.vector(observed)
        np.testing.assert_array_equal(vec, [1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
        sigma = layout.sigma(vec)
        self.assertAlmostEqual(sigma[0], 0.1 * 1.0)  # sigma_rel * |d|
        self.assertAlmostEqual(sigma[9], 0.1 * 50.0)
        self.assertTrue((sigma >= 1e-3).all())
        geometry = {
            "wells": {
                "P1": {"xyz": [0.0, 0.0, 36.0]},
                "P2": {"xyz": [100.0, 0.0, 36.0]},
            }
        }
        self.assertEqual(layout.positions(geometry).shape, (10, 2))

    def test_linear_gaussian_twin_reduces_misfit(self):
        try:
            import dageo
        except ImportError:
            self.skipTest("dageo not installed")
        rng = np.random.default_rng(1)
        g = rng.standard_normal((20, 8))
        truth = rng.standard_normal(8)
        d_obs = g @ truth + 0.05 * rng.standard_normal(20)
        prior = truth + rng.standard_normal((40, 8))

        def forward(models):
            return models @ g.T

        post, _ = dageo.esmda(
            model_prior=prior,
            forward=forward,
            data_obs=d_obs,
            sigma=0.05,
            alphas=4,
            random=rng,
        )
        self.assertLess(
            gates.chi2(forward(post), d_obs, np.full(20, 0.05)),
            gates.chi2(forward(prior), d_obs, np.full(20, 0.05)),
        )
        self.assertLess(
            np.sqrt(np.mean((post.mean(axis=0) - truth) ** 2)),
            np.sqrt(np.mean((prior.mean(axis=0) - truth) ** 2)),
        )


if __name__ == "__main__":
    unittest.main()
