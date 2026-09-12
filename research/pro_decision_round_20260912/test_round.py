import ast
import math
import unittest
from pathlib import Path
import numpy as np
from common import *
from geometry import continuous_scene, materialize_record, raster, boundary_band

class RoundTests(unittest.TestCase):
    def test_source_syntax(self):
        for f in Path(__file__).parent.glob("*.py"):
            if not f.name.startswith("._"):
                ast.parse(f.read_text())

    def test_cp_zero_count_resolution(self):
        self.assertGreater(cp_upper(0, 58), .05)
        self.assertLessEqual(cp_upper(0, 59), .05)
        self.assertGreater(cp_upper(0, 298), .01)
        self.assertLessEqual(cp_upper(0, 299), .01)
        self.assertEqual(cp_upper(0, 0), 1)

    def test_cp_nonzero_against_exact_polynomial(self):
        self.assertAlmostEqual(cp_upper(1, 2), math.sqrt(.95), places=12)
        self.assertGreater(cp_upper(1, 100, .001), cp_upper(1, 100, .05))

    def test_formula_matches_reference(self):
        e = np.random.default_rng(714).uniform(size=(8, 4, 2))
        for a, b in zip(formula(e), e):
            np.testing.assert_allclose(a, reference.risk_truth(b), atol=1e-15)

    def test_tie_selection_uniform_and_repeatable(self):
        cost = np.zeros((1000, 2, 4, 4))
        u = deployment(713, len(cost))["tie_uniform"]
        ids = choose(cost, u)
        np.testing.assert_array_equal(ids, np.floor(u * 4))
        np.testing.assert_array_equal(ids, choose(cost, u))

    def test_actual_deployment_excludes_trivial_card(self):
        d = deployment(182, 100)
        self.assertTrue(np.isin(d["deploy_card"], [0, 2, 3]).all())
        self.assertEqual(len(d["deploy_card"]), 100)

    def test_eta_does_not_count_unsafe_accepts(self):
        t = np.zeros((30, 2, 4, 4))
        t[:, :, 0, :] = .1
        p = np.ones_like(t)
        p[:, :, 0] = 0
        d = deployment(784, len(t))
        r = summarize(rows(t, p, d))
        self.assertEqual(r["coverage"], 1)
        self.assertEqual(r["eta"], 0)
        self.assertEqual(r["conditional_unsafe"], 1)
        abstained = summarize(rows(t, p, d, offset=.1))
        self.assertIsNone(abstained["conditional_unsafe"])

    def test_nested_scores_per_scene_not_flattened(self):
        d = deployment(734, 20)
        pred = np.random.default_rng(75).random((20, 2, 4, 4))
        truth = np.random.default_rng(76).random((20, 2, 4, 4))
        s = nested_scores(truth, pred, d)
        self.assertTrue(all(x.shape == (20,) for x in s.values()))
        self.assertTrue((s["pair_24_entries"] >= s["current_four_candidates"]).all())
        self.assertTrue((s["current_four_candidates"] >= s["fixed_selected_action"]).all())

    def test_continuous_identity_and_rgb_raster_invariance(self):
        p = protocol()
        for geometry in ["balanced_anchor", "independent"]:
            r = continuous_scene(97144, 1, geometry, p["appearance"]["train"], p)
            a = materialize_record(r, p, references=False)
            b = materialize_record(r, p, references=False)
            np.testing.assert_array_equal(a["rgb"], b["rgb"])
            self.assertEqual(a["target48"].shape, a["target64"].shape)
            self.assertEqual(r, continuous_scene(97144, 1, geometry, p["appearance"]["train"], p))

    def test_reference_mean_against_direct_boolean_sum(self):
        p = protocol()
        r = continuous_scene(97144, 9, "independent", p["appearance"]["train"], p)
        f, w, e = raster(r, 256)
        for k in range(4):
            for c in range(2):
                self.assertAlmostEqual(e[0,k,c], f[c][w[k] > 0].mean(), places=7)
        np.testing.assert_array_equal(e[0, :, ::-1], e[1])

    def test_crossed_ratio_constant(self):
        r = crossed_ratio(np.ones((5, 100)) * .4, np.ones((5, 100)), seed=442, resamples=100)
        np.testing.assert_allclose(r["ci95"], [.4, .4], atol=1e-15)

    def test_boundary_band_not_entire_interior(self):
        a = np.zeros((1, 2, 2, 16, 16))
        a[..., 4:12, 4:12] = 1
        b = boundary_band(a)
        self.assertTrue(b[..., 4, 4].all())
        self.assertFalse(b[..., 8, 8].any())

if __name__ == "__main__":
    unittest.main(verbosity=2)
