"""Disjoint fixtures, executed on a Quest CPU allocation before the freeze."""
import copy
import ast
from pathlib import Path
import unittest
import numpy as np
from common import protocol, reference
from generator import anonymous_geometry, anonymous_paths, make_random, exposures, formula, stream
from metrics import pair_validity, per_scene, calibrate, upper_cost, coverage_counts
from aggregate import interval


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = protocol()
        cls.spec = {"scene_seed": 910077, "scene_count": 96,
                    "appearance_families": ["aqua_sand", "rose_moss_v1"],
                    "shapes": ["ellipse", "box", "diamond", "superellipse"]}
        cls.data, cls.records = make_random(cls.spec, cls.p)

    def test_reference_identity(self):
        reference.verify_reference(self.p)

    def test_all_sources_parse(self):
        for path in Path(__file__).parent.glob("*.py"):
            if not path.name.startswith("._"):
                ast.parse(path.read_text())

    def test_crossed_bootstrap_constant_and_pairing(self):
        rng = np.random.default_rng(9331)
        sw = rng.multinomial(5, np.ones(5) / 5, size=100) / 5
        nw = rng.multinomial(20, np.ones(20) / 20, size=100)
        numerator = np.ones((5, 20)) * .3
        result, crossed, conditional = interval(numerator, np.ones_like(numerator), sw, nw)
        np.testing.assert_allclose(crossed, .3, atol=1e-15)
        np.testing.assert_allclose(conditional, .3, atol=1e-15)
        other, cx, cc = interval(numerator + .1, np.ones_like(numerator), sw, nw)
        np.testing.assert_allclose(cx - crossed, .1, atol=1e-15)


    def test_visual_intervention(self):
        d = self.data
        np.testing.assert_array_equal(d["target"][:, 0, ::-1], d["target"][:, 1])
        np.testing.assert_allclose(d["exposure"][:, 0, :, ::-1], d["exposure"][:, 1], atol=1e-14)
        self.assertTrue(all(not np.array_equal(x[0], x[1]) for x in d["rgb"]))

    def test_no_overlap_and_visible(self):
        for i in range(32):
            f, _ = anonymous_geometry(910077, i, self.spec["shapes"], self.p["generator"])
            self.assertFalse((f[0] * f[1]).any())
            self.assertTrue((f.sum((1, 2)) >= 18).all())

    def test_stream_separation(self):
        before, _ = anonymous_paths(910077, 0, self.p["generator"])
        anonymous_geometry(910077, 0, ["diamond"], self.p["generator"])
        stream(910077, 0, 17).permutation(2)
        after, _ = anonymous_paths(910077, 0, self.p["generator"])
        np.testing.assert_array_equal(before, after)

    def test_deterministic_replay(self):
        spec = dict(self.spec, scene_count=2)
        second, records = make_random(spec, self.p)
        for key in second:
            np.testing.assert_array_equal(second[key], self.data[key][:2])
        self.assertEqual(records, self.records[:2])

    def test_formula_against_reference(self):
        for e, truth in zip(self.data["exposure"][:8, 0], self.data["truth"][:8, 0]):
            np.testing.assert_allclose(truth, reference.risk_truth(e), atol=1e-15)

    def test_pool_against_explicit_sum(self):
        d = self.data
        for n in range(4):
            for v in range(2):
                for k in range(4):
                    expected = [(d["target"][n, v, c].astype(float) * d["footprints"][n, k]).sum()
                                / d["footprints"][n, k].sum() for c in range(2)]
                    np.testing.assert_allclose(d["exposure"][n, v, k], expected, atol=1e-15)

    def test_oracle_and_ties(self):
        t = self.data["truth"]
        r = per_scene(t, t, self.p)
        np.testing.assert_allclose(r["regret"], 0, atol=1e-15)
        np.testing.assert_allclose(r["paired_correct_sum"], r["paired_eligible"])
        r = per_scene(t, np.zeros_like(t), self.p)
        np.testing.assert_allclose(r["paired_correct_sum"], r["paired_eligible"] / 16)
        self.assertTrue((r["false_safe_count"] == r["false_safe_eligible"]).all())

    def test_optimum_gap_and_invalid_pairs(self):
        t = np.zeros((2, 2, 4, 4))
        t[0, 0, :, 0] = [0, .1, .2, .3]
        t[0, 1, :, 0] = [.1, 0, .2, .3]
        t[1, 0, :, 0] = [0, .001, .2, .3]
        t[1, 1, :, 0] = [.1, 0, .2, .3]
        valid, _ = pair_validity(t, .002)
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[1].any())

    def test_quantile_and_coverage(self):
        t = self.data["truth"]
        p = np.maximum(t - .01, 0)
        q, scores, rank = calibrate(t, p, .05)
        self.assertEqual(rank, int(np.ceil(97 * .95)))
        self.assertAlmostEqual(q, .01)
        upper = upper_cost(p, q)
        self.assertTrue((upper >= t - 1e-15).all())
        np.testing.assert_array_equal(upper[..., 1], 0)
        rows = coverage_counts(t, upper, self.p)
        self.assertTrue((np.diff(rows[:, :, 0], axis=1) >= 0).all())
        np.testing.assert_array_equal(rows[:, -1, 0], 6)
        all_abstain = coverage_counts(t, upper_cost(p, 1), self.p)
        np.testing.assert_array_equal(all_abstain[:, :-1, 0], 0)

    def test_no_cost_filter_and_random_assignment(self):
        self.assertTrue((self.data["exposure"] == 0).any())
        valid, _ = pair_validity(self.data["truth"], .002)
        self.assertTrue(valid.any())
        self.assertTrue((~valid[:, 0]).any())
        assignments = [r["assignment"][0] for r in self.records]
        self.assertGreater(sum(assignments), 20)
        self.assertLess(sum(assignments), 76)


if __name__ == "__main__":
    unittest.main(verbosity=2)
