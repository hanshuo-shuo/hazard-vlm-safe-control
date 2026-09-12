"""Distinct numerical and unknown-handling checks; run only on Quest."""
import json
from pathlib import Path
import unittest
import numpy as np
from geometry_audit import (corridor_queries, cost_interval, fit_surface,
                            project, visible_support)


class GeometryTests(unittest.TestCase):
    def test_metric_equal_area_and_forward_extent(self):
        protocol = json.loads((Path(__file__).parent / "qualification_protocol.json").read_text())
        q = corridor_queries(protocol)
        self.assertEqual(q.shape, (4, 720, 2))
        for points, theta in zip(q, np.deg2rad([-12, -4, 4, 12])):
            progress = points @ np.array([np.cos(theta), np.sin(theta)])
            self.assertAlmostEqual(progress.min(), 6.05)
            self.assertAlmostEqual(progress.max(), 11.95)

    def test_void_is_unknown_not_safe(self):
        labels = np.array([[0, 3, 6]], np.uint8)
        lo, hi, known, _ = cost_interval(labels, np.array([[0,0],[1,0],[2,0]]), np.ones(3,bool), [6], [3])
        self.assertAlmostEqual(lo, 1/3)
        self.assertAlmostEqual(hi, 2/3)
        self.assertEqual(known.tolist(), [False, True, True])

    def test_unsupported_labeled_grass_remains_unknown(self):
        lo, hi, known, _ = cost_interval(np.full((2,2),3), np.array([[0,0],[1,1]]), np.zeros(2,bool), [6], [3])
        self.assertEqual((lo, hi), (0., 1.))
        self.assertFalse(known.any())

    def test_direct_optical_axis_projection(self):
        k = np.array([[500,0,300],[0,500,200],[0,0,1]])
        uv, depth = project(np.array([[0.,0.,5.],[0.,0.,-5.]]), k, np.eye(4))
        np.testing.assert_array_equal(uv[0], [300,200])
        self.assertGreater(depth[0],0)
        self.assertLess(depth[1],0)

    def test_nonsemantic_plane_with_vertical_outliers(self):
        rng = np.random.default_rng(44)
        xy = rng.uniform([2,-5],[15,5],(12000,2))
        z = -1 + .02*xy[:,0] + rng.normal(0,.005,len(xy))
        points = np.column_stack([xy,z])
        obstacle = np.column_stack([rng.uniform([5,-1],[7,1],(2000,2)),rng.uniform(-.6,2,2000)])
        coef, ground, _ = fit_surface(np.vstack([points,obstacle]))
        self.assertLess(abs(coef[0]-.02), .01)
        self.assertLess(abs(coef[2]+1), .05)
        self.assertGreater(len(ground),11000)

    def test_front_depth_prevents_hidden_ground_acceptance(self):
        k = np.array([[100.,0,100],[0,100,100],[0,0,1]])
        transform = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1.]])
        ground = np.array([[8+dx,dy,-1.] for dx in [-.1,0,.1] for dy in [-.1,0,.1]])
        blocker = np.array([[4.,0.,-.5]])
        _, _, known, reasons, _ = visible_support(np.vstack([ground,blocker]),np.array([[8.,0.]]),np.array([0.,0.,-1.]),ground,k,transform,(200,200))
        self.assertFalse(known[0])
        self.assertTrue(reasons["occluded"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
