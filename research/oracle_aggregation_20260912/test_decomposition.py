import unittest
import numpy as np
from decompose import block_mean, formula, choose, selected, deployed

class DecompositionTests(unittest.TestCase):
    def test_identical_means_opposite_exposure(self):
        p=np.array([[1.,0.],[1.,0.]])
        f1=p.copy(); f2=1-p
        self.assertEqual(float(block_mean(p,1)[0,0]),.5)
        for f,true in [(f1,1.),(f2,0.)]:
            ref=(p*f).sum()/f.sum()
            pooled=float(block_mean(p,1)*block_mean(f,1)/block_mean(f,1))
            cov=(p*f).mean()-p.mean()*f.mean()
            self.assertEqual(ref,true)
            self.assertEqual(pooled,.5)
            self.assertAlmostEqual(pooled-ref,-cov/f.mean())

    def test_block_axes_and_mass(self):
        a=np.arange(2*8*8).reshape(2,8,8)
        b=block_mean(a,2)
        for c in range(2):
            for i in range(2):
                for j in range(2):
                    self.assertEqual(b[c,i,j],a[c,i*4:(i+1)*4,j*4:(j+1)*4].mean())
        np.testing.assert_array_equal(b.sum((1,2))*16,a.sum((1,2)))

    def test_nonlinear_cost_telescopes_but_not_linear_exposure(self):
        e=np.array([[.1,.2],[.2,.4],[.3,.5],[.4,.6]])
        c=formula(e)[:,2]
        self.assertAlmostEqual(np.diff(c).sum(),c[-1]-c[0])
        self.assertNotAlmostEqual(c[-1]-c[0],(e[-1]-e[0]).sum())

    def test_no_joint_boundary_zero_covariance(self):
        p=np.array([[1.,1.],[0.,0.]])
        for f in [np.zeros((2,2)),np.ones((2,2))]:
            self.assertEqual((p*f).mean()-p.mean()*f.mean(),0)

    def test_fixed_selection_cannot_silently_reselect(self):
        pred=np.array([[[[.01,.0,.01,.01],[.03,.0,.03,.03]]]])
        truth=pred[:,:,::-1].copy()
        ids=choose(pred,np.zeros((1,1,4)))
        self.assertEqual(float(selected(truth,ids)[0,0,0]),.03)
        self.assertEqual(float(selected(truth,choose(truth,np.zeros((1,1,4))))[0,0,0]),.01)

    def test_threshold_asymmetry_and_cancellation(self):
        costs=np.array([.02,np.nextafter(.02,np.inf)])
        np.testing.assert_array_equal(costs>.02,[False,True])
        parts=np.array([.01,-.012,.003])
        self.assertLess(abs(parts.sum()),abs(parts).sum())

if __name__=="__main__": unittest.main()
