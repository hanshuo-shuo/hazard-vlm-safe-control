import unittest
import numpy as np
from analyze import Curve, choose


class Curves(unittest.TestCase):
    def test_nonmonotone_conditional_risk(self):
        # An unsafe first item need not preclude a later valid prefix.
        c=Curve([0,1,2,3],[1,0,0,0])
        r=c.envelope([.3],3)[0]
        self.assertEqual((r['accepted'],r['safe'],r['unsafe']),(4,3,1))

    def test_equal_scores_cannot_be_split(self):
        c=Curve([0,0,1],[0,1,0])
        r=c.envelope([.1],2)[0]
        self.assertIsNone(r['risk'])
        self.assertEqual(r['accepted'],0)

    def test_weighted_bootstrap_matches_literal_repetition(self):
        score=np.array([.1,.2,.2,.4]);bad=np.array([0,1,0,0]);w=np.array([2,1,3,0])
        a=Curve(score,bad).envelope([0,.1,.2,.5],5,w)
        b=Curve(np.repeat(score,w),np.repeat(bad,w)).envelope([0,.1,.2,.5],5)
        self.assertEqual(a,b)

    def test_original_tie_rule(self):
        c=np.array([[.1,.1+5e-10,.2,.3],[.1,.1+2e-9,.2,.3]])
        self.assertEqual(choose(c,np.array([.75,.75])).tolist(),[1,0])


if __name__=='__main__':unittest.main()
