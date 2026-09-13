import unittest
import numpy as np
from audit import derive, standardized


class AuditTests(unittest.TestCase):
    def test_single_term_substitutions_and_cancellation(self):
        d={k:np.array([v]) for k,v in dict(q0=.021279,q1=.016964,t0=.031961,t1=.021431,r=.024972).items()}
        x=derive(d)
        self.assertAlmostEqual(float(x['q10'][0]),.010749)
        self.assertAlmostEqual(float(x['q01'][0]),.027494)
        self.assertTrue(x['both_improve'][0] and x['worse'][0] and x['harm'][0])
        self.assertLess(x['identity_max'],1e-12)

    def test_oracle_flag_requires_reference(self):
        d={k:np.array([v,v]) for k,v in dict(q0=.03,q1=.01,t0=.04,t1=.02).items()}
        d['r']=np.array([.035,.05])
        self.assertEqual(derive(d)['flag'].tolist(),[True,False])

    def test_equal_within_stratum_rates_remove_composition_difference(self):
        f=np.array([1]*10+[0]*20+[1]*20+[0]*10,dtype=bool)
        c=np.array([0]*30+[1]*30)
        y=np.array([1]+[0]*9+[1]*2+[0]*18+[1]*18+[0]*2+[1]*9+[0],dtype=bool)
        self.assertNotEqual(y[f].mean(),y[~f].mean())
        r=standardized(f,y,np.ones(60,bool),c)
        self.assertAlmostEqual(r['difference'],0)
        self.assertEqual(r['overlap_weight'],20)

    def test_hybrids_are_not_clipped(self):
        d={k:np.array([v]) for k,v in dict(q0=.01,q1=.02,t0=.5,t1=0,r=.02).items()}
        x=derive(d)
        self.assertAlmostEqual(float(x['q10'][0]),-.49)
        self.assertFalse(x['harm'][0])


if __name__=='__main__':unittest.main()
