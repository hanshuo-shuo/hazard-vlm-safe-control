import unittest
import math
import numpy as np
from common import *
from run_e3 import certify

class CertificationTests(unittest.TestCase):
    def test_zero_error_rule_is_certified_with_sufficient_independent_samples(self):
        p=protocol()
        d=deployment(98324,300)
        truth=np.zeros((300,2,4,4))
        pred=np.zeros_like(truth)
        c=certify(truth,pred,d,p)
        self.assertEqual(c["family_size"],120)
        self.assertAlmostEqual(c["delta_per_rule"]*120,.05)
        self.assertIsNotNone(c["chosen"])
        self.assertEqual(c["chosen"]["threshold"],0)
        self.assertEqual(c["chosen"]["accepted"],300)
        self.assertLess(c["chosen"]["upper"],.05)

    def test_no_acceptance_cannot_be_certified(self):
        p=protocol()
        d=deployment(98325,300)
        c=certify(np.zeros((300,2,4,4)),np.ones((300,2,4,4)),d,p)
        self.assertIsNone(c["chosen"])

    def test_unsafe_actions_are_not_certified_by_multiplying_correlated_rows(self):
        p=protocol()
        d=deployment(98326,50)
        truth=np.zeros((50,2,4,4))
        truth[:10]=.1
        pred=np.zeros_like(truth)
        c=certify(truth,pred,d,p)
        self.assertIsNone(c["chosen"])
        for r in c["candidate_rules"]:
            self.assertEqual(r["accepted"],50)
            self.assertEqual(r["unsafe_accepted"],10)

    def test_binomial_tail_inversion_at_certified_boundary(self):
        n,k,delta=400,4,.05/120
        upper=cp_upper(k,n,delta)
        cdf=sum(math.comb(n,j)*upper**j*(1-upper)**(n-j) for j in range(k+1))
        self.assertAlmostEqual(cdf,delta,places=10)

if __name__=="__main__":
    unittest.main(verbosity=2)
