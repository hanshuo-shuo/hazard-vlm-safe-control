import unittest
from decision_contract import interval_from_counts, qualification_decision, REQUIRED_EVIDENCE


class DecisionContractTest(unittest.TestCase):
    def test_unknown_below_five_percent_does_not_make_compliant(self):
        self.assertEqual(interval_from_counts([(1, 4, 100)]),
                         {'lower': .01, 'upper': .05, 'state': 'undetermined'})

    def test_full_area_and_asymmetric_boundary(self):
        self.assertEqual(interval_from_counts([(0, 2, 100)])['state'], 'compliant')
        self.assertEqual(interval_from_counts([(2, 3, 100)])['state'], 'undetermined')
        self.assertEqual(interval_from_counts([(10, 80, 100)])['state'], 'noncompliant')
        with self.assertRaises(ValueError):
            interval_from_counts([(1, 4, 100), (1, 0, 5)])

    def test_geometric_realizations_can_revoke_apparent_compliance(self):
        self.assertEqual(interval_from_counts([(0, 0, 100)])['state'], 'compliant')
        self.assertEqual(interval_from_counts([(0, 0, 100), (1, 4, 100)])['state'], 'undetermined')
        self.assertEqual(interval_from_counts([])['state'], 'undetermined')

    def scenes(self):
        return [{'id': str(i), 'block': str(i // 4), 'candidates': [
            {'state': 'compliant'}, {'state': 'noncompliant'},
            {'state': 'compliant'}, {'state': 'compliant'}]} for i in range(48)]

    def test_binary_path_never_grants_continuous_metrics(self):
        evidence = dict.fromkeys(REQUIRED_EVIDENCE, True)
        result = qualification_decision(self.scenes(), evidence, .01)
        self.assertEqual(result['decision'], 'BINARY_ONLY_PASS')
        self.assertFalse(result['continuous_cost_qualified'])
        self.assertEqual(qualification_decision(self.scenes(), evidence, .005)['decision'], 'FULL_PASS')

    def test_missing_or_all_compliant_scenes_cannot_be_replaced(self):
        scenes, evidence = self.scenes(), dict.fromkeys(REQUIRED_EVIDENCE, True)
        for scene in scenes[:10]:
            scene['candidates'] = []
        self.assertEqual(qualification_decision(scenes, evidence, .001)['decision'], 'FAIL_OR_UNCERTAIN')
        with self.assertRaises(ValueError):
            qualification_decision(scenes[10:], evidence, .001)
        for scene in scenes:
            scene['candidates'] = [{'state': 'compliant'}] * 4
        self.assertEqual(qualification_decision(scenes, evidence, .001)['decision'], 'FAIL_OR_UNCERTAIN')

    def test_coverage_does_not_override_physical_evidence_or_false_support(self):
        evidence = dict.fromkeys(REQUIRED_EVIDENCE, True)
        self.assertEqual(qualification_decision(self.scenes(), evidence, .001, True)['decision'], 'FAIL_OR_UNCERTAIN')
        evidence['coordinates_and_gravity'] = None
        self.assertEqual(qualification_decision(self.scenes(), evidence, .001)['decision'], 'FAIL_OR_UNCERTAIN')


if __name__ == '__main__':
    unittest.main()
