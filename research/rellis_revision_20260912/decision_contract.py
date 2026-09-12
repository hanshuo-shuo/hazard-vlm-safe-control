"""Full-area empirical intervals and preregistered qualification decisions.

This module never supplies geometric truth. Its inputs must come from an
independently qualified measurement interface. Exact fractions avoid rounding
changing the <= / > boundary at tau. Missing candidates remain undetermined.
"""
from fractions import Fraction
import math


def interval_from_counts(realizations, tau=Fraction(1, 50)):
    """realizations: nonempty [(known_bad, unknown, full_sample_count), ...]."""
    if not realizations:
        return {'lower': 0.0, 'upper': 1.0, 'state': 'undetermined'}
    totals = {total for _, _, total in realizations}
    if len(totals) != 1:
        raise ValueError('Every realization must retain the same full-area denominator')
    for bad, unknown, total in realizations:
        if any(type(x) is not int for x in (bad, unknown, total)):
            raise ValueError('Equal-area sample counts must be integers')
        if total <= 0 or bad < 0 or unknown < 0 or bad + unknown > total:
            raise ValueError('Invalid full-area counts')
    low = min(Fraction(b, n) for b, _, n in realizations)
    high = max(Fraction(b + u, n) for b, u, n in realizations)
    state = 'compliant' if high <= tau else 'noncompliant' if low > tau else 'undetermined'
    return {'lower': float(low), 'upper': float(high), 'state': state}


REQUIRED_EVIDENCE = (
    'time_pose_chain', 'coordinates_and_gravity', 'independent_registration',
    'support_and_visibility', 'measured_error_propagation',
    'complete_support_spatial_isolation', 'independent_human_review',
)


def qualification_decision(scenes, evidence, continuous_width_p95, systematic_false_support=False):
    """Exactly 48 fixed identities; missing measurements are kept in scenes."""
    if len(scenes) != 48 or len({r['id'] for r in scenes}) != 48:
        raise ValueError('Qualification requires all 48 unique fixed identities')
    if len({r['block'] for r in scenes}) < 12:
        raise ValueError('Qualification requires at least 12 frozen spatial blocks')
    determined, mixed = [], []
    for scene in scenes:
        candidates = scene.get('candidates', [])
        if len(candidates) != 4:
            continue
        states = {c['state'] for c in candidates}
        if not states <= {'compliant', 'noncompliant'}:
            continue
        determined.append(scene)
        if len(states) == 2:
            mixed.append(scene)
    missing = [k for k in REQUIRED_EVIDENCE if evidence.get(k) is not True]
    reasons = missing.copy()
    if systematic_false_support:
        reasons.append('systematic_false_support_changing_labels')
    if len(determined) < 39:
        reasons.append('insufficient_all_four_determined')
    if len(mixed) < 8 or len({r['block'] for r in mixed}) < 4:
        reasons.append('insufficient_mixed_decision_content')
    if reasons:
        decision = 'FAIL_OR_UNCERTAIN'
    elif continuous_width_p95 is not None and math.isfinite(continuous_width_p95) and 0 <= continuous_width_p95 <= .005:
        decision = 'FULL_PASS'
    else:
        decision = 'BINARY_ONLY_PASS'
    return {'decision': decision, 'reasons': reasons, 'total': len(scenes),
            'all_four_determined': len(determined), 'mixed': len(mixed),
            'mixed_blocks': len({r['block'] for r in mixed}),
            'continuous_cost_qualified': decision == 'FULL_PASS'}
