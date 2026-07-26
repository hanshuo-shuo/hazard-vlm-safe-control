from evaluation.interface_metrics import (
    canonical_semantic_consistency,
    contract_induced_safety_range,
    grounding_consistency,
    outcome_rates,
    parse_consistency,
    physical_action_iec,
    ranking_stability_envelope,
)


def _parsed(action: str, center=(0.5, 0.5)):
    return {
        "parse_status": "ok",
        "canonical_semantics": {
            "terrain_class": "water",
            "safe_to_traverse": False,
            "action": action,
        },
        "canonical_grounding": {
            "geometry_type": "disk",
            "center_norm": list(center),
            "radius_norm": 0.1,
        },
    }


def test_stage_consistency_metrics_are_separate():
    rows = [
        {"comparison_id": "a", "pair_side": "anchor", "parsed": _parsed("avoid"), "physical_action": "left"},
        {"comparison_id": "a", "pair_side": "mate", "parsed": _parsed("avoid", (0.51, 0.5)), "physical_action": "right"},
    ]
    assert parse_consistency(rows)["consistency"] == 1.0
    assert canonical_semantic_consistency(rows)["consistency"] == 1.0
    assert grounding_consistency(rows)["consistency"] == 1.0
    assert physical_action_iec(rows)["consistency"] == 0.0


def test_cisr_outcomes_and_rank_envelope_golden_fixture():
    rows = [
        {"model_budget_id": "a", "condition_id": "c1", "family": "f", "contract": "x", "STC": True, "semantic_violation": False, "collision": False, "false_conservative_detour": False},
        {"model_budget_id": "a", "condition_id": "c2", "family": "f", "contract": "y", "STC": False, "semantic_violation": True, "collision": False, "false_conservative_detour": False},
        {"model_budget_id": "b", "condition_id": "c1", "family": "f", "contract": "x", "STC": False, "semantic_violation": False, "collision": True, "false_conservative_detour": False},
        {"model_budget_id": "b", "condition_id": "c2", "family": "f", "contract": "y", "STC": True, "semantic_violation": False, "collision": False, "false_conservative_detour": True},
    ]
    cisr = contract_induced_safety_range(rows, group_keys=["model_budget_id", "family"], intervention_key="contract")
    assert cisr["maximum_range"] == 1.0
    rates = outcome_rates(rows)
    assert rates["STC"] == 0.5
    assert rates["collision"] == 0.25
    rank = ranking_stability_envelope(rows)
    assert rank["pairwise_rank_reversals"] == [["a", "b"]]
