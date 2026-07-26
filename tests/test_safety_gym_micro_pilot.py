import json

from scripts.run_safety_gym_micro_pilot import (
    MAX_CALLS,
    MAX_DISTINCT_SEEDS,
    build_matrix,
    frozen_protocol,
    parse_response,
)


def test_micro_pilot_uses_exactly_five_seeds_and_is_not_formal_scale_up():
    matrix = build_matrix()
    assert len(matrix) == MAX_CALLS == 100
    assert len({row["seed"] for row in matrix}) == MAX_DISTINCT_SEEDS == 5
    assert {row["seed"] for row in matrix} == set(range(20, 25))
    protocol = frozen_protocol()
    assert protocol["formal_scale_up_allowed"] is False
    assert protocol["base_calls"] == 40
    assert protocol["irrelevant_capability_calls"] == 40
    assert protocol["capability_omitted_calls"] == 20
    assert protocol["excluded_family"] == "clearance_footprint"


def test_micro_pilot_structured_parser_is_strict():
    valid = {
        "recognized_terrain": True,
        "terrain_class": "water",
        "applicability": "applicable",
        "reason_code": "TERRAIN_INCOMPATIBLE_WITH_CAPABILITY",
        "route_policy": "avoid",
    }
    parsed = parse_response(json.dumps(valid), "water")
    assert parsed["parse_status"] == "ok"
    assert parsed["recognition_correct"] is True
    invalid = parse_response(json.dumps({**valid, "extra": 1}), "water")
    assert invalid["parse_status"] == "error"
