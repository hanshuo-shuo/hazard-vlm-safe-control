from evaluation.interface_contracts import (
    CONTRACT_IDS,
    build_contract_prompt,
    fixed_path_outcome,
    parse_contract_response,
    planner_action,
)
from scripts.run_interface_contract_audit import (
    MAX_CALLS,
    MODELS,
    SEEDS,
    build_matrix,
    build_scenes,
    candidate_interface,
    frozen_protocol,
    parse_candidate_response,
)
from scripts.run_interface_contract_replacement import (
    REPLACEMENT_MODEL,
    frozen_replacement_protocol,
    replacement_matrix,
)


def test_explicit_label_polarities_normalize_to_same_action():
    cases = {
        "constraint_positive": (
            '{"terrain_class":"water","constraint_applies":"yes","action":"avoid"}',
            False,
        ),
        "constraint_negative": (
            '{"terrain_class":"water","constraint_does_not_apply":"no","action":"avoid"}',
            False,
        ),
        "compatibility_positive": (
            '{"terrain_class":"water","terrain_compatible":"no","action":"avoid"}',
            False,
        ),
        "compatibility_negative": (
            '{"terrain_class":"water","terrain_incompatible":"yes","action":"avoid"}',
            False,
        ),
    }
    for contract_id, (raw, expected_safe) in cases.items():
        parsed = parse_contract_response(contract_id, raw)
        assert parsed["parse_status"] == "ok"
        assert parsed["semantic_safe_to_traverse"] is expected_safe
        assert parsed["label_action_consistent"]


def test_ambiguous_applicability_keeps_both_planner_interpretations():
    parsed = parse_contract_response(
        "ambiguous_applicability",
        '{"terrain_class":"water","applicability":"not_applicable","action":"avoid"}',
    )
    assert parsed["semantic_safe_to_traverse"] is None
    assert planner_action(parsed, "ambiguous_applicability", "action_authoritative") == "avoid"
    assert (
        planner_action(
            parsed,
            "ambiguous_applicability",
            "applicable_means_constraint_applies",
        )
        == "traverse"
    )
    assert (
        planner_action(
            parsed,
            "ambiguous_applicability",
            "applicable_means_terrain_compatible",
        )
        == "avoid"
    )


def test_field_order_pair_changes_only_requested_order_block():
    canonical = build_contract_prompt("action_structured", "amphibious").decode()
    reversed_fields = build_contract_prompt(
        "action_structured_field_reversed", "amphibious"
    ).decode()
    assert canonical.split("Return JSON only", 1)[0] == reversed_fields.split(
        "Return JSON only", 1
    )[0]
    assert '"terrain_class"' in canonical
    assert reversed_fields.index('"confidence"') < reversed_fields.index('"terrain_class"')


def test_frozen_matrix_uses_only_five_seeds_per_model():
    matrix = build_matrix(build_scenes())
    assert len(matrix) == MAX_CALLS == 330
    assert frozen_protocol()["maximum_distinct_seeds_per_model"] == 5
    assert set(CONTRACT_IDS) == {
        row["contract_id"]
        for row in matrix
        if row["task"] == "semantic_route_decision"
    }
    for model in MODELS:
        assert {row["seed"] for row in matrix if row["model"] == model} == set(SEEDS)


def test_marker_permutation_preserves_physical_choices_and_changes_labels():
    scene = build_scenes()[("safety_gym_goal", SEEDS[0])]
    _, baseline_records, baseline_map, baseline_safe = candidate_interface(
        scene, "candidate_baseline"
    )
    _, permuted_records, permuted_map, permuted_safe = candidate_interface(
        scene, "marker_id_permutation"
    )
    assert baseline_safe == permuted_safe
    assert [record["world_xy"] for record in baseline_records] == [
        record["world_xy"] for record in permuted_records
    ]
    assert baseline_map != permuted_map
    parsed = parse_candidate_response(
        '{"selected_marker":"1","route_policy":"avoid"}', permuted_map
    )
    assert parsed["physical_choice"] == permuted_map["1"]


def test_replacement_arm_changes_only_model_and_keeps_five_seeds():
    rows = replacement_matrix()
    assert len(rows) == 110
    assert {row["model"] for row in rows} == {REPLACEMENT_MODEL}
    assert {row["seed"] for row in rows} == set(SEEDS)
    protocol = frozen_replacement_protocol()
    assert protocol["only_changed_factor"] == "model_identity"
    assert protocol["prompts_images_seeds_unchanged"] is True


def test_unknown_action_is_noop_failure_not_implicit_traverse():
    scene = build_scenes()[("safety_gym_goal", SEEDS[0])]
    public_scene = {key: value for key, value in scene.items() if key != "base_png"}
    outcome = fixed_path_outcome(public_scene, "wheeled_non_waterproof", "unknown")
    assert outcome["success"] is False
    assert outcome["entered_terrain"] is False
    assert outcome["semantic_violation"] is False
    assert outcome["STC"] is False
