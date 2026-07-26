from scripts.run_provider_free_geometry_calibration import (
    FRESH_DEV_SPLIT,
    run_safety_gym_twins,
)


def test_fresh_dev_split_is_disjoint_and_excludes_historical_pilot():
    historical = set(FRESH_DEV_SPLIT["historical_pilot_excluded"])
    calibration = set(FRESH_DEV_SPLIT["calibration_seeds"])
    attribution = set(FRESH_DEV_SPLIT["attribution_seeds"])
    reserve = set(FRESH_DEV_SPLIT["held_out_dev_reserve"])
    assert historical == set(range(5))
    assert not historical & calibration
    assert not historical & attribution
    assert not calibration & attribution
    assert not calibration & reserve
    assert not attribution & reserve
    assert calibration | attribution | reserve == set(range(5, 43))


def test_safety_gym_oracle_capability_twin_reverses_route_locally():
    result = run_safety_gym_twins()
    assert result["provider_calls"] == 0
    assert result["behavior_reversal_observed"] is True
    assert all(result["twin_invariants"].values())
    wheeled, amphibious = result["rows"]
    assert wheeled["expected_applicability"] == "applicable"
    assert wheeled["entered_water_geometry"] is False
    assert amphibious["expected_applicability"] == "not_applicable"
    assert amphibious["entered_water_geometry"] is True
