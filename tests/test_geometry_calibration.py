import pytest

from evaluation.geometry_calibration import (
    CalibrationBudget,
    GeometryArm,
    compose_geometry_arm,
    disk_metrics,
    pareto_rows,
    validate_operating_curve,
)


def test_disk_metrics_identical_and_disjoint():
    same = disk_metrics((0, 0, 1), (0, 0, 1))
    assert same["iou"] == pytest.approx(1)
    assert same["false_positive_area"] == pytest.approx(0)
    apart = disk_metrics((5, 0, 1), (0, 0, 1))
    assert apart["iou"] == 0


def test_missing_detector_is_not_fabricated():
    assert all(value is None for value in disk_metrics(None, (0, 0, 1)).values())


def test_test_split_requires_freeze_and_single_consumption():
    with pytest.raises(PermissionError):
        CalibrationBudget("test", 1, 0)
    with pytest.raises(PermissionError):
        CalibrationBudget("test", 1, 0, parameters_frozen=True, test_run_consumed=True)


def test_operating_curve_one_family_and_unique_hashes():
    validate_operating_curve([
        {"parameter_family": "soft_halo", "config_hash": "a"},
        {"parameter_family": "soft_halo", "config_hash": "b"},
    ])
    with pytest.raises(ValueError):
        validate_operating_curve([
            {"parameter_family": "soft_halo", "config_hash": "a"},
            {"parameter_family": "penalty", "config_hash": "b"},
        ])


def test_decomposition_composes_centers_and_never_fabricates_mask():
    detector = (1, 2, 0.5)
    oracle = (0, 0, 1.0)
    mixed = compose_geometry_arm(
        GeometryArm.DETECTOR_CENTER_ORACLE_RADIUS,
        detector_disk=detector,
        oracle_disk=oracle,
    )
    assert mixed["center_xy"] == [1, 2]
    assert mixed["radius"] == 1
    assert compose_geometry_arm(
        GeometryArm.DETECTOR_MASK,
        detector_disk=detector,
        oracle_disk=oracle,
    ) is None


def test_pareto_curve_removes_dominated_rows():
    rows = [
        {"parameter_family": "halo", "config_hash": "a", "success_rate": 0.8, "semantic_violation_rate": 0.1},
        {"parameter_family": "halo", "config_hash": "b", "success_rate": 0.7, "semantic_violation_rate": 0.2},
        {"parameter_family": "halo", "config_hash": "c", "success_rate": 0.6, "semantic_violation_rate": 0.0},
    ]
    assert [row["config_hash"] for row in pareto_rows(rows)] == ["a", "c"]
