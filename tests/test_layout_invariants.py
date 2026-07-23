"""Regression tests for semantic-zone layout validity.

The invariant sweep intentionally exercises the public ``reset`` path.  It
can be shortened for a quick local smoke run with, for example,
``LAYOUT_TEST_SEEDS=25 pytest tests/ -x``; the default is the full 10,000-seed
sweep specified by WP-1.2.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import os

import numpy as np
import pytest

from env_pointhazard import PointHazardConfig, PointHazardEnv, zone_layout_valid


SEED_COUNT = int(os.environ.get("LAYOUT_TEST_SEEDS", "10000"))
MAX_SWEEP_WORKERS = int(os.environ.get("LAYOUT_TEST_WORKERS", "8"))
GOLDEN_SEEDS = (0, 1, 2, 3, 4)
FALLBACK_REGRESSION_SEEDS = (157, 1347)


def _layout_cases() -> tuple[object, ...]:
    """Return the four layout factors required by WP-1.2."""
    return (
        pytest.param(
            "single_implicit_corridor_on",
            PointHazardConfig(
                n_semantic_zones=1,
                semantic_on_corridor=True,
                semantic_styles=(),
            ),
            id="single-implicit-corridor-on",
        ),
        pytest.param(
            "single_implicit_corridor_off",
            PointHazardConfig(
                n_semantic_zones=1,
                semantic_on_corridor=False,
                semantic_styles=(),
            ),
            id="single-implicit-corridor-off",
        ),
        pytest.param(
            "three_hetero_corridor_on",
            PointHazardConfig(
                n_semantic_zones=3,
                semantic_on_corridor=True,
                semantic_styles=("water", "mud", "grass"),
            ),
            id="three-hetero-corridor-on",
        ),
        pytest.param(
            "three_hetero_corridor_off",
            PointHazardConfig(
                n_semantic_zones=3,
                semantic_on_corridor=False,
                semantic_styles=("water", "mud", "grass"),
            ),
            id="three-hetero-corridor-off",
        ),
    )


def _overlap_counts(env: PointHazardEnv) -> dict[str, int]:
    """Count every semantic-zone overlap type in the sampled layout."""
    cfg = env.cfg
    zones = np.asarray(env.semantic_zones, dtype=np.float32)
    hazards = np.asarray(env.hazards, dtype=np.float32)
    start = np.asarray(env.pos, dtype=np.float32)
    goal = np.asarray(env.goal, dtype=np.float32)

    zone_hazard = 0
    zone_zone = 0
    zone_start = 0
    zone_goal = 0

    for i, (zx, zy, zr) in enumerate(zones):
        center = np.asarray((zx, zy), dtype=np.float32)
        if np.linalg.norm(center - start) < float(zr + cfg.agent_radius + 0.3):
            zone_start += 1
        if np.linalg.norm(center - goal) < float(zr + cfg.goal_radius + 0.3):
            zone_goal += 1
        for hx, hy, hr in hazards:
            if np.linalg.norm(center - np.asarray((hx, hy), dtype=np.float32)) < float(
                zr + hr + 0.3
            ):
                zone_hazard += 1
        for j in range(i):
            other = zones[j, :2]
            if np.linalg.norm(center - other) < float(zr + zones[j, 2] + 0.3):
                zone_zone += 1

    return {
        "zone_hazard": zone_hazard,
        "zone_zone": zone_zone,
        "zone_start": zone_start,
        "zone_goal": zone_goal,
    }


def _layout_digest(env: PointHazardEnv) -> str:
    """Hash the float32 layout state for a compact, exact golden snapshot."""
    layout = np.concatenate(
        (
            env.hazards.reshape(-1),
            env.pos,
            env.goal,
            env.semantic_zones.reshape(-1),
        )
    ).astype(np.float32, copy=False)
    return hashlib.sha256(layout.tobytes()).hexdigest()


def _sweep_layout_range(
    case_name: str,
    cfg: PointHazardConfig,
    seed_start: int,
    seed_stop: int,
) -> dict[str, int]:
    """Validate one disjoint seed range and return its overlap totals."""
    env = PointHazardEnv(cfg=cfg)
    totals = {"zone_hazard": 0, "zone_zone": 0, "zone_start": 0, "zone_goal": 0}

    for seed in range(seed_start, seed_stop):
        _, info = env.reset(seed=seed)
        assert info["layout_valid"] is True, (case_name, seed, info)
        assert info["semantic_zones"].shape == (cfg.n_semantic_zones, 3)

        valid, reasons = zone_layout_valid(
            env.hazards,
            env.semantic_zones,
            env.pos,
            env.goal,
            cfg,
        )
        assert valid, (case_name, seed, reasons)

        counts = _overlap_counts(env)
        for key, count in counts.items():
            totals[key] += count
            assert count == 0, (case_name, seed, key, count)
    return totals


@pytest.mark.parametrize("case_name,cfg", _layout_cases())
def test_layout_invariants_over_fixed_seed_sweep(case_name: str, cfg: PointHazardConfig) -> None:
    """No sampled semantic zone may overlap hazards, zones, start, or goal."""
    # The three-zone corridor case takes hours when swept serially.  Keep small
    # developer runs simple, but shard the formal gate into disjoint processes.
    # Each worker still exercises the public reset path over exactly the same
    # seeds and assertions; only execution order changes.
    worker_count = 1
    if SEED_COUNT >= 1000:
        worker_count = max(1, min(MAX_SWEEP_WORKERS, SEED_COUNT))

    ranges = []
    for worker_index in range(worker_count):
        start = SEED_COUNT * worker_index // worker_count
        stop = SEED_COUNT * (worker_index + 1) // worker_count
        ranges.append((start, stop))

    if worker_count == 1:
        partial_totals = [_sweep_layout_range(case_name, cfg, *ranges[0])]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(_sweep_layout_range, case_name, cfg, start, stop)
                for start, stop in ranges
            ]
            partial_totals = [future.result() for future in futures]

    totals = {"zone_hazard": 0, "zone_zone": 0, "zone_start": 0, "zone_goal": 0}
    for partial in partial_totals:
        for key, count in partial.items():
            totals[key] += count

    assert totals == {
        "zone_hazard": 0,
        "zone_zone": 0,
        "zone_start": 0,
        "zone_goal": 0,
    }


# These are exact float32 layout snapshots for the post-B01 sampler.  A hash
# keeps the regression fixture readable while still detecting any bit-level
# change in hazards, start/goal, or semantic-zone geometry.
GOLDEN_LAYOUT_SHA256 = {
    "single_implicit_corridor_on": (
        "74c2c519535bf31de24435efcf9ba753808fde8906df78234ada5b614b8b6036",
        "cdde69999f56cbb7fb9ff63773ada72841e0201e8897b559b8275af8e8847398",
        "de8b4f6cb7c8a343a26f6f3d5dbef597b203863abe605b08aad862241917ae86",
        "ebc39836ba473ba1c60d3a81758a311df9b773ce75858cf9a5b663f5239d9d71",
        "ce45041f34f723da7b58e6663c803bf194d3f507e41a58b44f69290295ec3a3c",
    ),
    "single_implicit_corridor_off": (
        "6a144586471b11aec6474acf0d1bad72fe7411b9dee8fce9f1f1a36ff383db38",
        "f9d8b6b01925831228497c8bb381692de831ac247ea2b0b099060ef23e9da13a",
        "0c3d65381b342cf4e3d301daca8e296c63adc3f446b421951f091ac76f8f96f1",
        "f2482dfce0607d553d11ac323ac63a07196a1c8f9fa31da88792964ae9e68de2",
        "f28e43640457f348b52b4fbc95cbd679f559b1dc1dbdecfd60b3c8112351d3f5",
    ),
    "three_hetero_corridor_on": (
        "64d14be639be919246216097113b40a7a2ea0eb5c20678a3b0c6a32ba3652d8e",
        "6763a6d2079d1ab3661d6a460d34dbd9de75e25c9dadafeae3388108e7b0600c",
        "5066320b92cb87036ef7460cd32015e6c60d2a481dc7525dfc059d2b8aff60f0",
        "3b5f8f26b1e1774c6392b9cb89cbfa8cac912e1e0cca0677e6e9c656f2826828",
        "b8a0e9615385e206dca4eeb54d98ab35979562c5a6aeb9f2b61dd642fdfb3c57",
    ),
    "three_hetero_corridor_off": (
        "896239d876c9d35918e7f9c4eac05c7170434d32f91386fea96ffe73b9916567",
        "c55d7ee22f7e740cf01950c5f909f9566e8d6a2be1b7d5014250d358672eb927",
        "a29d3457dccb7dead0308639bc01163466a26788c737e996804d054046bb63ee",
        "a833f485be14213242db3755b8b5efe1aab38582403578f3602d9425cf87471a",
        "ec8228f778795a79e7819dca4ad17e2e3bf3b5073aee6549a2de36570e2292e6",
    ),
}


@pytest.mark.parametrize("case_name,cfg", _layout_cases())
def test_golden_layout_snapshots(case_name: str, cfg: PointHazardConfig) -> None:
    env = PointHazardEnv(cfg=cfg)
    actual = []
    for seed in GOLDEN_SEEDS:
        env.reset(seed=seed)
        actual.append(_layout_digest(env))
    assert tuple(actual) == GOLDEN_LAYOUT_SHA256[case_name]


@pytest.mark.parametrize("seed", FALLBACK_REGRESSION_SEEDS)
def test_checked_fallback_closes_known_layout_liveness_failures(seed: int) -> None:
    """Known exhausted layouts must return valid zones, never an unchecked one."""
    cfg = PointHazardConfig(
        n_semantic_zones=3,
        semantic_on_corridor=True,
        semantic_styles=("water", "mud", "grass"),
    )
    env = PointHazardEnv(cfg=cfg)

    _, info = env.reset(seed=seed)

    valid, reasons = zone_layout_valid(
        env.hazards,
        env.semantic_zones,
        env.pos,
        env.goal,
        cfg,
    )
    assert info["layout_valid"] is True
    assert valid, reasons
    assert _overlap_counts(env) == {
        "zone_hazard": 0,
        "zone_zone": 0,
        "zone_start": 0,
        "zone_goal": 0,
    }
    # Both fixtures exhausted the normal deterministic child streams before
    # the checked final-attempt fallback was introduced.  Keeping this
    # assertion ensures an earlier RNG path is not silently changed.
    assert info["resample_count"] == cfg.max_layout_resamples
