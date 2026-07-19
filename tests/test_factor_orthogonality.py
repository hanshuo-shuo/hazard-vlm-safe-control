"""Byte-level snapshots for the three B06 prompt/render factors."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

import subgoal_pivot_hazard as harness
from env_pointhazard import PointHazardConfig, PointHazardEnv
from hazard_renderer import HazardRenderer


SEED = 17


def _scene(appearance: str) -> tuple[PointHazardEnv, bytes]:
    """Reset the same layout and return raw RGB bytes for one appearance."""
    cfg = PointHazardConfig(n_semantic_zones=1, max_episode_steps=20)
    env = PointHazardEnv(cfg=cfg)
    env.reset(seed=SEED)
    renderer = HazardRenderer.from_env(
        env,
        semantic_style=harness.SEMANTIC_SPECS[appearance].renderer_style,
    )
    env.attach_renderer(renderer)
    image = np.asarray(env.render(), dtype=np.uint8)
    return env, image.tobytes()


def _png_bytes(raw_rgb: bytes, size: int = 480) -> bytes:
    """Encode a raw RGB snapshot so the test also covers image serialization."""
    image = Image.frombytes("RGB", (size, size), raw_rgb)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _assert_same_scene(left: PointHazardEnv, right: PointHazardEnv) -> None:
    np.testing.assert_array_equal(left.hazards, right.hazards)
    np.testing.assert_array_equal(left.semantic_zones, right.semantic_zones)
    np.testing.assert_array_equal(left.pos, right.pos)
    np.testing.assert_array_equal(left.goal, right.goal)


def test_zone_semantics_changes_pixels_only() -> None:
    explicit_env, explicit_image = _scene("explicit")
    implicit_env, implicit_image = _scene("implicit")
    _assert_same_scene(explicit_env, implicit_env)

    explicit_prompt = harness.build_subgoal_prompt_segments(
        8, semantic=True, prompt_level="L1", capability="wheeled_non_waterproof"
    )
    implicit_prompt = harness.build_subgoal_prompt_segments(
        8, semantic=True, prompt_level="L1", capability="wheeled_non_waterproof"
    )

    assert explicit_prompt == implicit_prompt
    assert explicit_image != implicit_image
    assert hashlib.sha256(_png_bytes(explicit_image)).digest() != hashlib.sha256(
        _png_bytes(implicit_image)
    ).digest()


@pytest.mark.parametrize(
    "builder",
    [harness.build_direct_prompt_segments, harness.build_subgoal_prompt_segments],
)
def test_prompt_level_changes_privilege_segment_only(builder) -> None:
    _, image_l0 = _scene("explicit")
    _, image_l1 = _scene("explicit")
    assert image_l0 == image_l1

    l0 = builder(
        8, semantic=True, prompt_level="L0", capability="wheeled_non_waterproof"
    )
    l1 = builder(
        8, semantic=True, prompt_level="L1", capability="wheeled_non_waterproof"
    )

    assert l0.common == l1.common
    assert l0.task == l1.task
    assert l0.capability == l1.capability
    assert l0.output == l1.output
    assert l0.privilege != l1.privilege
    assert l0.render().encode("utf-8") != l1.render().encode("utf-8")


@pytest.mark.parametrize(
    "builder",
    [harness.build_direct_prompt_segments, harness.build_subgoal_prompt_segments],
)
def test_capability_changes_capability_segment_only(builder) -> None:
    _, wheeled_image = _scene("implicit")
    _, amphibious_image = _scene("implicit")
    assert wheeled_image == amphibious_image

    wheeled = builder(
        8, semantic=True, prompt_level="L1", capability="wheeled_non_waterproof"
    )
    amphibious = builder(
        8, semantic=True, prompt_level="L1", capability="amphibious"
    )

    assert wheeled.common == amphibious.common
    assert wheeled.task == amphibious.task
    assert wheeled.privilege == amphibious.privilege
    assert wheeled.output == amphibious.output
    assert wheeled.capability != amphibious.capability
    assert "id: wheeled_non_waterproof" in wheeled.capability
    assert "id: amphibious" in amphibious.capability


def test_cli_exposes_capability_as_independent_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["subgoal_pivot_hazard.py", "--zone_semantics", "implicit", "--prompt_level", "L0", "--capability", "amphibious"],
    )
    args = harness.parse_args()
    assert args.zone_semantics == "implicit"
    assert args.prompt_level == "L0"
    assert args.capability == "amphibious"

