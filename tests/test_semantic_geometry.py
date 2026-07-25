import hashlib

import pytest

from evaluation.semantic_geometry import (
    CoordinateFrame,
    GeometrySource,
    ImageCoordinateTransform,
    SemanticGeometryPayload,
    SemanticRegion,
    adapt_regions,
    detector_box_records,
)


def transform(**kwargs):
    values = dict(
        image_width=201,
        image_height=101,
        world_x_min=-5,
        world_x_max=5,
        world_y_min=-2,
        world_y_max=2,
    )
    values.update(kwargs)
    return ImageCoordinateTransform(**values)


def test_provider_normalized_pixel_and_world_round_trip():
    item = transform()
    assert item.provider_to_pixel((500, 500)) == pytest.approx((100, 50))
    assert item.normalized_to_pixel((0.5, 0.5)) == pytest.approx((100, 50))
    for world in ((-5, -2), (0, 0), (5, 2), (1.25, -0.75)):
        assert item.pixel_to_world(item.world_to_pixel(world)) == pytest.approx(world)


@pytest.mark.parametrize("point", [(-1, 0), (1001, 2), (0, float("nan"))])
def test_provider_out_of_bounds_rejected(point):
    with pytest.raises(ValueError):
        transform().provider_to_pixel(point)


def test_letterbox_padding_rejected_and_content_round_trip():
    item = transform(
        image_width=200,
        image_height=200,
        content_top=50,
        content_width=200,
        content_height=100,
    )
    assert item.pixel_to_world(item.world_to_pixel((0, 0))) == pytest.approx((0, 0))
    with pytest.raises(ValueError, match="letterbox"):
        item.pixel_to_world((20, 10))


@pytest.mark.parametrize(
    "kind,geometry",
    [
        ("disk", {"center_xy": [0, 0], "radius": -1}),
        ("aabb", {"min_xy": [1, 0], "max_xy": [0, 1]}),
        ("polygon", {"points": [[0, 0], [0, 0], [1, 1]]}),
        ("mask_reference", {"mask_sha256": "bad"}),
    ],
)
def test_malformed_geometry_rejected(kind, geometry):
    with pytest.raises(ValueError):
        SemanticRegion(
            region_id="r",
            semantic_class="water",
            geometry_type=kind,
            geometry=geometry,
            coordinate_frame="world_xy",
            confidence=1,
            source="fixture",
        )


def test_empty_and_deterministic_region_order_and_hash():
    item = transform()
    empty = adapt_regions(
        source="none",
        records=[],
        request_id="q",
        scene_id="s",
        image_sha256=None,
        coordinate_frame="world_xy",
        transform=item,
    )
    assert empty.regions == ()
    assert empty.parser_status.value == "empty"
    records = [
        {"region_id": "z", "class": "water", "geometry": {"center_xy": [1, 1], "radius": 1}},
        {"region_id": "a", "class": "mud", "geometry": {"center_xy": [0, 0], "radius": 1}},
    ]
    payload = adapt_regions(
        source="fixture",
        records=records,
        request_id="q",
        scene_id="s",
        image_sha256=hashlib.sha256(b"x").hexdigest(),
        coordinate_frame="world_xy",
        transform=item,
    )
    assert [region.region_id for region in payload.regions] == ["a", "z"]
    rebuilt = SemanticGeometryPayload(**{
        "request_id": payload.request_id,
        "scene_id": payload.scene_id,
        "image_sha256": payload.image_sha256,
        "coordinate_frame_metadata": payload.coordinate_frame_metadata,
        "image_to_world_transform_version": payload.image_to_world_transform_version,
        "regions": tuple(reversed(payload.regions)),
        "parser_status": payload.parser_status,
        "adapter_status": payload.adapter_status,
        "fallback_status": payload.fallback_status,
        "provenance": payload.provenance,
    })
    assert rebuilt.sha256 == payload.sha256


def test_oracle_requires_explicit_arm_and_forbidden_truth_is_rejected():
    kwargs = dict(
        source=GeometrySource.ORACLE,
        records=[{"class": "water", "geometry": {"center_xy": [0, 0], "radius": 1}}],
        request_id="q",
        scene_id="s",
        image_sha256=None,
        coordinate_frame=CoordinateFrame.WORLD,
        transform=transform(),
    )
    with pytest.raises(PermissionError):
        adapt_regions(**kwargs)
    kwargs["allow_oracle"] = True
    kwargs["records"][0]["ground_truth"] = True
    with pytest.raises(PermissionError):
        adapt_regions(**kwargs)


def test_multiple_detector_boxes_and_confidence_tie_order():
    boxes = detector_box_records(
        [
            {"region_id": "b", "class": "water", "min_xy": [0, 0], "max_xy": [500, 500], "confidence": 0.5},
            {"region_id": "a", "class": "mud", "min_xy": [500, 500], "max_xy": [1000, 1000], "confidence": 0.5},
        ],
        transform=transform(),
        input_frame=CoordinateFrame.PROVIDER_NATIVE,
    )
    payload = adapt_regions(
        source="detector",
        records=boxes,
        request_id="q",
        scene_id="s",
        image_sha256=None,
        coordinate_frame="image_pixel_xy",
        transform=transform(),
    )
    assert [item.region_id for item in payload.regions] == ["a", "b"]
    assert all(item.confidence == 0.5 for item in payload.regions)
    assert all(item.adapter_version for item in payload.regions)
