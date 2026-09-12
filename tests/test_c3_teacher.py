"""Teacher transport integrity, malformed response handling and unknown masks."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from c3_safe.data import teacher_request
from c3_safe.geometry import GroundProjection, measure_exposure
from c3_safe.teacher_io import file_hash, load_bundle, parse_response, rasterize_response, save_json, validate_request
from scripts.prepare_c3_teacher_bundle import prepare


def answer():
    return {"regions": [{"channel": "water_ingress_demand", "polygon_norm_xy": [[.25,.25],[.75,.25],[.75,.75],[.25,.75]], "confidence": .2}],
            "unknown_regions": []}


def test_polygon_uses_pixel_centers_and_keeps_confidence_separate():
    data = parse_response(json.dumps(answer()))
    field, known = rasterize_response(data, 16, 16)
    assert field[..., 0].sum() == 64
    assert np.all(field[4:12, 4:12, 0] == 1.)
    assert not field[..., 1:].any() and known.all()


@pytest.mark.parametrize("raw", [
    '{"regions":[],"regions":[],"unknown_regions":[]}',
    '{"regions":[],"unknown_regions":[],"action":[0,1]}',
    '{"regions":[],"unknown_regions":["somewhere"]}',
    '{"regions":[{"channel":"water_ingress_demand","polygon_norm_xy":[[0,0],[1,0],[1,1]],"confidence":NaN}],"unknown_regions":[]}',
    'Commentary {"regions":[],"unknown_regions":[]}',
])
def test_malformed_or_nonspatial_outputs_cannot_become_safe_labels(raw):
    with pytest.raises(ValueError):
        parse_response(raw)


@pytest.mark.parametrize("vertices", [
    [[0,0],[1,0],[1.01,1]], [[0,0],[.5,.5],[1,1]],
    [[0,0],[1,1],[1,0],[0,1]], [["0",0],[1,0],[1,1]],
])
def test_invalid_polygons_are_rejected(vertices):
    data = answer()
    data["regions"][0]["polygon_norm_xy"] = vertices
    with pytest.raises(ValueError):
        parse_response(json.dumps(data))


def test_teacher_unknown_footprint_is_not_zero_exposure():
    data = answer()
    data["unknown_regions"] = [{"polygon_norm_xy": [[.4,.4],[.6,.4],[.6,.6],[.4,.6]], "reason": "occluded"}]
    field, known = rasterize_response(parse_response(json.dumps(data)), 32, 32)
    projection = GroundProjection.orthographic(32, 32, (-1,1,-1,1))
    exposure, _ = measure_exposure(field, [[0,0],[0,0]], .15, projection, visibility=known)
    assert exposure.validity == "unknown_projection" and exposure.values is None


def test_unknown_region_confidence_is_metadata_and_remains_unknown():
    data = {"regions": [], "unknown_regions": [{"polygon_norm_xy": [[0,0],[1,0],[1,1],[0,1]], "confidence": .9}]}
    field, known = rasterize_response(parse_response(json.dumps(data)), 8, 8)
    assert not known.any() and not field.any()
    data["unknown_regions"][0]["confidence"] = 1.5
    with pytest.raises(ValueError, match="confidence"):
        parse_response(json.dumps(data))


def test_bundle_exports_only_hashed_rgb_and_rejects_oracle_even_if_manifest_lists_it(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "bundle"
    dataset.mkdir()
    image = dataset / "oracle_arm_name.png"
    Image.new("RGB", (16,16), "blue").save(image)
    request = teacher_request(image, image_id=file_hash(image))
    request["image_path"] = image.name
    (dataset / "TEACHER_REQUESTS_NOT_SUBMITTED.jsonl").write_text(json.dumps(request) + "\n")
    save_json(dataset / "MANIFEST.json", {"fixture": True})
    manifest = prepare(dataset, output)
    verified, rows = load_bundle(output)
    assert verified == manifest and len(rows) == 1
    assert rows[0]["image_path"] == f"images/{file_hash(image)}.png"
    assert rows[0]["prompt_sha256"] == hashlib.sha256(request["prompt"].encode()).hexdigest()
    extra = output / "oracle_truth.json"
    extra.write_text('{"unsafe": true}')
    manifest["files"][extra.name] = file_hash(extra)
    save_json(output / "BUNDLE_MANIFEST.json", manifest)
    with pytest.raises(ValueError, match="only RGB"):
        load_bundle(output)


def test_request_extra_cards_and_path_escape_are_rejected(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (16,16)).save(image)
    request = teacher_request(image, image_id=file_hash(image))
    request["image_path"] = image.name
    validate_request(request, tmp_path)
    with pytest.raises(ValueError, match="only image"):
        validate_request({**request, "capability": [1,1]}, tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        validate_request({**request, "image_path": "../image.png"}, tmp_path)
