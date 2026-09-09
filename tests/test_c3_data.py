"""Split and provenance checks prevent the audited counterfactual leakage."""
import hashlib

import pytest

from c3_safe.artifacts import file_sha
from c3_safe.data import audit_splits, counterfactual_twin, load_spec, teacher_request, validate_teacher_record
from env_pointhazard import PointHazardEnv
import numpy as np


def row():
    return {"split":"train", "geometry_seed":1000,"appearance_id":"water-blue-v1",
            "capability_vector":[0,0],"rule_vector":[0,0,0],"geometry_hash":"g0",
            "image_sha256":"image0","twin_group_id":"transition0","state_hash":"s0",
            "next_state_hash":"s1","action_hash":"a0","field_sha256":"field0",
            "swept_footprint_hash":"swept0","motion_samples":[[-1,0],[1,0]],"robot_radius":.2,
            "oracle_regions":[{"region_id":"r","terrain_class":"water","center_xy":[0,0],"radius":.5}],
            "exposure":{"validity":"valid","values":[1.,0.,0.]}}


def test_counterfactual_relabel_cannot_include_held_out_capability_or_rule():
    spec = load_spec()
    original = row()
    with pytest.raises(ValueError,match="capability_vector"):
        counterfactual_twin(original,[1,1],[0,0,0],spec)
    with pytest.raises(ValueError,match="rule_vector"):
        counterfactual_twin(original,[0,0],[1,1,0],spec)
    twin = counterfactual_twin(original,[1,0],[0,0,0],spec)
    assert twin["oracle_contact_violation"] is False
    assert twin["oracle_soft_cost"]["semantic_cost"] == 0
    assert original == row()
    for key in ("state_hash","next_state_hash","action_hash","field_sha256","swept_footprint_hash","image_sha256","motion_samples"):
        assert twin[key] == original[key]


@pytest.mark.parametrize("key",["image_sha256","geometry_hash","twin_group_id"])
def test_cross_split_derived_data_reuse_is_rejected(key):
    a, b = row(),row()
    b.update(split="validation",geometry_seed=2000,appearance_id="water-teal-v1",
             image_sha256="image1",geometry_hash="g1",twin_group_id="transition1")
    b[key] = a[key]
    with pytest.raises(ValueError,match=key):
        audit_splits([a,b],load_spec())


def test_teacher_requests_have_only_image_and_constant_action_free_prompt(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"diagnostic image bytes")
    request = teacher_request(image,image_id="i0")
    assert request["prompt_sha256"] == hashlib.sha256(request["prompt"].encode()).hexdigest()
    assert not {"action","capability","rule","trajectory","oracle_regions"}.intersection(request)
    with pytest.raises(ValueError,match="lacks real model"):
        validate_teacher_record({},request)


def test_teacher_provenance_uses_exact_raw_bytes_and_rejects_field_tamper(tmp_path):
    image, field = tmp_path/"image.png", tmp_path/"field.npz"
    image.write_bytes(b"image")
    field.write_bytes(b"field")
    request = teacher_request(image,image_id="i0")
    raw = '{"regions":[]}\n'
    record = {"teacher_model":"fixture-only","teacher_revision":"fixture-v1",
              "prompt_sha256":request["prompt_sha256"],"image_sha256":request["image_sha256"],
              "raw_response":raw,"response_sha256":hashlib.sha256(raw.encode()).hexdigest(),
              "field_path":str(field),"field_sha256":file_sha(field)}
    assert validate_teacher_record(record,request)
    with pytest.raises(ValueError,match="forbidden"):
        validate_teacher_record({**record,"action":[0,0]},request)
    field.write_bytes(b"tamper")
    with pytest.raises(ValueError,match="field hash"):
        validate_teacher_record(record,request)


def test_fallback_respects_configured_body_radius():
    point = PointHazardEnv._first_valid_grid_zone(
        np.array([[.9,0.],[1.2,0.]],dtype=np.float32), .2, np.empty((0,3)),
        np.array([0.,0.]),np.array([4.,0.]),[],agent_radius=.6,goal_radius=.5,
    )
    np.testing.assert_allclose(point,[1.2,0.])


def test_shared_global_router_completes_the_local_mpc_trap(tmp_path):
    from scripts.collect_c3_oracle_data import collect_episode
    spec = load_spec()
    episode, transitions, requests = collect_episode(
        tmp_path, split="train", seed=1005, arm="oracle_geometry_mpc", spec=spec,
    )
    assert episode["reached_goal"] and episode["STC"]
    assert not any(episode["native_costs"]) and not any(episode["semantic_violations"])
    assert episode["steps"] < spec["collection"]["max_episode_steps"]
    assert transitions and requests
    assert all(r["label_source"] == "SIMULATOR_ORACLE_NOT_VLM" for r in transitions)
