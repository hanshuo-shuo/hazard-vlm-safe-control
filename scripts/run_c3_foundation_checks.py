#!/usr/bin/env python3
"""Provider-free spatial truth checks and optional actual MuJoCo camera QA."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3_safe.artifacts import finish_run, new_run, write_json, run_cli
from c3_safe.costs import Capability, Rules, relation_cost, semantic_contact
from c3_safe.geometry import GroundProjection, intersects_disk, measure_exposure, oracle_field


def font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def diagnostic_cases(output):
    projection = GroundProjection.orthographic(256, 256, (-2, 2, -2, 2))
    base = {"region_id": "zone_0", "terrain_class": "water", "center_xy": [0., 0.], "radius": .5}
    cross = [[-1.2, 0.], [1.2, 0.]]
    bypass = [[-1.2, 0.], [-.9, 1.], [.9, 1.], [1.2, 0.]]
    cases = [
        ("Cross / non-waterproof", base, cross, Capability(), Rules(), True, False),
        ("Bypass / same image", base, bypass, Capability(), Rules(), False, False),
        ("Cross / waterproof", base, cross, Capability(1, 0), Rules(), False, False),
        ("Cross / rule activated", base, cross, Capability(1, 0), Rules(1, 0, 0), True, False),
        ("Body overlaps / center out", base, [[.6,0.],[.6,0.]], Capability(), Rules(), True, False),
        ("Thin grazing contact", base, [[-.6,.695],[.6,.695]], Capability(), Rules(), True, False),
        ("Visual twin / region moved", {**base, "center_xy": [0., 1.2]}, cross, Capability(), Rules(), False, False),
        ("Mud / ordinary mobility", {**base,"terrain_class":"mud"}, cross, Capability(), Rules(), True, False),
        ("Mud / rough mobility", {**base,"terrain_class":"mud"}, cross, Capability(0,1), Rules(), False, False),
        ("Fragile / no rule", {**base,"terrain_class":"fragile"}, cross, Capability(), Rules(), False, False),
        ("Fragile / protect rule", {**base,"terrain_class":"fragile"}, cross, Capability(), Rules(0,0,1), True, False),
        ("Occluded / explicit unknown", base, cross, Capability(), Rules(), True, True),
    ]
    board = Image.new("RGB", (4 * 300, 3 * 340), "#f4f7fb")
    records = []
    for i, (name, region, motion, cap, rules, expected, occluded) in enumerate(cases):
        field = oracle_field([region], projection)
        exposure, footprint = measure_exposure(field, motion, .2, projection,
            visibility=np.zeros((256,256), dtype=bool) if occluded else None)
        contact = semantic_contact(motion, .2, [region], cap, rules)
        costs = relation_cost(exposure, cap, rules)
        assert contact == expected
        record = {"name": name, "fixture_only": True, "regions": [region], "motion": motion,
                  "capability": asdict(cap), "rules": asdict(rules), "robot_radius": .2,
                  "binary_violation": contact, "expected_violation": expected,
                  "exposure": exposure.to_dict(), "soft_cost": costs}
        records.append(record)
        rgb = np.full((256,256,3), 249, dtype=np.uint8)
        rgb[np.any(field > 0, axis=-1)] = [130,195,232]
        rgb[footprint] = (.5 * rgb[footprint] + .5 * np.array([240,120,110])).astype(np.uint8)
        image = Image.fromarray(rgb)
        d = ImageDraw.Draw(image)
        uv, _ = projection.project(motion)
        if len(uv) > 1:
            d.line([tuple(x) for x in uv], fill="#842b2b", width=3)
        tile = Image.new("RGB", (300,340), "#f4f7fb")
        tile.paste(image, (20,30))
        d = ImageDraw.Draw(tile)
        d.text((18,7), name, font=font(15), fill="#172b42")
        d.text((20,291), f"Contact violation: {contact}", font=font(15), fill="#842b2b" if contact else "#12633e")
        soft = costs["semantic_cost"]
        d.text((20,314), "Exposure: UNKNOWN" if soft is None else f"q95 semantic cost: {soft:.3f}", font=font(14), fill="#334155")
        board.paste(tile, ((i%4)*300, (i//4)*340))
    board.save(output / "motion_cases.png")
    write_json(output / "CASES.json", records)
    return {"n": len(records), "binary_checks_passed": len(records),
            "soft_cost_zero_with_contact": [r["name"] for r in records if r["binary_violation"] and r["soft_cost"]["semantic_cost"] == 0],
            "note": "q95 severity cannot substitute for binary swept-contact truth"}


def raster_checks(count):
    rng = np.random.default_rng(928)
    projection = GroundProjection.orthographic(128,128,(-2,2,-2,2))
    agreements = 0
    false_positive = negatives = 0
    valid_count = 0
    fractions = np.linspace(0,1,3001)
    for _ in range(count):
        motion = rng.uniform(-1.6, 1.6, (2,2))
        center = rng.uniform(-1., 1., 2)
        radius, body = rng.uniform(.25,.65), rng.uniform(.10,.25)
        region = {"region_id":"r", "terrain_class":"water", "center_xy":center.tolist(), "radius":radius}
        field = oracle_field([region], projection)
        exposure, footprint = measure_exposure(field, motion, body, projection)
        assert exposure.validity == "valid"
        valid_count += 1
        dense_path = motion[0] + fractions[:,None] * (motion[1]-motion[0])
        oracle = bool(np.linalg.norm(dense_path - center, axis=1).min() <= radius + body)
        raster = bool(field[...,0][footprint].max() > 0)
        agreements += raster == oracle
        negatives += not oracle
        false_positive += raster and not oracle
    result = {"n":count, "valid":valid_count, "agreement":agreements/count,
              "negative_transitions":negatives, "false_positive_rate":false_positive/max(1,negatives),
              "reference":"independent 3001-point path sampling", "resolution":[128,128]}
    result["passed"] = result["agreement"] >= .98 and result["false_positive_rate"] <= .02
    return result


def native_checks(output, count):
    import mujoco
    from envs.safety_gym_goal_adapter import SemanticSafetyPointGoalAdapter, SafetyGymGoalAdapter
    from evaluation.semantic_evaluator import evaluate_scene_manifest
    adapter = SemanticSafetyPointGoalAdapter()
    reference = SafetyGymGoalAdapter(camera_name="fixednear")
    try:
        observation, _ = adapter.reset(seed=200)
        reference.reset(seed=200)
        frame = adapter.render_public_rgb()
        Image.fromarray(frame).save(output / "native_semantic_rgb.png")
        task = adapter._env.unwrapped.task
        native_projection = adapter.ground_projection(frame.shape)
        source = adapter.scene_manifest()
        renderer = mujoco.Renderer(task.model, height=frame.shape[0], width=frame.shape[1])
        errors = []
        occluded = 0
        rng = np.random.default_rng(903)
        try:
            for _ in range(count):
                xy = rng.uniform(-1.3,1.3,2)
                # Ground-level thin marker rendered by MuJoCo, not by our
                # homography. Compare its raster centroid with our projection.
                renderer.update_scene(task.data, camera=adapter.camera_name)
                scene = renderer.scene
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([.027,.027,.001]), np.array([*xy,.003]),
                    np.eye(3).reshape(-1), np.array([1.,0.,1.,1.],dtype=np.float32))
                geom.emission = 1.
                scene.ngeom += 1
                image = renderer.render()
                mask = (image[...,0] > 170) & (image[...,2] > 170) & (image[...,1] < 90)
                ys, xs = np.nonzero(mask)
                uv, valid = native_projection.project([xy])
                if len(xs) == 0 or not valid[0]:
                    occluded += 1
                    continue
                errors.append(float(np.linalg.norm(np.array([xs.mean()+.5, ys.mean()+.5]) - uv[0])))
        finally:
            # MuJoCo 2.3.3 predates Renderer.close(). Release its owned contexts.
            if hasattr(renderer, "close"):
                renderer.close()
            else:
                renderer._mjr_context.free()
                renderer._gl_context.free()
        unchanged = True
        pose_samples = []
        for _ in range(12):
            action = rng.uniform(-1,1,2).astype(np.float32)
            actual, expected = adapter.step(action), reference.step(action)
            unchanged &= np.array_equal(actual[0], expected[0]) and actual[1:5] == expected[1:5]
            pose_samples.append(len(adapter.evaluator_context().trajectory[-1]["motion_samples"]))
        context = adapter.evaluator_context()
        independent = evaluate_scene_manifest(context.trajectory, context.scene_manifest, adapter.capability)
        agreement = bool(context.semantic_violations == independent.per_step_violation)
        result = {"backend":"native SafetyPointGoal1-v0", "camera":adapter.camera_name,
                  "calibration":native_projection.to_dict(), "calibration_hash":native_projection.sha256,
                  "marker_attempts":count, "visible_markers":len(errors), "occluded_or_outside":occluded,
                  "max_error_pixels":max(errors) if errors else None,
                  "mean_error_pixels":float(np.mean(errors)) if errors else None,
                  "within_2_pixels_rate":sum(e <= 2 for e in errors)/max(1,len(errors)),
                  "native_transition_reward_cost_flags_unchanged":bool(unchanged),
                  "semantic_step_evaluator_agree":agreement,
                  "substep_samples":pose_samples, "robot_radius":adapter.robot_radius,
                  "layout_valid":source["semantic_layout_valid"],
                  "render_contract":"ground annotation over native RGB; not physically simulated water"}
        result["passed"] = bool(len(errors) >= count//2 and result["within_2_pixels_rate"] >= .98 and unchanged and agreement)
        return result
    finally:
        adapter.close()
        reference.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transitions", type=int, default=400)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--native-markers", type=int, default=100)
    args = parser.parse_args()
    output = new_run(args.output)
    summary = {"status":"DEVELOPMENT_FOUNDATION_CHECKS", "cases":diagnostic_cases(output),
               "raster":raster_checks(args.transitions), "native":"NOT_RUN"}
    if args.native:
        summary["native"] = native_checks(output, args.native_markers)
    summary["full_c3_gate0"] = "NOT_CLAIMED: teacher provenance and full multi-terrain data still pending"
    finish_run(output, summary, source_paths=["scripts/run_c3_foundation_checks.py"])
    print(json.dumps(summary, indent=2))
    if not summary["raster"]["passed"] or (args.native and not summary["native"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    run_cli(main)
