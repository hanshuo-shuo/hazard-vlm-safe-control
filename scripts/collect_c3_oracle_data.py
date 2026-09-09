#!/usr/bin/env python3
"""Collect native PointHazard transitions, oracle labels and split-safe twins.

The two arms share the same CEM-MPC implementation and configuration. The oracle
arm receives evaluator terrain disks explicitly as an upper-bound diagnostic;
the blind arm receives native physical geometry only. Neither arm is C3-Safe or
a learned SAC baseline. Teacher requests are prepared but never submitted.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3_safe.artifacts import digest, file_sha, finish_run, new_run, write_json, run_cli
from c3_safe.costs import Capability, Rules, active_region, relation_cost
from c3_safe.data import audit_splits, counterfactual_twin, load_spec, teacher_request
from c3_safe.geometry import measure_exposure, oracle_field
from c3_safe.point_adapter import C3PointHazardAdapter
from c3_safe.oracle_control import RoutedMPC
from env_pointhazard import PointHazardConfig
from evaluation.outcomes import OUTCOME_SCHEMA_VERSION, context_outcome
from mpc_expert import MPCConfig


PALETTES = {
    "water-blue-v1": ((90,170,205,95), (40,110,150), (240,250,255,175)),
    "water-teal-v1": ((65,170,150,110), (30,100,95), (200,245,235,185)),
    "water-slate-v1": ((120,145,170,120), (65,85,110), (220,235,245,200)),
}


def collect_episode(output, *, split, seed, arm, spec):
    config = spec["collection"]
    cfg = PointHazardConfig(n_hazards=config["n_hazards"], n_semantic_zones=1,
        semantic_styles=("water",), semantic_terrain_classes=("water",),
        max_episode_steps=config["max_episode_steps"], render_size=config["image_size"])
    cap = Capability(*spec["capability_splits"][split][0])
    rules = Rules(*spec["rule_splits"][split][0])
    appearance = spec["appearance_splits"][split][0]
    adapter = C3PointHazardAdapter(cfg, capability=cap, rules=rules)
    rows, requests = [], []
    try:
        obs, _ = adapter.reset(seed=seed)
        renderer = adapter._env._renderer
        renderer.water_fill, renderer.water_outline, renderer.water_ripple = PALETTES[appearance]
        scene = adapter.scene_manifest()
        regions = scene["semantic_terrain"]
        geometry_hash = digest({k:scene[k] for k in ("start", "goal", "physical_hazards", "semantic_terrain", "agent_radius")})
        projection = adapter.ground_projection()
        field = oracle_field(regions, projection)
        field_file = output / "fields" / f"{geometry_hash}.npz"
        field_file.parent.mkdir(parents=True, exist_ok=True)
        if not field_file.exists():
            np.savez_compressed(field_file, requirement_field=field)
        field_hash = file_sha(field_file)
        mpc_cfg = MPCConfig(horizon=config["mpc_horizon"], n_samples=config["mpc_samples"],
                            n_iters=config["mpc_iterations"])
        physical = np.asarray(obs[6:6+3*cfg.n_hazards]).reshape(-1,3)
        terrain = np.array([[*r["center_xy"],r["radius"]] for r in regions if active_region(r,cap,rules)],dtype=np.float32).reshape(-1,3)
        avoid_set = np.concatenate((physical,terrain)) if arm == "oracle_geometry_mpc" else physical
        controller = RoutedMPC(cfg,mpc_cfg,obs,avoid_set,seed=seed+90000,
            grid_res=config["global_grid_resolution"],route_margin=config["global_route_margin"])
        episode_id = f"{split}-{seed}-{arm}"
        initial_image_hash = None
        sampled_contact_zero_q95 = 0
        for t in range(cfg.max_episode_steps):
            sample = t % config["image_stride"] == 0
            image = adapter.render_public_rgb() if sample else None
            before = obs.copy()
            action = controller.act(before)
            obs, reward, cost, terminated, truncated, _ = adapter.step(action)
            if sample:
                png_path = output / "images" / f"{episode_id}-{t:04d}.png"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(png_path)
                image_sha = file_sha(png_path)
                initial_image_hash = initial_image_hash or image_sha
                motion = np.stack((before[:2],obs[:2])).tolist()
                exposure, footprint = measure_exposure(field,motion,cfg.agent_radius,projection,
                    quantile=spec["soft_exposure_quantile"])
                soft_cost = relation_cost(exposure,cap,rules)
                contact = bool(adapter._semantic_violations[-1])
                sampled_contact_zero_q95 += contact and soft_cost["semantic_cost"] == 0
                identity = {"state":before.tolist(),"action":action.tolist(),"next_state":obs.tolist(),
                            "image_sha256":image_sha,"field_sha256":field_hash,
                            "motion_samples":motion,"camera_calibration_hash":projection.sha256}
                transition_id = digest(identity)
                row = {"schema_version":"c3-spatial-transition-v1", "episode_id":episode_id,
                    "transition_id":transition_id, "twin_group_id":transition_id,"t":t,"split":split,
                    "geometry_seed":seed, "geometry_hash":geometry_hash,"appearance_id":appearance,
                    "image_path":str(png_path.relative_to(output)),"image_sha256":image_sha,
                    "state":before.tolist(),"state_hash":digest(before.tolist()),
                    "action":action.tolist(),"action_hash":digest(action.tolist()),
                    "next_state":obs.tolist(),"next_state_hash":digest(obs.tolist()),
                    "requirement_field_path":str(field_file.relative_to(output)),"field_sha256":field_hash,
                    "field_schema_version":"c3-spatial-field-v1","label_source":"SIMULATOR_ORACLE_NOT_VLM",
                    "camera_calibration":projection.to_dict(),"camera_calibration_hash":projection.sha256,
                    "motion_samples":motion,"robot_radius":cfg.agent_radius,
                    "swept_footprint_hash":hashlib.sha256(footprint.tobytes()).hexdigest(),
                    "exposure":exposure.to_dict(),"capability_vector":list(cap.vector),"rule_vector":list(rules.vector),
                    "native_physical_cost":cost,"native_reward":reward,
                    "terminated":terminated,"truncated":truncated,
                    "oracle_regions":regions,"oracle_contact_violation":contact,"oracle_soft_cost":soft_cost,
                    "collector":arm,"controller_config":asdict(mpc_cfg),
                    "router":config["router"],"route_margin":config["global_route_margin"],"intervention":"observed"}
                rows.append(row)
                request = teacher_request(png_path,image_id=image_sha)
                request["image_path"] = str(png_path.relative_to(output))
                requests.append(request)
            if terminated or truncated:
                break
        context = adapter.evaluator_context()
        outcome = context_outcome(context)
        episode = {"episode_id":episode_id,"split":split,"geometry_seed":seed,"appearance_id":appearance,
            "geometry_hash":geometry_hash,"initial_image_sha256":initial_image_hash,
            "router":config["router"],"global_path":[p.tolist() for p in controller.path],
            "capability_vector":cap.vector,"rule_vector":rules.vector,"collector":arm,
            "outcome_schema_version":OUTCOME_SCHEMA_VERSION, **outcome.to_dict(),
            "semantic_safe_success":outcome.semantic_safe_success,
            "native_cost_sum":sum(context.native_costs),"steps":len(context.actions),
            "native_rewards":list(context.rewards),"native_costs":list(context.native_costs),
            "semantic_violations":list(context.semantic_violations),"trajectory":list(context.trajectory),
            "actions":list(context.actions),"sampled_contact_zero_q95":int(sampled_contact_zero_q95)}
        # Store detached JSON values; the saved episode is sufficient to recount
        # completion and compare the paired controllers without rerunning them.
        write_json(output / "episodes" / f"{episode_id}.json", episode)
        return episode, rows, requests
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seeds-per-split",type=int,default=None,
                        help="Development subset; all selected seed IDs remain frozen in the manifest")
    args = parser.parse_args()
    spec = load_spec()
    if args.seeds_per_split is not None and args.seeds_per_split < 1:
        parser.error("seeds-per-split must be positive")
    output = new_run(args.output)
    episodes, rows, requests = [], [], []
    for split,seeds in spec["seed_splits"].items():
        selected = seeds if args.seeds_per_split is None else seeds[:args.seeds_per_split]
        for seed in selected:
            for arm in spec["collection"]["arms"]:
                episode,new_rows,new_requests = collect_episode(output,split=split,seed=seed,arm=arm,spec=spec)
                episodes.append(episode); rows.extend(new_rows); requests.extend(new_requests)
                print(json.dumps({"episode":episode["episode_id"],"success":episode["reached_goal"],
                                  "STC":episode["STC"],"steps":episode["steps"]}),flush=True)
    twins = []
    for row in rows:
        for cap in spec["capability_splits"][row["split"]]:
            for rules in spec["rule_splits"][row["split"]]:
                twin = counterfactual_twin(row,cap,rules,spec)
                twins.append({k:twin[k] for k in ("transition_id","twin_group_id","split",
                    "geometry_seed","geometry_hash","appearance_id","image_sha256",
                    "state_hash","next_state_hash","action_hash","field_sha256","swept_footprint_hash",
                    "capability_vector","rule_vector","oracle_contact_violation","oracle_soft_cost","intervention")})
    split_audit = audit_splits(rows + twins,spec)
    for name,records in (("TRANSITIONS.jsonl",rows),("COUNTERFACTUALS.jsonl",twins),
                         ("TEACHER_REQUESTS_NOT_SUBMITTED.jsonl",list({r["image_sha256"]:r for r in requests}.values()))):
        (output/name).write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records))
    by_arm = {}
    for arm in spec["collection"]["arms"]:
        group = [e for e in episodes if e["collector"] == arm]
        by_arm[arm] = {"n":len(group),"success":sum(e["reached_goal"] for e in group),
                      "STC":sum(e["STC"] for e in group),
                      "semantic_violation":sum(e["applicable_semantic_violation"] for e in group),
                      "native_safety_event":sum(e["physical_collision"] for e in group)}
    paired = {}
    for e in episodes:
        paired.setdefault((e["split"],e["geometry_seed"]),[]).append(e)
    paired_inputs = all(len(v)==2 and v[0]["geometry_hash"]==v[1]["geometry_hash"] and
                       v[0]["initial_image_sha256"]==v[1]["initial_image_sha256"] for v in paired.values())
    summary = {"status":"ORACLE_AND_BLIND_MPC_INFRASTRUCTURE_NOT_C3_POLICY_RESULT",
               "episodes":len(episodes),"sampled_transitions":len(rows),"counterfactuals":len(twins),
               "teacher_requests":len({r["image_sha256"] for r in requests}),"teacher_calls":0,
               "label_source":"SIMULATOR_ORACLE_NOT_VLM","scope":spec["scope_note"],
               "split_audit":split_audit,"paired_initial_inputs_identical":paired_inputs,
               "by_arm":by_arm,"sampled_contact_zero_q95":sum(e["sampled_contact_zero_q95"] for e in episodes),
               "next_requirement":"real action-free teacher labels; no SAC training or C3 validation is claimed"}
    assert paired_inputs
    finish_run(output,summary,source_paths=["scripts/collect_c3_oracle_data.py"])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    run_cli(main)
