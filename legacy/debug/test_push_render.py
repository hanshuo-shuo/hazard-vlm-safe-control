"""Archived visual smoke: render PointPush agent and box trajectories."""

import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
from env_pointpushhazard import PointPushHazardConfig, make_env
from pointpush_hazard_renderer import PointPushHazardRenderer
from pivot_vlm import generate_candidates
from shared_autonomy_pointpush_hazard_learned_physics_pivot import (
    PushPrediction, annotate_push_predictions, _obs_parts, _push_pose_for_box,
    _agent_clearance, _box_clearance, _push_pose_metrics,
)

def fake_rollout(obs, action, cfg, horizon=5):
    """Simulate a rough rollout without learned physics (just apply force).

    Uses multiple substeps per horizon step for better contact detection.
    """
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)
    ap, av = agent_pos.copy(), agent_vel.copy()
    bp, bv = box_pos.copy(), box_vel.copy()
    agent_traj = [ap.copy()]
    box_traj = [bp.copy()]
    dt = cfg.dt
    substeps = 4
    sub_dt = dt / substeps
    for _ in range(horizon):
        # Apply force at start of step
        av = av + action * cfg.force_scale * dt
        for _s in range(substeps):
            av = av * (1 - cfg.agent_drag * sub_dt)
            speed = float(np.linalg.norm(av))
            if speed > cfg.max_agent_speed:
                av *= cfg.max_agent_speed / speed
            ap = ap + av * sub_dt

            # Contact: use effective radius for square box
            dist_ab = float(np.linalg.norm(ap - bp))
            contact_dist = cfg.agent_radius + cfg.box_half_size * 0.9
            if dist_ab < contact_dist and dist_ab > 1e-6:
                push_n = (bp - ap) / dist_ab
                overlap = contact_dist - dist_ab
                ap = ap - push_n * overlap * 0.5
                push_force = float(np.dot(av, push_n))
                if push_force > 0:
                    transfer = push_force * (cfg.agent_mass / (cfg.agent_mass + cfg.box_mass))
                    bv = bv + push_n * transfer * 1.5
                    av = av - push_n * transfer * 0.5

            bv = bv * (1 - cfg.box_drag * sub_dt)
            bspeed = float(np.linalg.norm(bv))
            if bspeed > cfg.max_box_speed:
                bv *= cfg.max_box_speed / bspeed
            bp = bp + bv * sub_dt

            ap = np.clip(ap, -cfg.arena_half + cfg.agent_radius, cfg.arena_half - cfg.agent_radius)
            bp = np.clip(bp, -cfg.arena_half + cfg.box_half_size, cfg.arena_half - cfg.box_half_size)

        agent_traj.append(ap.copy())
        box_traj.append(bp.copy())
    return np.array(agent_traj), np.array(box_traj)


def main():
    cfg = PointPushHazardConfig()
    env = make_env(cfg=cfg, seed=42)
    obs, info = env.reset()

    # Drive agent to directly behind the box (push position), touching distance
    for step_i in range(200):
        ap, _, bp, _, g, _ = _obs_parts(obs, cfg)
        pp, pd, pg = _push_pose_for_box(bp, g, cfg)
        # First go to push pose, then push slightly toward box
        to_pp = pp - ap
        dist_pp = float(np.linalg.norm(to_pp))
        if dist_pp < 0.15:
            # Close enough to push pose - nudge toward box to be touching
            to_box = bp - ap
            dist_box = float(np.linalg.norm(to_box))
            if dist_box < cfg.agent_radius + cfg.box_half_size + 0.05:
                break  # touching!
            action = np.clip(to_box / max(dist_box, 1e-6) * 0.8, -1, 1).astype(np.float32)
        else:
            action = np.clip(to_pp / max(dist_pp, 1e-6), -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            break

    renderer = PointPushHazardRenderer.from_env(env)
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)

    # Render base scene
    base_frame = renderer.render(
        agent_xy=agent_pos,
        box_xy=box_pos,
        goal_xy=goal,
        hazards=hazards,
        agent_vel_xy=agent_vel,
        box_vel_xy=box_vel,
    )
    from PIL import Image
    base_image = Image.fromarray(base_frame)

    # Match the runtime PIVOT action set: 8 directions x 1 magnitude.
    candidates = generate_candidates(n_directions=8, n_magnitudes=1)
    n = len(candidates)

    # Build fake predictions
    predictions = []
    for i, cand in enumerate(candidates):
        action = np.clip(cand, -1, 1)
        agent_traj, box_traj = fake_rollout(obs, action, cfg, horizon=3)
        min_ac = _agent_clearance(agent_traj[-1], hazards, cfg)
        min_bc = _box_clearance(box_traj[-1], hazards, cfg)
        start_dist = float(np.linalg.norm(box_pos - goal))
        final_dist = float(np.linalg.norm(box_traj[-1] - goal))
        progress = start_dist - final_dist
        pp_dist, pp_behind, pp_lat, pp_slot = _push_pose_metrics(
            agent_traj[-1], box_traj[-1], goal, cfg
        )
        pp_start_dist, _, _, _ = _push_pose_metrics(agent_pos, box_pos, goal, cfg)

        predictions.append(PushPrediction(
            choice_id=i + 1,
            action=action,
            agent_xy=agent_traj,
            box_xy=box_traj,
            min_agent_clearance=min_ac,
            min_box_clearance=min_bc,
            min_system_clearance=min(min_ac, min_bc),
            safe=bool(min_ac >= 0.12 and min_bc >= 0.12),
            box_goal_progress=progress,
            final_box_goal_dist=final_dist,
            agent_push_pose_progress=pp_start_dist - pp_dist,
            final_agent_push_pose_dist=pp_dist,
            final_push_slot_score=pp_slot,
            final_push_behind=pp_behind,
            final_push_lateral_error=pp_lat,
            score=progress + 0.25 * min(min_ac, min_bc),
            model_ready=True,
        ))

    # Render annotated image
    annotated = annotate_push_predictions(
        base_image,
        candidates,
        predictions,
        renderer,
        agent_world_xy=agent_pos,
        arrow_length_world=0.8,
        model_ready=True,
        obs=obs,
        cfg=cfg,
    )

    out_path = os.path.join(os.path.dirname(__file__), "test_push_render_output.png")
    annotated.save(out_path)
    print(f"Saved annotated image to: {out_path}")
    print(f"Image size: {annotated.size}")

    # Also save just the main-image portion for closer look
    main_only = annotated.crop((0, 0, base_image.size[0], base_image.size[1]))
    main_path = os.path.join(os.path.dirname(__file__), "test_push_render_main.png")
    main_only.save(main_path)
    print(f"Saved main-only image to: {main_path}")

    # Also save the OLD style (arrows only) for comparison
    from pivot_vlm import annotate_candidates
    old_style = annotate_candidates(
        base_image, candidates, renderer,
        agent_world_xy=agent_pos, arrow_length_world=0.8,
    )
    old_path = os.path.join(os.path.dirname(__file__), "test_push_render_old.png")
    old_style.save(old_path)
    print(f"Saved old-style (arrows only) to: {old_path}")


if __name__ == "__main__":
    main()
