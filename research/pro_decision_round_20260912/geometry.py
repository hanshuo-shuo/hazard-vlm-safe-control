"""Continuous scene identities; rasterization never controls scene acceptance."""
from __future__ import annotations
import json
import numpy as np
import torch
from common import reference, array_sha, deployment, formula
from evaluation.oracle_spatial_exposure import rasterize_polyline
from evaluation.simulator_spatial_field import APPEARANCE_SPECS

def rng_for(seed, i, key):
    return np.random.default_rng(np.random.SeedSequence([seed, i, key]))

def field_mask(shape, center, radii, angle, size):
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    dx, dy = xx - center[0], yy - center[1]
    u = (np.cos(angle) * dx + np.sin(angle) * dy) / radii[0]
    v = (-np.sin(angle) * dx + np.cos(angle) * dy) / radii[1]
    if shape == "ellipse":
        a = u**2 + v**2 <= 1
    elif shape == "box":
        a = np.maximum(abs(u), abs(v)) <= 1
    elif shape == "diamond":
        a = abs(u) + abs(v) <= 1
    elif shape == "superellipse":
        a = u**4 + v**4 <= 1
    else:
        raise ValueError(shape)
    return a.astype(np.float32)

def raster(record, size):
    fields = np.asarray([field_mask(k, c, r, a, size) for k, c, r, a in
                         zip(record["shapes"], record["centers"], record["radii"], record["angles"])])
    fields = fields[record["assignment"]]
    footprints = np.asarray([rasterize_polyline(size, size, pts, record["radius"]) for pts in record["points"]], np.float32)
    e = np.einsum("chw,khw->kc", fields.astype(float), footprints.astype(float)) / footprints.sum((1, 2))[:, None]
    return fields, footprints, np.stack([e, e[:, ::-1]])

def low(array, size=16):
    # Same area pooling as the reference, including the noninteger 48->16 case.
    a = torch.tensor(np.ascontiguousarray(array), dtype=torch.float32)
    return torch.nn.functional.interpolate(a[None], size=(size, size), mode="area")[0].numpy()

def boundary_band(target):
    binary = target > 0
    padded = np.pad(binary, [(0, 0)] * (binary.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    neighbors = [padded[..., y:y+16, x:x+16] for y in range(3) for x in range(3)]
    return np.logical_or.reduce(neighbors) ^ np.logical_and.reduce(neighbors)

def continuous_scene(seed, i, geometry, families, p):
    rng = rng_for(seed, i, 11)
    for attempt in range(10000):
        if geometry == "balanced_anchor":
            centers = np.array([[-.34 + rng.uniform(-.025, .025), rng.uniform(-.2, .2)],
                                [.34 + rng.uniform(-.025, .025), rng.uniform(-.2, .2)]])
            radii = np.column_stack([rng.uniform(.12, .17, 2), rng.uniform(.10, .15, 2)])
            angles, shapes = np.zeros(2), np.array(["ellipse", "ellipse"])
        else:
            centers = rng.uniform(-.62, .62, (2, 2))
            radii = rng.uniform(.13, .28, (2, 2))
            angles = rng.uniform(0, 2 * np.pi, 2)
            shapes = rng.choice(["ellipse", "box"], 2)
        bound = np.linalg.norm(radii, axis=1)
        # Conservative continuous criteria independent of any requested source raster.
        if (np.abs(centers) + bound[:, None] < .98).all() and np.linalg.norm(centers[0] - centers[1]) > bound.sum():
            break
    else:
        raise RuntimeError("Continuous sampler exhausted")
    path_rng = rng_for(seed, i, 23)
    side = int(path_rng.integers(4))
    t0, t1 = path_rng.uniform(-.8, .8, 2)
    angle = side * np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    start, goal = np.array([-.92, t0]) @ rotation, np.array([.92, t1]) @ rotation
    radius = float(path_rng.uniform(.05, .10))
    points = np.asarray([np.vstack([start, path_rng.uniform(-.85, .85, (2, 2)), goal]) for _ in range(4)])
    points = points[rng_for(seed, i, 37).permutation(4)]
    assignment = rng_for(seed, i, 17).permutation(2).tolist()
    record = {"id": f"{geometry}:{seed}:{i}", "centers": centers.tolist(), "radii": radii.tolist(),
              "angles": angles.tolist(), "shapes": shapes.tolist(), "assignment": assignment,
              "points": points.tolist(), "radius": radius, "continuous_attempt": attempt,
              "family": str(rng_for(seed, i, 29).choice(families)), "render_seed": seed * 100000 + i}
    record["continuous_sha256"] = __import__("hashlib").sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record

def materialize_record(record, p, references=True):
    APPEARANCE_SPECS.update(p["appearance"]["specs"])
    result = {}
    for size in [48, 64]:
        f, w, e = raster(record, size)
        t, fp = low(f), low(w)
        result[f"target{size}"] = np.stack([t, t[::-1]])
        result[f"footprints{size}"] = fp
        ep = np.einsum("vchw,khw->vkc", result[f"target{size}"].astype(float), fp.astype(float)) / fp.sum((1, 2), dtype=float)[None, :, None]
        result[f"truth_native{size}"] = formula(ep)
        if size == 64:
            result["rgb"] = np.asarray([reference.render_requirement_rgb(x, family=record["family"],
                image_size=64, render_seed=record["render_seed"]) for x in [f, f[::-1].copy()]])
    if references:
        for size in [p["reference_check_size"], p["reference_size"]]:
            _, w, e = raster(record, size)
            result[f"truth_ref{size}"] = formula(e)
            if size == p["reference_size"]:
                result["footprints_common"] = low(w)
    result["boundary"] = boundary_band(result["target64"])
    return result

def pool_task(args):
    record, p, references = args
    torch.set_num_threads(1)
    return materialize_record(record, p, references)

def reference_audit(d, p):
    a = d[f"truth_ref{p['reference_check_size']}"][..., [0, 2, 3]]
    b = d[f"truth_ref{p['reference_size']}"][..., [0, 2, 3]]
    difference = abs(a - b)
    flip = (a > p["tau"]) != (b > p["tau"])
    value = {"label_flip_rate": float(flip.mean()), "cost_difference_p95": float(np.quantile(difference, .95)),
             "maximum_cost_difference": float(difference.max()), "label_flip_count": int(flip.sum()), "entries": int(flip.size)}
    value["pass"] = (value["label_flip_rate"] <= p["reference_gate"]["maximum_danger_label_disagreement"] and
                     value["cost_difference_p95"] <= p["reference_gate"]["maximum_cost_difference_p95"])
    return value
