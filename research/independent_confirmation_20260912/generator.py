"""Independent anonymous geometry/actions; labels only determine downstream costs."""
from __future__ import annotations

import numpy as np
from common import reference, array_sha
from evaluation.oracle_spatial_exposure import rasterize_polyline
from evaluation.simulator_spatial_field import APPEARANCE_SPECS, lowres_footprint


def stream(seed, index, component):
    return np.random.default_rng(np.random.SeedSequence([seed, index, component]))


def mask(shape, center, radii, angle, size):
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    dx, dy = xx - center[0], yy - center[1]
    u = (np.cos(angle) * dx + np.sin(angle) * dy) / radii[0]
    v = (-np.sin(angle) * dx + np.cos(angle) * dy) / radii[1]
    if shape == "ellipse":
        selected = u**2 + v**2 <= 1
    elif shape == "box":
        selected = np.maximum(np.abs(u), np.abs(v)) <= 1
    elif shape == "diamond":
        selected = np.abs(u) + np.abs(v) <= 1
    elif shape == "superellipse":
        selected = u**4 + v**4 <= 1
    else:
        raise ValueError(shape)
    return selected.astype(np.float32)


def anonymous_geometry(seed, index, shapes, spec):
    rng = stream(seed, index, 11)
    size = spec["highres_size"]
    for attempt in range(1000):
        centers = rng.uniform(*spec["center_range"], size=(2, 2))
        radii = rng.uniform(*spec["radius_range"], size=(2, 2))
        angles = rng.uniform(*spec["rotation_range"], size=2)
        kinds = rng.choice(shapes, size=2)
        fields = np.asarray([mask(k, c, r, a, size) for k, c, r, a in zip(kinds, centers, radii, angles)])
        # These constraints are independent of property identity and paths.
        if (fields.sum((1, 2)) >= spec["minimum_pixels_per_patch"]).all() and not (fields[0] * fields[1]).any():
            record = {"centers": centers.tolist(), "radii": radii.tolist(), "angles": angles.tolist(),
                      "shapes": kinds.tolist(), "attempt": attempt,
                      "unordered_geometry_sha256": array_sha(np.sort(fields, axis=0))}
            return fields, record
    raise RuntimeError("geometry sampler exhausted")


def anonymous_paths(seed, index, spec):
    rng = stream(seed, index, 23)
    side = int(rng.integers(4))
    t0, t1 = rng.uniform(-0.8, 0.8, size=2)
    start, goal = np.array([-0.92, t0]), np.array([0.92, t1])
    angle = side * np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    start, goal = start @ rotation, goal @ rotation
    radius = float(rng.uniform(*spec["footprint_radius_range"]))
    points = np.asarray([np.vstack((start, rng.uniform(*spec["waypoint_range"], size=(2, 2)), goal))
                         for _ in range(spec["candidates"])])
    order = stream(seed, index, 37).permutation(spec["candidates"])
    points = points[order]
    high = np.asarray([rasterize_polyline(spec["highres_size"], spec["highres_size"], p, radius) for p in points])
    low = np.asarray([lowres_footprint(f, spec["field_size"]) for f in high])
    return low, {"points": points.tolist(), "radius": radius, "permutation": order.tolist()}


def exposures(fields, footprints):
    # N x V x C x H x W and N x K x H x W -> N x V x K x C.
    f = np.asarray(fields, dtype=np.float64)
    w = np.asarray(footprints, dtype=np.float64)
    return np.einsum("nvchw,nkhw->nvkc", f, w) / w.sum((-2, -1))[:, None, :, None]


def formula(exposure):
    ew, ef = np.moveaxis(np.asarray(exposure, dtype=np.float64), -1, 0)
    return np.stack((ew, np.zeros_like(ew), ew + ef - ew * ef, ef), axis=-1)


def make_random(spec, p, paired=True):
    APPEARANCE_SPECS.update(p["new_appearance_specs"])
    rgbs, targets, footprints, records = [], [], [], []
    for index in range(spec["scene_count"]):
        fields, geometry = anonymous_geometry(spec["scene_seed"], index, spec["shapes"], p["generator"])
        assignment = stream(spec["scene_seed"], index, 17).permutation(2)
        fields = fields[assignment]
        family = str(stream(spec["scene_seed"], index, 29).choice(spec["appearance_families"]))
        render_seed = spec["scene_seed"] * 100000 + index
        paths, paths_record = anonymous_paths(spec["scene_seed"], index, p["generator"])
        variants = [fields, fields[::-1].copy()] if paired else [fields]
        rgb = np.asarray([reference.render_requirement_rgb(f, family=family, image_size=64, render_seed=render_seed)
                          for f in variants])
        target = np.asarray([reference.lowres_field(f, 16) for f in variants])
        rgbs.append(rgb)
        targets.append(target)
        footprints.append(paths)
        records.append({"id": f"{spec['scene_seed']}:{index}", "family": family,
                        "assignment": assignment.tolist(), "render_seed": render_seed,
                        **geometry, **paths_record, "rgb_sha256": [array_sha(x) for x in rgb]})
    return pack(rgbs, targets, footprints), records


def make_legacy(spec, p, *, split, balanced=False, paired=True):
    base = reference.build_rgb_scenes(split, spec, image_size=64, field_size=16)
    chosen = set()
    if balanced:
        chosen = set(np.random.default_rng(p["balance_seeds"][split]).permutation(len(base))[:len(base)//2].tolist())
    rgbs, targets, footprints, records = [], [], [], []
    for index, item in enumerate(base):
        f = item.scene.fields[::-1].copy() if index in chosen else item.scene.fields
        order = stream(spec["scene_seed"], index, 37).permutation(4)
        variants = [f, f[::-1].copy()] if paired else [f]
        rgb = np.asarray([reference.render_requirement_rgb(v, family=item.family, image_size=64, render_seed=item.render_seed)
                          for v in variants])
        target = np.asarray([reference.lowres_field(v, 16) for v in variants])
        rgbs.append(rgb)
        targets.append(target)
        footprints.append(item.footprints[order])
        records.append({"id": f"legacy:{spec['scene_seed']}:{index}", "family": item.family,
                        "unordered_geometry_sha256": array_sha(np.sort(f, axis=0)),
                        "rgb_sha256": [array_sha(x) for x in rgb], "permutation": order.tolist()})
    return pack(rgbs, targets, footprints), records


def pack(rgb, target, footprints):
    arrays = {"rgb": np.asarray(rgb, dtype=np.uint8), "target": np.asarray(target, dtype=np.float32),
              "footprints": np.asarray(footprints, dtype=np.float32)}
    arrays["exposure"] = exposures(arrays["target"], arrays["footprints"])
    arrays["truth"] = formula(arrays["exposure"])
    return arrays
