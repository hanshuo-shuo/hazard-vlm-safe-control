"""Color-blob keep-out detector — the "why not just write a CV detector?" baseline.

This is the standard perception competitor a roboticist reaches for before a VLM:
a hand-coded, per-appearance detector that reads the keep-out zones straight off
the rendered image and feeds them to the SAME safe controller the VLM arms use.

It estimates each semantic keep-out's (x, y, r) in world coordinates from pixels
alone — no env ground-truth, no VLM. On a clean rendered toy it is near-perfect,
which is exactly the point of the baseline: it makes the VLM earn its place not on
*this* scene but on what the detector CANNOT do without per-terrain re-engineering
(generalising zero-shot to an unseen/implicit terrain category). Each terrain
appearance needs its own colour target hand-coded here (water/mud/grass/amber);
add a new terrain and the detector is blind until a human adds its colour — the
"one VLM vs N hand-coded detectors" cost the paper quantifies.

Honesty notes:
- It is hand-coded to the renderer palette (that IS the baseline — a per-appearance
  detector). It is NOT a VLM and gets NO leakage-clean credit; it is the classical
  perception stack the VLM is measured against.
- It returns ESTIMATED geometry (centroid + radius from pixels), so feeding it to
  the controller is apples-to-apples with B+ (VLM-estimated) and with the soft
  oracle (true geometry): the three differ only in where the keep-out comes from.
"""

from __future__ import annotations

import numpy as np


def _blend(fill_rgb, alpha_255: int, bg) -> np.ndarray:
    """Expected on-screen RGB of a translucent fill drawn over the background."""
    a = alpha_255 / 255.0
    return a * np.asarray(fill_rgb, np.float32) + (1.0 - a) * np.asarray(bg, np.float32)


def terrain_targets(renderer) -> dict[str, np.ndarray]:
    """Expected blended RGB for each known keep-out appearance, from the palette."""
    bg = renderer.bg_color
    return {
        "water": _blend(renderer.water_fill[:3], renderer.water_fill[3], bg),
        "mud": _blend(renderer.mud_fill[:3], renderer.mud_fill[3], bg),
        "grass": _blend(renderer.grass_fill[:3], renderer.grass_fill[3], bg),
        "restricted": _blend(renderer.semantic_fill[:3], renderer.semantic_fill[3], bg),
    }


def estimate_zones_from_image(
    img_rgb: np.ndarray,
    renderer,
    *,
    color_thresh: float = 33.0,
    grid: int = 64,
    min_cells: int = 3,
    occ_frac: float = 0.30,
    styles: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Detect keep-out zones from a rendered RGB frame; return (K, 3) world (x,y,r).

    Pipeline: per-pixel colour match to the expected blended terrain colours ->
    accumulate matches into a coarse world-space grid -> 4-connected components ->
    one (centroid, radius) disk per component. Pure perception, no ground-truth.
    """
    img = np.asarray(img_rgb, dtype=np.float32)
    targets = terrain_targets(renderer)
    if styles:
        targets = {k: v for k, v in targets.items() if k in styles}

    wmin = renderer.world_x_min
    wspan = renderer.world_x_max - renderer.world_x_min
    size = renderer.img_size
    wpp = wspan / size            # world units per pixel
    cell_px = size / grid         # pixels per coarse cell edge
    cell_w = wspan / grid         # world units per coarse cell edge
    min_count = max(1, int(occ_frac * cell_px * cell_px))

    # ONE detector per terrain colour (the literal "N hand-coded detectors"
    # baseline): segment + connected-component each colour separately, so two
    # different terrains that touch in a corridor are NOT merged into one blob.
    zones: list[tuple[float, float, float]] = []
    for tgt in targets.values():
        d = np.linalg.norm(img - tgt[None, None, :], axis=2)
        mask = d <= color_thresh
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        gi = np.clip(((xs + 0.5) / cell_px).astype(int), 0, grid - 1)
        gj = np.clip(((1.0 - (ys + 0.5) / size) * grid).astype(int), 0, grid - 1)
        counts = np.zeros((grid, grid), dtype=np.int32)
        np.add.at(counts, (gi, gj), 1)
        # Require a cell to be DENSELY filled (interior of a disk) — excludes the
        # 1px grid lines that share water's pale-blue hue but sparsely fill a cell.
        occupied = counts >= min_count

        seen = np.zeros((grid, grid), dtype=bool)
        for a in range(grid):
            for b in range(grid):
                if not occupied[a, b] or seen[a, b]:
                    continue
                stack = [(a, b)]          # BFS over occupied 4-neighbourhood
                seen[a, b] = True
                comp: list[tuple[int, int]] = []
                while stack:
                    ca, cb = stack.pop()
                    comp.append((ca, cb))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = ca + da, cb + db
                        if 0 <= na < grid and 0 <= nb < grid and occupied[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            stack.append((na, nb))
                if len(comp) < min_cells:
                    continue
                tot = float(sum(counts[ca, cb] for ca, cb in comp))
                cxw = sum(counts[ca, cb] * (wmin + (ca + 0.5) * cell_w) for ca, cb in comp) / tot
                cyw = sum(counts[ca, cb] * (wmin + (cb + 0.5) * cell_w) for ca, cb in comp) / tot
                # Radius: blend area- and extent-based estimates (robust to the
                # hazards/markers drawn ON TOP of a zone, which undercount pixels).
                area = tot * wpp * wpp
                r_area = float(np.sqrt(area / np.pi))
                r_ext = max(
                    np.hypot((wmin + (ca + 0.5) * cell_w) - cxw,
                             (wmin + (cb + 0.5) * cell_w) - cyw)
                    for ca, cb in comp
                ) + 0.5 * cell_w
                zones.append((cxw, cyw, float(0.5 * (r_area + r_ext))))

    return np.asarray(zones, dtype=np.float32).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Self-test: render known scenes via the env and report detection accuracy.
#   python zone_detector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from env_pointhazard import PointHazardConfig, PointHazardEnv
    from hazard_renderer import HazardRenderer

    def _match(true_z, det_z):
        """Greedy nearest match true->det; return list of (true, det|None)."""
        det = list(map(tuple, det_z))
        out = []
        for tz in true_z:
            best, bi = None, None
            for i, dz in enumerate(det):
                d = np.hypot(tz[0] - dz[0], tz[1] - dz[1])
                if best is None or d < best:
                    best, bi = d, i
            out.append((tz, det[bi] if bi is not None else None))
        return out

    def _iou(t, d):
        # circle-circle IoU (numeric, cheap).
        import numpy as _np
        R = max(t[2], d[2]) + 0.05
        xs = _np.linspace(min(t[0], d[0]) - R, max(t[0], d[0]) + R, 200)
        ys = _np.linspace(min(t[1], d[1]) - R, max(t[1], d[1]) + R, 200)
        gx, gy = _np.meshgrid(xs, ys)
        int = ((gx - t[0]) ** 2 + (gy - t[1]) ** 2 <= t[2] ** 2)
        ind = ((gx - d[0]) ** 2 + (gy - d[1]) ** 2 <= d[2] ** 2)
        i = _np.logical_and(int, ind).sum()
        u = _np.logical_or(int, ind).sum()
        return i / max(1, u)

    for label, nz, styles in (
        ("single water", 1, ()),
        ("single amber", 1, ()),
        ("hetero water,mud,grass", 3, ("water", "mud", "grass")),
    ):
        zsem = "explicit" if "amber" in label else "implicit"
        cfg = PointHazardConfig(
            n_hazards=8, n_semantic_zones=nz,
            semantic_styles=styles,
        )
        env = PointHazardEnv(cfg=cfg)
        rnd = HazardRenderer.from_env(
            env, semantic_style=("restricted" if zsem == "explicit" else "water")
        )
        env.attach_renderer(rnd)
        cerr, rerr, ious, ndet, ntrue = [], [], [], 0, 0
        for seed in range(43, 63):
            env.reset(seed=seed)
            true_z = np.asarray(env.semantic_zones, np.float32).reshape(-1, 3)
            img = env.render()
            det_z = estimate_zones_from_image(img, rnd, styles=styles or None)
            ntrue += len(true_z); ndet += len(det_z)
            for tz, dz in _match(true_z, det_z):
                if dz is None:
                    continue
                cerr.append(np.hypot(tz[0] - dz[0], tz[1] - dz[1]))
                rerr.append(abs(tz[2] - dz[2]))
                ious.append(_iou(tz, dz))
        print(f"[{label:24}] zones true={ntrue} det={ndet}  "
              f"centroid_err={np.mean(cerr):.3f}  radius_err={np.mean(rerr):.3f}  "
              f"IoU={np.mean(ious):.3f}  (matched={len(ious)})")
