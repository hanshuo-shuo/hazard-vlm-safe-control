"""Conservative qualification of visible metric corridor surface support.

No model is trained or evaluated. Semantic labels are accessed only AFTER
metric query construction, nonsemantic surface estimation and visibility masks.
This is a qualification diagnostic, not calibrated physical ground truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DISTORTION = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0])
SUPPORT_RADIUS = 0.35
PLANE_TOLERANCE = 0.15
OCCLUSION_MARGIN_M = 0.35


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ply(path):
    types = {"float": "<f4", "double": "<f8", "uint": "<u4", "int": "<i4",
             "ushort": "<u2", "short": "<i2", "uchar": "u1", "char": "i1"}
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 100:
                raise ValueError("Unexpected PLY header")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only documented little-endian PLY is supported")
        n = int(next(s.split()[-1] for s in header if s.startswith("element vertex")))
        fields = [(s.split()[2], types[s.split()[1]]) for s in header if s.startswith("property ")]
        data = np.fromfile(f, dtype=np.dtype(fields), count=n)
    if len(data) != n:
        raise ValueError("Truncated PLY")
    return np.column_stack([data[k] for k in ("x", "y", "z")]).astype(float)


def load_calibration(root):
    info = np.loadtxt(root / "metadata/intrinsics/Rellis-3D/00000/camera_info.txt")
    k = np.array([[info[0], 0, info[2]], [0, info[1], info[3]], [0, 0, 1.]])
    d = yaml.safe_load((root / "metadata/extrinsics/Rellis_3D/00000/transforms.yaml").read_text())["os1_cloud_node-pylon_camera_node"]
    q = np.array([d["q"][a] for a in ("x", "y", "z", "w")])
    r = Rotation.from_quat(q).as_matrix()
    t = np.array([d["t"][a] for a in ("x", "y", "z")])
    # Official utility inverts the stored camera-to-cloud pose.
    transform = np.eye(4)
    transform[:3, :3] = r.T
    transform[:3, 3] = -r.T @ t
    # Public URDF orients os1_lidar opposite to the body-forward X axis. Keep
    # the origin at the LiDAR (segment ranges are from the sensor), but rotate
    # the metric construction into body-oriented axes. No semantic inference.
    urdf = ET.parse(root / "official_rellis/catkin_ws/src/platform_description/urdf/warthog.urdf").getroot()
    joints = {j.find("child").attrib["link"]: j for j in urdf.findall("joint")}
    name, body_from_cloud = "ouster1/os1_lidar", np.eye(3)
    chain = []
    while name != "base_link":
        j = joints[name]
        origin = j.find("origin")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        body_from_cloud = Rotation.from_euler("xyz", rpy).as_matrix() @ body_from_cloud
        chain.append(j.attrib["name"])
        name = j.find("parent").attrib["link"]
    camera_from_body = transform.copy()
    camera_from_body[:3, :3] = transform[:3, :3] @ body_from_cloud.T
    if camera_from_body[2, 0] < .9:
        raise ValueError("Official body frame not consistent with forward camera axis")
    return k, camera_from_body, body_from_cloud, chain


def project(points, k, transform):
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    depth = xyz[:, 2]
    x = np.divide(xyz[:, 0], depth, out=np.zeros_like(depth), where=depth > 0)
    y = np.divide(xyz[:, 1], depth, out=np.zeros_like(depth), where=depth > 0)
    r2 = x*x + y*y
    a, b, p, q, c = DISTORTION
    radial = 1 + a*r2 + b*r2*r2 + c*r2*r2*r2
    xd = x*radial + 2*p*x*y + q*(r2+2*x*x)
    yd = y*radial + p*(r2+2*y*y) + 2*q*x*y
    return np.column_stack([k[0, 0]*xd+k[0, 2], k[1, 1]*yd+k[1, 2]]), depth


def fit_surface(points):
    # Equalize point density spatially before fitting: one lower-quartile height
    # per 0.4 m cell. No labels or image content enter this operation.
    keep = ((points[:, 0] >= 2) & (points[:, 0] <= 15) &
            (np.abs(points[:, 1]) <= 5) & (points[:, 2] > -3) & (points[:, 2] < -0.25))
    p = points[keep]
    cells = np.floor(p[:, :2] / 0.4).astype(int)
    groups = {}
    for i, key in enumerate(map(tuple, cells)):
        groups.setdefault(key, []).append(i)
    representatives = []
    for ids in groups.values():
        if len(ids) >= 3:
            part = p[ids]
            representatives.append([*np.mean(part[:, :2], axis=0), np.quantile(part[:, 2], .25)])
    psmall = np.asarray(representatives)
    if len(psmall) < 20:
        raise ValueError("Insufficient geometric surface support")
    design = np.column_stack([psmall[:, :2], np.ones(len(psmall))])
    rng = np.random.default_rng(20260912)
    best = None
    for _ in range(300):
        idx = rng.choice(len(psmall), 3, replace=False)
        if np.linalg.cond(design[idx]) > 1000:
            continue
        coef = np.linalg.solve(design[idx], psmall[idx, 2])
        if np.linalg.norm(coef[:2]) > math.tan(math.radians(15)) or not -2.5 < coef[2] < -0.3:
            continue
        inside = np.abs(design @ coef - psmall[:, 2]) <= PLANE_TOLERANCE
        if best is None or inside.sum() > best.sum():
            best = inside
    if best is None or best.sum() < 20:
        raise ValueError("No supported modest-slope local surface")
    coef = np.linalg.lstsq(design[best], psmall[best, 2], rcond=None)[0]
    residual = p[:, 2] - (p[:, :2] @ coef[:2] + coef[2])
    ground = p[np.abs(residual) <= PLANE_TOLERANCE]
    return coef, ground, {"fit_cells": len(psmall), "inlier_cells": int(best.sum()),
                          "plane": coef.tolist(), "surface_points": len(ground),
                          "cell_residual_p95_m": float(np.quantile(np.abs(design[best] @ coef - psmall[best, 2]), .95))}


def corridor_queries(protocol):
    c = protocol["corridors"]
    spacing = c["sampling_spacing_m"]
    s = np.arange(c["longitudinal_segment_m"][0] + spacing/2,
                  c["longitudinal_segment_m"][1], spacing)
    lateral = np.arange(-c["width_m"]/2 + spacing/2, c["width_m"]/2, spacing)
    ss, ll = np.meshgrid(s, lateral, indexing="ij")
    out = []
    for angle in c["heading_degrees"]:
        theta = np.deg2rad(angle)
        out.append(np.column_stack([(ss*np.cos(theta)-ll*np.sin(theta)).ravel(),
                                    (ss*np.sin(theta)+ll*np.cos(theta)).ravel()]))
    return np.stack(out)


def visible_support(points, queries, coef, ground, k, transform, shape):
    h, w = shape
    tree = cKDTree(ground[:, :2])
    dist, ids = tree.query(queries, k=8)
    heights = ground[ids, 2] - (ground[ids, :2] @ coef[:2] + coef[2])
    local_offset = np.median(heights, axis=1)
    z = queries @ coef[:2] + coef[2] + local_offset
    xyz = np.column_stack([queries, z])
    uv, depth = project(xyz, k, transform)
    geometric = (dist[:, 0] <= SUPPORT_RADIUS) & (dist[:, 7] <= .75)
    inview = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w-1) & (uv[:, 1] >= 0) & (uv[:, 1] < h-1)
    puv, pd = project(points, k, transform)
    pvalid = (pd > 0) & (puv[:, 0] >= 0) & (puv[:, 0] < w) & (puv[:, 1] >= 0) & (puv[:, 1] < h)
    # Nearest depth in an 8x8 sensor projection bin; no inpainting of unseen bins.
    bin_size = 8
    bh, bw = (h+7)//8, (w+7)//8
    depth_map = np.full(bh*bw, np.inf)
    pixels = (puv[pvalid]/bin_size).astype(int)
    np.minimum.at(depth_map, pixels[:, 1]*bw+pixels[:, 0], pd[pvalid])
    bins = np.floor(uv/bin_size).astype(int)
    bounded = np.column_stack([np.clip(bins[:, 0], 0, bw-1), np.clip(bins[:, 1], 0, bh-1)])
    front = depth_map[bounded[:, 1]*bw+bounded[:, 0]]
    depth_observed = np.isfinite(front)
    unoccluded = depth_observed & (front >= depth-OCCLUSION_MARGIN_M)
    known = geometric & inview & unoccluded
    return xyz, uv, known, {"outside_view": ~inview, "unsupported_surface": ~geometric,
                            "no_projected_depth": ~depth_observed,
                            "occluded": depth_observed & ~unoccluded}, (puv[pvalid], pd[pvalid])


def cost_interval(labels, uv, geometry_known, prohibited, allowed):
    h, w = labels.shape
    pixels = np.rint(uv).astype(int)
    values = labels[np.clip(pixels[:, 1], 0, h-1), np.clip(pixels[:, 0], 0, w-1)]
    known = geometry_known & np.isin(values, prohibited + allowed)
    bad = known & np.isin(values, prohibited)
    lower, unknown = float(bad.mean()), float((~known).mean())
    return lower, lower+unknown, known, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    src = root / "research/rellis_external_20260912"
    protocol = json.loads((src / "qualification_protocol.json").read_text())
    pilot = json.loads((root / "artifacts/PILOT_ACQUISITION.json").read_text())
    outdir = root / "geometry_audit"
    outdir.mkdir(exist_ok=False)
    k, transform, body_from_cloud, transform_chain = load_calibration(root)
    queries = corridor_queries(protocol)
    policy = protocol["semantic_policy"]
    report = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "protocol_sha256": sha(src / "qualification_protocol.json"),
              "source_sha256": sha(Path(__file__)), "n_requested": len(pilot["records"]),
              "registration_uncertainty_validated": False,
              "body_and_gravity_frame_fully_validated": False,
              "scope": "LiDAR-centred, URDF-body-oriented frame + nonsemantic local surface estimate. Ground-area cost intervals bound unknown support only, conditional on projection. They do not bound unmeasured calibration/label error or verify gravity alignment.",
              "body_from_cloud_rotation": body_from_cloud.tolist(), "urdf_chain": transform_chain,
              "records": []}
    thumbnails = []
    for index, record in enumerate(pilot["records"]):
        item = {"image": record["image"], "sync_delta_seconds": record.get("sync_delta_seconds")}
        try:
            if not all(record.get(t, {}).get("status") == "DOWNLOADED" for t in ["rgb", "labels", "lidar"]):
                raise ValueError("Missing fixed pilot data")
            for kind in ["rgb", "labels", "lidar"]:
                assert sha(root / record[kind]["path"]) == record[kind]["sha256"]
            rgb = cv2.imread(str(root / record["rgb"]["path"]))
            points = load_ply(root / record["lidar"]["path"])
            points = points[np.isfinite(points).all(1) & (np.linalg.norm(points, axis=1) > .5)]
            points = points @ body_from_cloud.T
            coef, ground, fit = fit_surface(points)
            item["surface"] = fit
            xyz, uv, geometric, reasons, cloud = visible_support(points, queries.reshape(-1, 2), coef, ground, k, transform, rgb.shape[:2])
            rvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
            cvuv = cv2.projectPoints(xyz, rvec, transform[:3, 3], k, DISTORTION)[0][:, 0]
            item["independent_projection_max_difference_px"] = float(np.max(np.abs(cvuv-uv)))
            assert item["independent_projection_max_difference_px"] < 1e-6
            # Only now open the permitted pilot human labels.
            labels = cv2.imread(str(root / record["labels"]["path"]), cv2.IMREAD_UNCHANGED)
            if labels.ndim != 2 or labels.shape != rgb.shape[:2]:
                raise ValueError("Native image/label dimensions mismatch")
            item["raw_ids_present"] = np.unique(labels).tolist()
            item["candidates"] = []
            overlay = rgb.copy()
            uv = uv.reshape(4, -1, 2)
            geometric = geometric.reshape(4, -1)
            for j in range(4):
                lower, upper, known, label = cost_interval(labels, uv[j], geometric[j], policy["prohibited_raw_ids"], policy["allowed_raw_ids"])
                n = len(known)
                item["candidates"].append({"heading_degrees": protocol["corridors"]["heading_degrees"][j],
                    "surface_samples": n, "geometrically_known_fraction": float(geometric[j].mean()),
                    "unknown_fraction": float((~known).mean()), "cost_lower": lower, "cost_upper": upper,
                    "danger_status": "noncompliant" if lower > .02 else "compliant" if upper <= .02 else "unresolved",
                    "unknown_reasons_fraction": {key: float(value.reshape(4, -1)[j].mean()) for key, value in reasons.items()}})
                # Show every tenth registered sample; unknown is red, supported cyan.
                for pixel, usable in zip(uv[j][::10], known[::10]):
                    u, v = np.rint(pixel).astype(int)
                    if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                        cv2.circle(overlay, (u, v), 4, (0, 220, 0) if usable else (0, 0, 255), -1)
            item["all_four_evaluable"] = all(c["unknown_fraction"] <= .05 for c in item["candidates"])
            item["status"] = "AUDITED"
            thumb = cv2.resize(overlay, (640, 400))
            cv2.putText(thumb, f"Pilot {index:02d} | unknown " + "/".join(f"{c['unknown_fraction']:.0%}" for c in item["candidates"]),
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(outdir / f"pilot_{index:02d}.png"), thumb)
            thumbnails.append(thumb)
        except Exception as exc:
            item.update(status="UNEVALUABLE", reason=repr(exc), all_four_evaluable=False)
            thumb = np.zeros((400,640,3), np.uint8)
            cv2.putText(thumb, f"Pilot {index:02d}: unevaluable", (20,180), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            thumbnails.append(thumb)
        report["records"].append(item)
        print(json.dumps({"index": index, "status": item["status"], "all_four_evaluable": item["all_four_evaluable"]}), flush=True)
    report["n_audited"] = sum(r["status"] == "AUDITED" for r in report["records"])
    report["n_all_four_evaluable"] = sum(r["all_four_evaluable"] for r in report["records"])
    widths = [c["cost_upper"]-c["cost_lower"] for r in report["records"] for c in r.get("candidates", [])]
    report["unknown_interval_width_p95"] = float(np.quantile(widths, .95)) if widths else None
    report["known_support_screen_pass"] = report["n_all_four_evaluable"] / report["n_requested"] >= .8
    report["support_ambiguity_screen_pass"] = bool(widths and report["unknown_interval_width_p95"] <= .005)
    support_pass = report["known_support_screen_pass"] and report["support_ambiguity_screen_pass"]
    report["qualification_pass"] = None if support_pass else False
    report["qualification_status"] = "PENDING_REGISTRATION_AND_SPLITS" if support_pass else "STOP_CURRENT_GEOMETRY_SUPPORT"
    report["decision"] = "Do not train or open external final tests until support, independent registration uncertainty, and spatial split prerequisites are satisfied."
    cv2.imwrite(str(outdir / "all_24_pilot_geometry.png"), np.vstack([np.hstack(thumbnails[i:i+4]) for i in range(0,24,4)]))
    (outdir / "GEOMETRY_AUDIT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {p.name: sha(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
