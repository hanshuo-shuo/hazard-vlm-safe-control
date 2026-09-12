"""Independent scalar counts and an optimistic support ceiling, on Quest.

Dropping depth/occlusion checks supplies a diagnostic ceiling only. It never
changes the registered rule, produces acceptance results, or qualifies a scene.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from geometry_audit import (sha, load_calibration, load_ply, fit_surface,
                            corridor_queries, visible_support)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    src = root / "research/rellis_external_20260912"
    protocol = json.loads((src / "qualification_protocol.json").read_text())
    selection = json.loads((root / "artifacts/PILOT_SELECTION.json").read_text())["records"]
    pilot = json.loads((root / "artifacts/PILOT_ACQUISITION.json").read_text())["records"]
    report = json.loads((root / "geometry_audit/GEOMETRY_AUDIT.json").read_text())
    checked = 0
    for name, expected in json.loads((root / "geometry_audit/MANIFEST.json").read_text()).items():
        assert sha(root / "geometry_audit" / name) == expected
        checked += 1
    assert sha(src / "qualification_protocol.json") == report["protocol_sha256"]
    assert sha(src / "geometry_audit.py") == report["source_sha256"]
    assert [r["image"] for r in selection] == [r["image"] for r in pilot] == [r["image"] for r in report["records"]]
    assert all(r["sync_delta_seconds"] <= .05 for r in pilot)
    k, transform, body_from_cloud, _ = load_calibration(root)
    xy = corridor_queries(protocol).reshape(-1,2)
    allowed = set(protocol["semantic_policy"]["allowed_raw_ids"])
    prohibited = set(protocol["semantic_policy"]["prohibited_raw_ids"])
    legal = allowed | prohibited
    scalar = []
    for data, logged in zip(pilot, report["records"]):
        points = load_ply(root / data["lidar"]["path"])
        points = points[np.isfinite(points).all(1) & (np.linalg.norm(points,axis=1)>.5)] @ body_from_cloud.T
        coef, ground, _ = fit_surface(points)
        labels = cv2.imread(str(root / data["labels"]["path"]),cv2.IMREAD_UNCHANGED)
        xyz, _, known, reasons, _ = visible_support(points,xy,coef,ground,k,transform,labels.shape)
        uv = cv2.projectPoints(xyz,Rotation.from_matrix(transform[:3,:3]).as_rotvec(),transform[:3,3],k,
                              np.array([-.134313,-.025905,.002181,.00084,0.]))[0][:,0]
        entries = []
        for j in range(4):
            bad = unknown = optimistic_unknown = 0
            for idx in range(j*720,(j+1)*720):
                u,v = map(int,np.rint(uv[idx]))
                in_bounds = 0 <= u < labels.shape[1] and 0 <= v < labels.shape[0]
                raw = int(labels[v,u]) if in_bounds else -1
                if not known[idx] or raw not in legal:
                    unknown += 1
                elif raw in prohibited:
                    bad += 1
                if not in_bounds or reasons["outside_view"][idx] or reasons["unsupported_surface"][idx] or raw not in legal:
                    optimistic_unknown += 1
            expected = logged["candidates"][j]
            assert abs(bad/720-expected["cost_lower"]) < 1e-12
            assert abs((bad+unknown)/720-expected["cost_upper"]) < 1e-12
            assert abs(unknown/720-expected["unknown_fraction"]) < 1e-12
            assert optimistic_unknown <= unknown
            entries.append(dict(candidate=j, known_bad=bad, unknown=unknown,
                                optimistic_unknown_without_depth_or_occlusion=optimistic_unknown))
        scalar.append({"image": data["image"], "entries": entries,
                       "optimistically_all_four_evaluable": all(e["optimistic_unknown_without_depth_or_occlusion"]/720<=.05 for e in entries)})
    candidates = [c for r in report["records"] for c in r["candidates"]]
    out = {"status": "PASS", "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
           "verification_source_sha256": sha(Path(__file__)), "manifest_files_checked": checked,
           "scalar_sample_labels_checked": 24*4*720, "candidate_counts_checked": len(candidates),
           "max_projection_implementation_difference_px": max(r["independent_projection_max_difference_px"] for r in report["records"]),
           "unknown_fraction_range": [min(c["unknown_fraction"] for c in candidates),max(c["unknown_fraction"] for c in candidates)],
           "mean_unknown_reason_fractions_nonadditive": {name: float(np.mean([c["unknown_reasons_fraction"][name] for c in candidates])) for name in candidates[0]["unknown_reasons_fraction"]},
           "optimistically_all_four_evaluable_scenes": sum(r["optimistically_all_four_evaluable"] for r in scalar),
           "registered_all_four_evaluable_scenes": report["n_all_four_evaluable"],
           "records": scalar,
           "interpretation": "Verified the implemented conservative single-scan geometry screen. The optimistic ceiling drops depth/occlusion evidence and is not an admissible acceptance rule. Neither is a measured calibration-error bound or a statement that RELLIS cannot support other geometrically validated constructions."}
    (root / "artifacts/QUALIFICATION_VERIFICATION.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2))


if __name__ == "__main__":
    main()
