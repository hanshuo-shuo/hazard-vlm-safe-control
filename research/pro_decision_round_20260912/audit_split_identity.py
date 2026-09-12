"""Final data QA excluding scene IDs, appearance, labels and source raster from identities."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def main(root):
    seen_fields,seen_scenes={},{}
    counts,overlaps={},[]
    for stage in ["e2_data","e3_data"]:
        for path in sorted((root/stage).glob("*.json")):
            if path.name in ["AUDIT.json","MANIFEST.json"]:
                continue
            records=json.loads(path.read_text())
            group=f"{stage}/{path.stem}"
            counts[group]=len(records)
            for record in records:
                patches=sorted([dict(center=c,radii=r,angle=a,shape=s) for c,r,a,s in
                    zip(record["centers"],record["radii"],record["angles"],record["shapes"])],key=lambda v:json.dumps(v,sort_keys=True))
                field=digest(patches)
                paths=sorted(record["points"],key=lambda v:json.dumps(v))
                scene=digest({"patches":patches,"paths":paths,"radius":record["radius"]})
                for kind,value,seen in [("field_geometry",field,seen_fields),("field_and_path_geometry",scene,seen_scenes)]:
                    if value in seen:
                        overlaps.append({"kind":kind,"previous":seen[value],"current":group,"scene_id":record["id"]})
                    seen[value]=group
    result={"status":"PASS" if not overlaps else "FAIL","at_utc":datetime.now(timezone.utc).isoformat(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"groups":counts,
            "unique_field_geometries":len(seen_fields),"unique_complete_scene_geometries":len(seen_scenes),
            "overlaps":overlaps,"excluded_from_identity":["scene_id","seed","render_seed","family","property_assignment","source_raster","candidate_order"],
            "scope":"Exact stored continuous geometry, with anonymous patch/path order. No claim of equivalence under arbitrary geometric symmetries."}
    (root/"CANONICAL_SPLIT_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    if overlaps:
        raise ValueError("Duplicate field/scene geometry found")
    print(result["status"],result["unique_field_geometries"],result["unique_complete_scene_geometries"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
