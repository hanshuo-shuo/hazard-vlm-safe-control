"""Final integrity inventory; no repeated statistical estimation."""
import argparse
import json
from pathlib import Path
from decompose import HERE, sha, now, write_json, verify_manifest, manifest

def main(root):
    stages=["discovery","report","verification","confirmation","confirmation_verification",
            "publication","publication_v2","publication_v3","final_publication"]
    counts={stage:verify_manifest(root/stage) for stage in stages}
    final=json.loads((root/"confirmation_verification/VERIFICATION.json").read_text())
    assert final["status"]=="PASS"
    addendum=json.loads((root/"DIRECTION_ADDENDUM_FREEZE.json").read_text())
    h=json.loads((root/"confirmation/HYPOTHESES.json").read_text())
    assert addendum["at_utc"]<h["at_utc"]
    old_inputs=json.loads((root/"discovery/INPUT_HASHES.json").read_text())
    for path,digest in old_inputs.items():assert sha(path)==digest
    out=root/"delivery";out.mkdir(exist_ok=False)
    write_json(out/"DELIVERY.json",{"status":"PASS","at_utc":now(),
        "stage_manifest_entries":counts,"all_stage_manifest_entries":sum(counts.values()),
        "original_discovery_inputs_unchanged":len(old_inputs),
        "scientific_decision":final["final_decision"],"new_confirmation_count":2000,
        "new_training_runs":0,"new_risk_certificates":0,"rellis_reopened":False,
        "confirmation_consumed":True,"current_figures":["publication_v3/task_scene.png","publication_v3/oracle_discovery.png","final_publication/oracle_confirmation.png"],
        "source_files":{p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        "publication_source_matches":all(sha(HERE/name)==json.loads((root/stage/"PROVENANCE.json").read_text())["source_sha256" if stage=="final_publication" else "script_sha256"] for name,stage in [("publication_v3.py","publication_v3"),("final_publication.py","final_publication")])})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
