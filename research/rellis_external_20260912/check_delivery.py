"""Final document links and frozen artifact integrity, on Quest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root.resolve()
    checks = []
    for folder in ["geometry_audit", "publication"]:
        for name, expected in json.loads((root/folder/"MANIFEST.json").read_text()).items():
            path = (root/folder/name).resolve()
            assert sha(path) == expected, str(path)
            checks.append(str(path.relative_to(root)))
    result = json.loads((root/"artifacts/QUALIFICATION_VERIFICATION.json").read_text())
    assert result["status"] == "PASS"
    assert result["candidate_counts_checked"] == 96
    geometry = json.loads((root/"geometry_audit/GEOMETRY_AUDIT.json").read_text())
    assert geometry["source_sha256"] == sha(root/"research/rellis_external_20260912/geometry_audit.py")
    assert geometry["protocol_sha256"] == sha(root/"research/rellis_external_20260912/qualification_protocol.json")
    documents = ["paper/visual_risk_diagnosis/manuscript.md",
                 "paper/visual_risk_diagnosis/CLAIMS_AND_EVIDENCE.md",
                 "docs/RELLIS_EXTERNAL_PROGRESS_2026-09-12.md",
                 "research/rellis_external_20260912/README.md"]
    links = []
    for name in documents:
        path = root/name
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", path.read_text()):
            if target.startswith(("http:","https:","#")):
                continue
            linked = (path.parent/target.split("#")[0]).resolve()
            assert linked.is_file(), (name,target)
            links.append({"document":name,"target":target})
    out = {"status":"PASS", "at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
           "artifact_hashes_checked":len(checks),"local_document_links_checked":len(links),
           "documents":{name:sha(root/name) for name in documents},
           "sources":{str(f.relative_to(root)):sha(f) for f in sorted((root/"research/rellis_external_20260912").glob("*.py"))},
           "qualification":"STOP_CURRENT_GEOMETRY_SUPPORT",
           "external_model_training_performed":False,"external_final_test_performed":False,
           "note":"Delivery integrity check, not a new scientific experiment or physical calibration validation."}
    (root/"artifacts/DELIVERY_CHECK.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
