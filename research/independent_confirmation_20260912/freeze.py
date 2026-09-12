"""Record source/protocol identity before any study dataset is generated."""
import argparse
from datetime import datetime, timezone
import subprocess
from common import HERE, ROOT, sha, write_json, protocol, reference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=__import__("pathlib").Path)
    args = parser.parse_args()
    reference.verify_reference(protocol())
    if (args.root / "datasets").exists() or (args.root / "FREEZE.json").exists():
        raise FileExistsError("Existing freeze or dataset; create a new version instead")
    source = list(HERE.rglob("*.py")) + list(HERE.rglob("*.json")) + list(HERE.rglob("*.sbatch"))
    source += [HERE.parent / "composition_audit_20260912/run_audit.py"]
    source += list((HERE.parent / "quest_reference").rglob("*.py"))
    source += list((HERE.parent / "quest_reference").rglob("*.json"))
    source += list((HERE.parent / "quest_reference").rglob("*.pt"))
    record = {"experiment_id": protocol()["experiment_id"],
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source)) if not p.name.startswith("._")},
              "dataset_generated": False, "test_predictions_accessed": False,
              "hostname": subprocess.check_output(["hostname"], text=True).strip(),
              "scope": "Prospective procedural component confirmation; not independent administration or a C3 formal gate."}
    write_json(args.root / "FREEZE.json", record)
    print(record["frozen_at_utc"], sha(args.root / "FREEZE.json"), flush=True)
