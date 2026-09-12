import argparse
import json
from pathlib import Path
import numpy as np
from common import *

def main(root):
    p = protocol()
    verify_freeze(root, "E1")
    audit = json.loads((root / "e1_data/AUDIT.json").read_text())
    results = []
    for i in range(5):
        run = root / "e1_runs" / f"seed_{i}"
        verify_manifest(run)
        results.append(json.loads((run / "RESULTS.json").read_text()))
    reference_ok = all(v["reference"]["pass"] for v in audit.values())
    feasible = all(results[0]["splits"][s]["oracle"]["deployed"]["oracle_coverage"] >= .5 for s in p["e1"]["splits"])
    simple = all(r["splits"][s]["restored_zero"]["reference"]["raw"]["conditional_unsafe"] is not None and
                 r["splits"][s]["restored_zero"]["reference"]["raw"]["conditional_unsafe"] <= .05 and
                 r["splits"][s]["restored_zero"]["reference"]["raw"]["eta"] >= .5
                 for r in results for s in ["random_iid", "random_appearance"])
    out = root / "e1_report"
    out.mkdir(exist_ok=False)
    summary = {"status": "COMPLETE", "at_utc": now(), "evidence": "Consumed-data mechanism diagnostic; no training",
               "reference_precision_pass": reference_ok, "oracle_feasibility_pass": feasible,
               "simple_postprocessing_suffices_diagnostically": simple,
               "proceed_to_e2": reference_ok and feasible,
               "measurement_warning": any(v["low16_vs_reference_flip"] > .02 for v in audit.values()),
               "reference_audit": audit, "per_seed": results}
    write_json(out / "SUMMARY.json", summary)
    write_json(out / "MANIFEST.json", manifest(out))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["per_seed", "reference_audit"]}, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
