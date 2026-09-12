#!/usr/bin/env python3
"""Verify recovered historical arrays and execute the byte-preserved comparator."""
from pathlib import Path
import json
import run_audit as audit
from autoresearch.exp01b_r2.compare_runs import compare_results

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "results/c3_quest_teacher_20260911/old_quest_runs"
inputs = {}
for path in sorted(source.glob("*/RESULTS.json")):
    result = json.loads(path.read_text())
    metrics = path.with_name("PER_SCENE_METRICS.npz")
    if result.get("status") != "COMPLETE" or not metrics.exists():
        continue
    actual = audit.sha(metrics)
    assert actual == result["per_scene_metrics_sha256"]
    inputs[path.parent.name] = {"results_sha256": audit.sha(path), "per_scene_metrics_sha256": actual}
baseline_path = source / "baseline-confirm-v2/RESULTS.json"
baseline = json.loads(baseline_path.read_text())
comparisons = []
for name in ("ar03-calibrated-polarity-confirm", "ar05-runtime-margin-confirm", "ar10-cosine-schedule-confirm"):
    path = source / name / "RESULTS.json"
    comparisons.append(compare_results(baseline, json.loads(path.read_text()),
                                        baseline_path=baseline_path, candidate_path=path))
output = {"reference_comparator_unmodified": True, "scope": "development loop KEEP/DISCARD, not publication approval",
          "verified_historical_array_count": len(inputs), "inputs": inputs, "comparisons": comparisons}
(Path(__file__).parent / "HISTORICAL_COMPARISONS.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"verified_historical_array_count":len(inputs), "decisions":[(r["candidate_run_id"],r["status"],r["reasons"]) for r in comparisons]},indent=2))
