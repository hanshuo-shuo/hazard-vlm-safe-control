# Marker/interface robustness

- Generated: 2026-07-24T06:12:11.479319+00:00
- Seeds: [0, 1, 2, 3, 4]
- Models: ['google/gemini-2.5-flash-lite', 'qwen/qwen3-vl-30b-a3b-instruct']
- Status: pilot / go-no-go evidence, not a formal result

## Summary

```json
{
  "baseline_8": {
    "n": 10,
    "selected_safe_rate": 0.5,
    "physical_choice_matches_baseline_rate": 1.0
  },
  "candidate_order_permutation": {
    "n": 10,
    "selected_safe_rate": 0.5,
    "physical_choice_matches_baseline_rate": 0.3
  },
  "count_16": {
    "n": 10,
    "selected_safe_rate": 0.6,
    "physical_choice_matches_baseline_rate": 0.2
  },
  "count_4": {
    "n": 10,
    "selected_safe_rate": 0.8,
    "physical_choice_matches_baseline_rate": 0.0
  },
  "direct_coordinates": {
    "n": 10,
    "selected_safe_rate": 0.5,
    "physical_choice_matches_baseline_rate": 0.4
  },
  "marker_id_permutation": {
    "n": 10,
    "selected_safe_rate": 0.8,
    "physical_choice_matches_baseline_rate": 0.2
  },
  "unmarked_image": {
    "n": 10,
    "selected_safe_rate": 0.4,
    "physical_choice_matches_baseline_rate": 0.1
  },
  "go_no_go": {
    "marker_match_rate": 0.2,
    "candidate_order_match_rate": 0.3,
    "marker_dependency_detected": true,
    "ICLR_mainline": "TERMINATE",
    "threshold": 0.8,
    "order_reference": "direct_coordinates with identical coordinate records in canonical order"
  }
}
```

## Artifacts

- `results.json`: row-level results and request hashes
- `summary.png`: compact visual summary
- `example.png`: first visual input/overlay
- `audit.gif`: all matched seed inputs/overlays
