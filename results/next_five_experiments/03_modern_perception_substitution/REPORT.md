# Modern perception substitution

- Generated: 2026-07-24T06:12:12.750173+00:00
- Seeds: [0, 1, 2, 3, 4]
- Models: ['google/gemini-2.5-flash-lite', 'qwen/qwen3-vl-30b-a3b-instruct']
- Status: pilot / go-no-go evidence, not a formal result

## Summary

```json
{
  "detector_detection_rate": 1.0,
  "detector_mean_center_error_px": 6.328082647267324,
  "coordinate_adapter": "norm1000_top_left -> world_xy",
  "same_planner_executor_closed_loop": {
    "blind": {
      "success_rate": 1.0,
      "semantic_violation_rate": 1.0,
      "safe_task_completion_rate": 0.0
    },
    "detector": {
      "success_rate": 0.4,
      "semantic_violation_rate": 0.0,
      "safe_task_completion_rate": 0.4
    },
    "vlm": {
      "success_rate": 1.0,
      "semantic_violation_rate": 0.8,
      "safe_task_completion_rate": 0.2
    },
    "oracle": {
      "success_rate": 0.8,
      "semantic_violation_rate": 0.0,
      "safe_task_completion_rate": 0.8
    }
  }
}
```

## Artifacts

- `results.json`: row-level results and request hashes
- `summary.png`: compact visual summary
- `example.png`: first visual input/overlay
- `audit.gif`: all matched seed inputs/overlays
