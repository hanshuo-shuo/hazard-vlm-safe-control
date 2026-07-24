# Core information audit pilot

- Generated: 2026-07-24T06:12:10.682546+00:00
- Seeds: [0, 1, 2, 3, 4]
- Models: ['google/gemini-2.5-flash-lite', 'qwen/qwen3-vl-30b-a3b-instruct']
- Status: pilot / go-no-go evidence, not a formal result

## Summary

```json
{
  "P0": {
    "n": 10,
    "recognition_correct_rate": 0.7,
    "unsafe_set_exact_rate": 0.1,
    "selected_safe_rate": 0.5
  },
  "P1": {
    "n": 10,
    "recognition_correct_rate": 0.6,
    "unsafe_set_exact_rate": 0.4,
    "selected_safe_rate": 0.6
  },
  "P2": {
    "n": 10,
    "recognition_correct_rate": 0.5,
    "unsafe_set_exact_rate": 0.2,
    "selected_safe_rate": 0.4
  },
  "noisy_geometry": {
    "n": 10,
    "recognition_correct_rate": 0.7,
    "unsafe_set_exact_rate": 0.4,
    "selected_safe_rate": 0.6
  },
  "permuted_geometry": {
    "n": 10,
    "recognition_correct_rate": 0.6,
    "unsafe_set_exact_rate": 0.2,
    "selected_safe_rate": 0.4
  },
  "same_length_irrelevant": {
    "n": 10,
    "recognition_correct_rate": 0.5,
    "unsafe_set_exact_rate": 0.2,
    "selected_safe_rate": 0.3
  },
  "wrong_class": {
    "n": 10,
    "recognition_correct_rate": 0.6,
    "unsafe_set_exact_rate": 0.2,
    "selected_safe_rate": 0.3
  },
  "rank_and_correctness_analysis": {
    "model_selected_safe_rates": {
      "P0": {
        "google/gemini-2.5-flash-lite": 0.2,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.8
      },
      "P1": {
        "google/gemini-2.5-flash-lite": 0.4,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.8
      },
      "P2": {
        "google/gemini-2.5-flash-lite": 0.0,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.8
      },
      "same_length_irrelevant": {
        "google/gemini-2.5-flash-lite": 0.0,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.6
      },
      "wrong_class": {
        "google/gemini-2.5-flash-lite": 0.2,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.4
      },
      "permuted_geometry": {
        "google/gemini-2.5-flash-lite": 0.2,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.6
      },
      "noisy_geometry": {
        "google/gemini-2.5-flash-lite": 0.4,
        "qwen/qwen3-vl-30b-a3b-instruct": 0.8
      }
    },
    "model_rankings": {
      "P0": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "P1": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "P2": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "same_length_irrelevant": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "wrong_class": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "permuted_geometry": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ],
      "noisy_geometry": [
        "qwen/qwen3-vl-30b-a3b-instruct",
        "google/gemini-2.5-flash-lite"
      ]
    },
    "primary_P0_P1_P2_rank_change": false,
    "correctness_interaction": {
      "p_safe_given_recognition_correct": 0.7142857142857143,
      "p_safe_given_recognition_incorrect": 0.03571428571428571
    }
  }
}
```

## Artifacts

- `results.json`: row-level results and request hashes
- `summary.png`: compact visual summary
- `example.png`: first visual input/overlay
- `audit.gif`: all matched seed inputs/overlays
