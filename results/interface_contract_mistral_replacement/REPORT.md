# Interface-contract replacement arm

- Generated: 2026-07-26T02:46:47.492675+00:00
- Invalid arm retained: `openai/gpt-5-mini`
- Replacement: `mistralai/mistral-small-3.2-24b-instruct`
- Same prompts, images, and seeds: yes
- Status: exploratory five-seed evidence only

```json
{
  "go_no_go": {
    "minimum_contract_action_consistency_below_0_8": false,
    "marker_or_order_consistency_below_0_8": false,
    "planner_mapping_STC_range_at_least_0_2": true,
    "model_rank_reversal_observed": false,
    "contract_effect_replicates_in_two_environments": false,
    "decision": "INSUFFICIENT_FOR_ICLR_MAINLINE"
  },
  "paired_consistency": {
    "field_order": 1.0,
    "structured_vs_free_text": 1.0,
    "constraint_label_polarity": 1.0,
    "compatibility_label_polarity": 1.0,
    "constraint_vs_compatibility_wording": 1.0,
    "explicit_vs_ambiguous": 1.0
  },
  "candidate_robustness": {
    "candidate_baseline": {
      "n": 10,
      "parse_rate": 1.0,
      "selected_safe_rate": 1.0,
      "physical_choice_matches_baseline": 1.0
    },
    "marker_id_permutation": {
      "n": 10,
      "parse_rate": 1.0,
      "selected_safe_rate": 1.0,
      "physical_choice_matches_baseline": 0.9
    },
    "candidate_order_permutation": {
      "n": 10,
      "parse_rate": 1.0,
      "selected_safe_rate": 1.0,
      "physical_choice_matches_baseline": 0.9
    }
  }
}
```
