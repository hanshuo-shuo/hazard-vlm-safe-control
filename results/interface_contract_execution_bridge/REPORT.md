# Interface-contract execution bridge

- Provider calls: 0
- PointHazard: repository fixed CEM-MPC
- Safety-Gym: deterministic headless six-value dynamics contract
- Unknown action: no-op task failure
- Ambiguous-mapping executed STC range: 0.300

```json
{
  "point_hazard": {
    "action_authoritative": {
      "n": 15,
      "success": 1.0,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 1.0
    },
    "contract_aware_semantic": {
      "n": 15,
      "success": 1.0,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 1.0
    },
    "applicable_means_constraint_applies": {
      "n": 15,
      "success": 1.0,
      "semantic_violation": 0.2,
      "collision": 0.0,
      "STC": 0.8
    },
    "applicable_means_terrain_compatible": {
      "n": 15,
      "success": 1.0,
      "semantic_violation": 0.4,
      "collision": 0.0,
      "STC": 0.6
    },
    "conservative_fusion": {
      "n": 15,
      "success": 1.0,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 1.0
    }
  },
  "safety_gym_goal": {
    "action_authoritative": {
      "n": 15,
      "success": 0.6,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 0.6
    },
    "contract_aware_semantic": {
      "n": 15,
      "success": 0.6,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 0.6
    },
    "applicable_means_constraint_applies": {
      "n": 15,
      "success": 0.6,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 0.6
    },
    "applicable_means_terrain_compatible": {
      "n": 15,
      "success": 0.6,
      "semantic_violation": 0.2,
      "collision": 0.0,
      "STC": 0.4
    },
    "conservative_fusion": {
      "n": 15,
      "success": 0.6,
      "semantic_violation": 0.0,
      "collision": 0.0,
      "STC": 0.6
    }
  }
}
```
