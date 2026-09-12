"""Tie-aware decisions and counts without treating cards/variants as scenes."""
from __future__ import annotations

import numpy as np

CARDS = [0, 2, 3]


def decisions(prediction, atol=1e-9):
    tied = np.isclose(prediction, prediction.min(axis=2, keepdims=True), atol=atol, rtol=0)
    return tied / tied.sum(axis=2, keepdims=True)


def pair_validity(truth, gap):
    costs = np.sort(truth, axis=2)
    unique = costs[:, :, 1] - costs[:, :, 0] >= gap
    optima = truth.argmin(axis=2)
    valid = unique[:, 0] & unique[:, 1] & (optima[:, 0] != optima[:, 1])
    return valid, optima


def per_scene(truth, prediction, p):
    prob = decisions(prediction, p["predicted_tie_atol"])
    selected = (prob * truth).sum(axis=2)
    regret = (selected - truth.min(axis=2))[:, :, CARDS].mean((1, 2))
    eligible = truth[..., CARDS] > p["false_safe_threshold"]
    false = eligible & (prediction[..., CARDS] <= p["false_safe_threshold"])
    valid, optimal = pair_validity(truth, p["changed_optimum_minimum_gap"])
    correct = np.take_along_axis(prob, optimal[:, :, None, :], axis=2).squeeze(2)
    joint = correct[:, 0] * correct[:, 1] * valid
    return {"regret": regret, "false_safe_count": false.sum((1, 2, 3)),
            "false_safe_eligible": eligible.sum((1, 2, 3)),
            "paired_correct_sum": joint[:, CARDS].sum(1), "paired_eligible": valid[:, CARDS].sum(1),
            "mae": np.abs(prediction[..., CARDS] - truth[..., CARDS]).mean((1, 2, 3)),
            "original_regret": (selected[:, 0] - truth[:, 0].min(axis=1))[:, CARDS].mean(1)}


def field_counts(prediction, target):
    pred, true = prediction >= 0.5, target >= 0.5
    return {"intersection": (pred & true).sum((-2, -1)), "union": (pred | true).sum((-2, -1))}


def summarize(rows):
    eligible = int(rows["false_safe_eligible"].sum())
    paired = int(rows["paired_eligible"].sum())
    return {"regret": float(rows["regret"].mean()), "original_regret": float(rows["original_regret"].mean()),
            "mae": float(rows["mae"].mean()), "false_safe_count": int(rows["false_safe_count"].sum()),
            "false_safe_eligible": eligible,
            "false_safe_micro": float(rows["false_safe_count"].sum() / eligible) if eligible else None,
            "paired_eligible": paired,
            "paired_correct": float(rows["paired_correct_sum"].sum() / paired) if paired else None}


def calibrate(truth, prediction, alpha):
    scores = np.maximum(truth[..., CARDS] - prediction[..., CARDS], 0).max((1, 2, 3))
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    q = float(np.sort(scores)[rank - 1]) if rank <= len(scores) else 1.0
    return q, scores, rank


def upper_cost(prediction, offset):
    result = np.clip(prediction + offset, 0, 1)
    result[..., 1] = 0
    return result


def coverage_counts(truth, prediction, p):
    prob = decisions(prediction, p["predicted_tie_atol"])
    cost = (prob * truth).sum(axis=2)[..., CARDS]
    unsafe = (prob * (truth > p["false_safe_threshold"])).sum(axis=2)[..., CARDS]
    minimum = prediction.min(axis=2)[..., CARDS]
    rows = []
    for threshold in p["coverage_thresholds"]:
        accept = minimum <= threshold
        rows.append(np.stack((accept.sum((1, 2)), (accept * unsafe).sum((1, 2)),
                              (accept * cost).sum((1, 2))), axis=-1))
    return np.stack(rows, axis=1)
