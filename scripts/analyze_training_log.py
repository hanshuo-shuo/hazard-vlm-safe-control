#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics as st
from pathlib import Path


EP_RE = re.compile(r"ep(\d+)\.jsonl$")


def avg(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.mean(values) if values else math.nan


def load(log_dir: Path):
    episodes = []
    steps = []

    paths = sorted(log_dir.glob("ep*.jsonl"), key=lambda p: int(EP_RE.search(p.name).group(1)))
    for path in paths:
        ep = int(EP_RE.search(path.name).group(1))
        end = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "episode_end":
                    end = row
                elif "t" in row:
                    steps.append(
                        {
                            "ep": ep,
                            "bucket": ep // 100,
                            "choice": row.get("choice"),
                            "teacher_choice": row.get("teacher_choice"),
                            "conf": row.get("vlm_confidence"),
                            "physics_ready": row.get("physics_ready"),
                            "physics_loss": row.get("physics_loss"),
                            "lora_loss": row.get("lora_loss"),
                            "reason": row.get("vlm_reason", ""),
                            "safe_count": sum(1 for x in row.get("candidate_safe", []) if x),
                            "hazard_step": row.get("hazard_hit"),
                            "goal_step": row.get("goal_success"),
                        }
                    )
        if end is None:
            raise ValueError(f"Missing episode_end in {path}")
        episodes.append({"ep": ep, "bucket": ep // 100, **end})
    return episodes, steps


def rate(values):
    values = list(values)
    return sum(bool(v) for v in values) / len(values) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    args = parser.parse_args()

    episodes, steps = load(Path(args.log_dir))
    buckets = sorted(set(row["bucket"] for row in episodes))

    print(
        "bucket,success,hazard,timeout,mean_steps,mean_min_clearance,"
        "teacher_agree,mean_conf,physics_ready_pct,online_reason_pct,"
        "lora_seen_pct,mean_lora_loss,all_unsafe_step_pct"
    )
    for bucket in buckets:
        erows = [row for row in episodes if row["bucket"] == bucket]
        srows = [row for row in steps if row["bucket"] == bucket]
        agree_rows = [
            row
            for row in srows
            if row["choice"] is not None
            and row["teacher_choice"] is not None
            and row["choice"] > 0
            and row["teacher_choice"] > 0
        ]
        lora_losses = [row["lora_loss"] for row in srows if row["lora_loss"] is not None]
        print(
            f"{bucket * 100:03d}-{bucket * 100 + 99:03d},"
            f"{rate(row['success'] for row in erows):.3f},"
            f"{rate(row['hazard_hit'] for row in erows):.3f},"
            f"{rate((not row['success'] and not row['hazard_hit']) for row in erows):.3f},"
            f"{avg(row['steps'] for row in erows):.2f},"
            f"{avg(row['min_clearance'] for row in erows):.3f},"
            f"{rate(row['choice'] == row['teacher_choice'] for row in agree_rows):.3f},"
            f"{avg(row['conf'] for row in srows):.3f},"
            f"{rate(row['physics_ready'] for row in srows):.3f},"
            f"{rate('Online learned physics' in row['reason'] for row in srows):.3f},"
            f"{len(lora_losses) / len(srows) if srows else math.nan:.3f},"
            f"{avg(lora_losses):.3f},"
            f"{rate(row['safe_count'] == 0 for row in srows):.3f}"
        )

    first_half = episodes[: len(episodes) // 2]
    second_half = episodes[len(episodes) // 2 :]
    print()
    print(f"episodes={len(episodes)}")
    print(f"overall_success={rate(row['success'] for row in episodes):.3f}")
    print(f"first_half_success={rate(row['success'] for row in first_half):.3f}")
    print(f"second_half_success={rate(row['success'] for row in second_half):.3f}")
    print(f"overall_hazard={rate(row['hazard_hit'] for row in episodes):.3f}")
    print(f"success_mean_steps={avg(row['steps'] for row in episodes if row['success']):.2f}")
    print(f"failure_mean_steps={avg(row['steps'] for row in episodes if not row['success']):.2f}")
    print(f"success_mean_min_clearance={avg(row['min_clearance'] for row in episodes if row['success']):.3f}")
    print(f"failure_mean_min_clearance={avg(row['min_clearance'] for row in episodes if not row['success']):.3f}")


if __name__ == "__main__":
    main()
