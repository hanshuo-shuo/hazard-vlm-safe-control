#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


EP_RE = re.compile(r"ep(\d+)\.jsonl$")


def load_episode(path: Path) -> dict:
    episode = None
    end = None
    last_step = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") == "episode_meta":
                episode = row.get("episode")
            elif row.get("type") == "episode_end":
                end = row
            elif "t" in row:
                last_step = row

    if end is None:
        raise ValueError(f"Missing episode_end in {path}")

    match = EP_RE.search(path.name)
    file_episode = int(match.group(1)) if match else None
    episode = episode if episode is not None else file_episode

    return {
        "episode": episode,
        "source_episode": episode,
        "file": path.name,
        "source_dir": str(path.parent),
        "success": bool(end.get("success", False)),
        "hazard_hit": bool(end.get("hazard_hit", False)),
        "timeout": not bool(end.get("success", False)) and not bool(end.get("hazard_hit", False)),
        "return": float(end.get("return", math.nan)),
        "steps": int(end.get("steps", 0)),
        "min_clearance": float(end.get("min_clearance", math.nan)),
        "final_dist": float(end.get("dist", math.nan)),
        "vlm_calls": int(end.get("vlm_calls", 0)),
        "parse_ok": int(end.get("parse_ok", 0)),
        "parse_total": int(end.get("parse_total", 0)),
        "end_physics_train_steps": int(last_step.get("physics_train_steps", 0)) if last_step else 0,
    }


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def bucket_rows(rows, bucket_size):
    buckets = []
    max_ep = max(row["episode"] for row in rows)
    for start in range(0, max_ep + 1, bucket_size):
        end = min(start + bucket_size - 1, max_ep)
        members = [row for row in rows if start <= row["episode"] <= end]
        if not members:
            continue
        n = len(members)
        successes = sum(row["success"] for row in members)
        hazards = sum(row["hazard_hit"] for row in members)
        timeouts = sum(row["timeout"] for row in members)
        buckets.append(
            {
                "episode_start": start,
                "episode_end": end,
                "n": n,
                "successes": successes,
                "success_rate": successes / n,
                "hazard_hits": hazards,
                "hazard_hit_rate": hazards / n,
                "timeouts": timeouts,
                "timeout_rate": timeouts / n,
                "mean_return": mean([row["return"] for row in members]),
                "mean_steps": mean([row["steps"] for row in members]),
                "mean_min_clearance": mean([row["min_clearance"] for row in members]),
                "mean_final_dist": mean([row["final_dist"] for row in members]),
                "end_physics_train_steps": max(row["end_physics_train_steps"] for row in members),
            }
        )
    return buckets


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_buckets(buckets: list[dict], episode_rows: list[dict], out_path: Path, title: str):
    labels = [f'{b["episode_start"]}-{b["episode_end"]}' for b in buckets]
    x = list(range(len(buckets)))
    success_pct = [b["success_rate"] * 100 for b in buckets]
    hazard_pct = [b["hazard_hit_rate"] * 100 for b in buckets]
    overall = 100 * sum(row["success"] for row in episode_rows) / len(episode_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [2.2, 1]},
        constrained_layout=True,
    )

    bars = ax1.bar(x, success_pct, color="#2f6f73", width=0.66, label="Success rate")
    ax1.plot(x, hazard_pct, color="#b64a3f", marker="o", linewidth=2.0, label="Hazard hit rate")
    ax1.axhline(overall, color="#333333", linestyle="--", linewidth=1.2, label=f"Overall success {overall:.1f}%")
    ax1.set_title(title, fontsize=15, weight="bold")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right")
    for bar, pct in zip(bars, success_pct):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            min(pct + 2, 98),
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    mean_steps = [b["mean_steps"] for b in buckets]
    mean_clearance = [b["mean_min_clearance"] for b in buckets]
    ax2.plot(x, mean_steps, color="#6c5a8c", marker="s", linewidth=2.0, label="Mean steps")
    ax2.set_ylabel("Mean steps")
    ax2.set_xlabel("Episode window")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2b = ax2.twinx()
    ax2b.plot(x, mean_clearance, color="#c68632", marker="^", linewidth=2.0, label="Mean min clearance")
    ax2b.set_ylabel("Mean min clearance")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="qwen32btrain")
    parser.add_argument("--log-dirs", nargs="+", default=None)
    parser.add_argument("--bucket-size", type=int, default=100)
    parser.add_argument("--out-prefix", default="qwen32btrain_success_by_100")
    parser.add_argument("--title", default="Training Success Rate per 100 Episodes")
    args = parser.parse_args()

    log_dirs = [Path(path) for path in args.log_dirs] if args.log_dirs else [Path(args.log_dir)]
    episode_rows = []
    offset = 0
    for log_dir in log_dirs:
        paths = sorted(log_dir.glob("ep*.jsonl"), key=lambda p: int(EP_RE.search(p.name).group(1)))
        dir_rows = [load_episode(path) for path in paths]
        dir_rows = sorted(dir_rows, key=lambda row: row["source_episode"])
        for idx, row in enumerate(dir_rows):
            row["episode"] = offset + idx
        episode_rows.extend(dir_rows)
        offset += len(dir_rows)
    buckets = bucket_rows(episode_rows, args.bucket_size)

    write_csv(Path(f"{args.out_prefix}_episodes.csv"), episode_rows)
    write_csv(Path(f"{args.out_prefix}_buckets.csv"), buckets)
    plot_buckets(buckets, episode_rows, Path(f"{args.out_prefix}.png"), args.title)

    print(f"episodes={len(episode_rows)}")
    print(f"overall_success_rate={sum(row['success'] for row in episode_rows) / len(episode_rows):.4f}")
    print(f"overall_hazard_hit_rate={sum(row['hazard_hit'] for row in episode_rows) / len(episode_rows):.4f}")
    print(f"overall_timeout_rate={sum(row['timeout'] for row in episode_rows) / len(episode_rows):.4f}")
    for bucket in buckets:
        print(
            "bucket "
            f"{bucket['episode_start']:03d}-{bucket['episode_end']:03d}: "
            f"success={bucket['successes']}/{bucket['n']} ({bucket['success_rate']:.2%}), "
            f"hazard={bucket['hazard_hits']}/{bucket['n']} ({bucket['hazard_hit_rate']:.2%}), "
            f"timeout={bucket['timeouts']}/{bucket['n']} ({bucket['timeout_rate']:.2%}), "
            f"mean_steps={bucket['mean_steps']:.1f}, "
            f"mean_min_clearance={bucket['mean_min_clearance']:.3f}"
        )


if __name__ == "__main__":
    main()
