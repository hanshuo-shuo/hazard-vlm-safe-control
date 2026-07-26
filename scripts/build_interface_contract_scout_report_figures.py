#!/usr/bin/env python3
"""Build publication-style PNG figures for the frozen 120-call scout report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "interface_contract_scout_native_analysis"
PAID_DIR = ROOT / "results" / "interface_contract_scout_paid"
DRY_RUN_DIR = ROOT / "results" / "interface_contract_provider_free_dry_run"
OUTPUT_DIR = ROOT / "docs" / "assets" / "interface_contract_scout"

BLUE = "#2878B5"
ORANGE = "#F39C35"
GREEN = "#2A9D8F"
RED = "#D9534F"
PURPLE = "#7A5195"
GRAY = "#707070"
LIGHT_GRAY = "#D9D9D9"
DARK = "#202124"
WHITE = "#FFFFFF"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    font_value: ImageFont.ImageFont,
    *,
    fill: str = DARK,
    anchor: str = "mm",
) -> None:
    draw.multiline_text(xy, value, font=font_value, fill=fill, anchor=anchor, align="center")


def base_canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    text_center(draw, (width / 2, 52), title, font(34, bold=True))
    return image, draw


def save(image: Image.Image, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / name, optimize=True)


def panel_title(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    left, top, right, _bottom = box
    text_center(draw, ((left + right) / 2, top + 20), title, font(23, bold=True), anchor="ma")


def draw_single_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    *,
    y_max: float,
    gate: float | None = None,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 75, top + 70, right - 25, bottom - 95
    for tick in np.linspace(0, y_max, 6):
        y = plot_bottom - (tick / y_max) * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#E7E7E7", width=2)
        draw.text((plot_left - 12, y), f"{tick:.1f}", font=font(15), fill=GRAY, anchor="rm")
    if gate is not None:
        y = plot_bottom - (gate / y_max) * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=GRAY, width=2)
        draw.text((plot_right, y - 8), f"{gate:.2f} gate", font=font(14), fill=GRAY, anchor="rb")
    slot = (plot_right - plot_left) / len(labels)
    bar_width = slot * 0.62
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        center = plot_left + slot * (index + 0.5)
        y = plot_bottom - (value / y_max) * (plot_bottom - plot_top)
        draw.rounded_rectangle(
            (center - bar_width / 2, y, center + bar_width / 2, plot_bottom),
            radius=5,
            fill=color,
        )
        draw.text((center, y - 12), f"{value:.2f}", font=font(16, bold=True), fill=DARK, anchor="mb")
        text_center(draw, (center, plot_bottom + 38), label, font(15), anchor="ma")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=2)


def draw_grouped_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    y_max: float,
    value_labels: bool = False,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 75, top + 92, right - 25, bottom - 82
    for tick in np.linspace(0, y_max, 6):
        y = plot_bottom - (tick / y_max) * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#E7E7E7", width=2)
        draw.text((plot_left - 12, y), f"{tick:.1f}", font=font(14), fill=GRAY, anchor="rm")
    legend_x = plot_left
    for name, _values, color in series:
        draw.rectangle((legend_x, top + 48, legend_x + 18, top + 62), fill=color)
        draw.text((legend_x + 25, top + 55), name, font=font(14), fill=DARK, anchor="lm")
        legend_x += 32 + draw.textlength(name, font=font(14))
    slot = (plot_right - plot_left) / len(labels)
    group_width = slot * 0.78
    bar_width = group_width / len(series)
    for label_index, label in enumerate(labels):
        group_left = plot_left + slot * label_index + (slot - group_width) / 2
        for series_index, (_name, values, color) in enumerate(series):
            value = float(values[label_index])
            x1 = group_left + bar_width * series_index + 2
            x2 = group_left + bar_width * (series_index + 1) - 2
            y = plot_bottom - (value / y_max) * (plot_bottom - plot_top)
            draw.rectangle((x1, y, x2, plot_bottom), fill=color)
            if value_labels:
                draw.text(((x1 + x2) / 2, y - 7), f"{value:.2f}", font=font(12), fill=DARK, anchor="mb")
        text_center(draw, (plot_left + slot * (label_index + 0.5), plot_bottom + 28), label, font(16), anchor="ma")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=2)


def environment_overview() -> None:
    paid_rows = {
        int(path.stem): load_json(path)
        for path in (PAID_DIR / "rows").glob("*.json")
    }
    examples = [
        (0, "PointHazard native", "Top-down custom environment · fixed CEM-MPC"),
        (30, "Safety-Gym native", "MuJoCo Point agent · heading/forward control"),
    ]
    canvas, draw = base_canvas(1800, 900, "Two native closed-loop environments")
    for index, (row_index, title, subtitle) in enumerate(examples):
        row = paid_rows[row_index]
        path = DRY_RUN_DIR / "inputs" / "images" / f"{row['image_sha256']}.png"
        source = Image.open(path).convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
        x = 170 + index * 820
        y = 120
        canvas.paste(source, (x, y))
        draw.rectangle((x, y, x + 640, y + 640), outline=LIGHT_GRAY, width=4)
        text_center(draw, (x + 320, y + 675), title, font(25, bold=True))
        text_center(draw, (x + 320, y + 715), subtitle, font(18), fill=GRAY)
    save(canvas, "environment-overview.png")


def consistency_results(analysis: dict[str, Any]) -> None:
    canvas, draw = base_canvas(2200, 920, "From clean parsing to divergent native behavior")
    left_box = (40, 95, 1080, 880)
    right_box = (1120, 95, 2160, 880)
    panel_title(draw, left_box, "Global consistency cascade")
    panel_title(draw, right_box, "Model heterogeneity")
    metrics = analysis["all_call"]
    keys = [
        "parse_consistency",
        "canonical_semantic_consistency",
        "grounding_consistency",
        "planner_action_IEC",
        "physical_action_IEC",
        "trajectory_IEC",
    ]
    draw_single_bar_chart(
        draw,
        left_box,
        ["Parse", "Semantic", "Grounding", "Planner\naction", "Native\naction", "Trajectory"],
        [metrics[key]["consistency"] for key in keys],
        [BLUE, BLUE, ORANGE, GREEN, RED, PURPLE],
        y_max=1.1,
        gate=0.9,
    )
    model_ids = ["mistral-small-3.2-24b", "qwen3.5-plus-02-15"]
    selected = [
        ("Semantic", "canonical_semantic_consistency", BLUE),
        ("Grounding", "grounding_consistency", ORANGE),
        ("Planner", "planner_action_IEC", GREEN),
        ("Native", "physical_action_IEC", RED),
        ("Trajectory", "trajectory_IEC", PURPLE),
    ]
    series = [
        (
            label,
            [analysis["by_model"][model]["all_call"][key]["consistency"] for model in model_ids],
            color,
        )
        for label, key, color in selected
    ]
    draw_grouped_bar_chart(draw, right_box, ["Mistral", "Qwen"], series, y_max=1.1, value_labels=True)
    save(canvas, "consistency-results.png")


def world_mapper(
    box: tuple[int, int, int, int], points: np.ndarray
) -> tuple[callable, float]:
    left, top, right, bottom = box
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 0.5) * 1.25
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min, x_max = center_x - span / 2, center_x + span / 2
    y_min, y_max = center_y - span / 2, center_y + span / 2
    scale = min((right - left) / span, (bottom - top) / span)

    def mapper(point: Sequence[float]) -> tuple[float, float]:
        x, y = point
        return left + (x - x_min) * scale, bottom - (y - y_min) * scale

    return mapper, scale


def trajectory_propagation(primary: list[dict[str, Any]], blocks: dict[str, Any]) -> None:
    canvas, draw = base_canvas(1900, 1680, "Equivalent language, different native trajectories")
    by_index = {int(row["call_index"]): row for row in primary}
    panels = [
        (0, 4, "Mistral · PointHazard"),
        (30, 34, "Mistral · Safety-Gym"),
        (60, 64, "Qwen · PointHazard"),
        (90, 94, "Qwen · Safety-Gym"),
    ]
    panel_boxes = [
        (60, 115, 920, 825),
        (980, 115, 1840, 825),
        (60, 890, 920, 1600),
        (980, 890, 1840, 1600),
    ]
    for (anchor_index, mate_index, title), panel in zip(panels, panel_boxes):
        anchor, mate = by_index[anchor_index], by_index[mate_index]
        block = blocks[anchor["block_id"]]
        left, top, right, bottom = panel
        draw.rounded_rectangle(panel, radius=14, outline=LIGHT_GRAY, width=3)
        text_center(draw, ((left + right) / 2, top + 34), title, font(22, bold=True))
        plot_box = (left + 70, top + 105, right - 45, bottom - 65)
        trajectories = [
            np.asarray([state["agent_center"] for state in row["trajectory"]], dtype=float)
            for row in (anchor, mate)
        ]
        start = np.asarray(block["start_xy"], dtype=float)
        goal = np.asarray(block["goal_xy"], dtype=float)
        center = np.asarray(anchor["evaluator_geometry"]["center_xy"], dtype=float)
        points = np.vstack([*trajectories, start[None, :], goal[None, :], center[None, :]])
        mapper, scale = world_mapper(plot_box, points)
        for fraction in np.linspace(0, 1, 6):
            x = plot_box[0] + fraction * (plot_box[2] - plot_box[0])
            y = plot_box[1] + fraction * (plot_box[3] - plot_box[1])
            draw.line((x, plot_box[1], x, plot_box[3]), fill="#EEEEEE", width=2)
            draw.line((plot_box[0], y, plot_box[2], y), fill="#EEEEEE", width=2)
        terrain_xy = mapper(center)
        radius_px = float(anchor["evaluator_geometry"]["radius"]) * scale
        overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.ellipse(
            (
                terrain_xy[0] - radius_px,
                terrain_xy[1] - radius_px,
                terrain_xy[0] + radius_px,
                terrain_xy[1] + radius_px,
            ),
            fill=(217, 83, 79, 50),
            outline=(217, 83, 79, 190),
            width=3,
        )
        canvas.paste(overlay, (0, 0), overlay)
        draw = ImageDraw.Draw(canvas)
        for row, trajectory, label, color in (
            (anchor, trajectories[0], "constraint", BLUE),
            (mate, trajectories[1], "compatibility", ORANGE),
        ):
            path = [mapper(point) for point in trajectory]
            draw.line(path, fill=color, width=5, joint="curve")
            legend = f"{label}: {row['physical_action']} · STC={int(bool(row['STC']))}"
            legend_y = top + 70 if label == "constraint" else top + 94
            draw.line((left + 78, legend_y, left + 115, legend_y), fill=color, width=6)
            draw.text((left + 125, legend_y), legend, font=font(15), fill=DARK, anchor="lm")
        start_xy, goal_xy = mapper(start), mapper(goal)
        draw.ellipse((start_xy[0] - 7, start_xy[1] - 7, start_xy[0] + 7, start_xy[1] + 7), fill=DARK)
        draw.regular_polygon((goal_xy[0], goal_xy[1], 12), n_sides=5, rotation=-18, fill=GREEN)
        draw.text((plot_box[0], plot_box[3] + 24), "native world coordinates", font=font(14), fill=GRAY)
    draw.text((80, 1640), "Red disk is evaluator truth shown post hoc; it was never passed to the controller.", font=font(17), fill=GRAY)
    save(canvas, "trajectory-propagation.png")


def environment_results(analysis: dict[str, Any]) -> None:
    canvas, draw = base_canvas(2200, 920, "Native execution succeeds, but safety propagation differs")
    left_box = (40, 95, 1080, 880)
    right_box = (1120, 95, 2160, 880)
    panel_title(draw, left_box, "Closed-loop outcome rates")
    panel_title(draw, right_box, "Image-to-world calibration error")
    env_ids = ["point_hazard_native", "safety_gym_goal_native"]
    performance = [
        ("Move", "nonstationary_rate", BLUE),
        ("Success", "task_success_rate", GREEN),
        ("STC", "STC_rate", PURPLE),
        ("Collision", "collision_rate", RED),
        ("Semantic", "semantic_violation_rate", ORANGE),
    ]
    draw_grouped_bar_chart(
        draw,
        left_box,
        ["PointHazard", "Safety-Gym"],
        [
            (label, [analysis["by_environment"][env][key] for env in env_ids], color)
            for label, key, color in performance
        ],
        y_max=1.1,
        value_labels=True,
    )
    calibration_series = [
        (
            "Point mean",
            [
                analysis["by_environment"][env_ids[0]]["calibration"]["center_error_world_mean"],
                analysis["by_environment"][env_ids[0]]["calibration"]["radius_error_world_mean"],
            ],
            BLUE,
        ),
        (
            "Point max",
            [
                analysis["by_environment"][env_ids[0]]["calibration"]["center_error_world_max"],
                analysis["by_environment"][env_ids[0]]["calibration"]["radius_error_world_max"],
            ],
            "#91B9D5",
        ),
        (
            "Safety mean",
            [
                analysis["by_environment"][env_ids[1]]["calibration"]["center_error_world_mean"],
                analysis["by_environment"][env_ids[1]]["calibration"]["radius_error_world_mean"],
            ],
            ORANGE,
        ),
        (
            "Safety max",
            [
                analysis["by_environment"][env_ids[1]]["calibration"]["center_error_world_max"],
                analysis["by_environment"][env_ids[1]]["calibration"]["radius_error_world_max"],
            ],
            "#F7C987",
        ),
    ]
    draw_grouped_bar_chart(draw, right_box, ["Center", "Radius"], calibration_series, y_max=5.0, value_labels=True)
    save(canvas, "environment-results.png")


def five_stage_results(analysis: dict[str, Any]) -> None:
    canvas, draw = base_canvas(2100, 900, "Five-stage audit localizes the failure")
    left_box = (40, 95, 1030, 860)
    right_box = (1070, 95, 2060, 860)
    panel_title(draw, left_box, "Stage accuracy")
    panel_title(draw, right_box, "Earliest equivalent-pair difference")
    scores = analysis["five_stage"]["score"]
    draw_single_bar_chart(
        draw,
        left_box,
        ["Recognition", "Applicability", "Grounding", "Action", "Outcome"],
        [
            scores["recognition_accuracy"],
            scores["applicability_accuracy"],
            scores["grounding_accuracy"],
            scores["action_proposal_accuracy"],
            scores["enforcement_outcome_accuracy"],
        ],
        [BLUE, BLUE, ORANGE, GREEN, PURPLE],
        y_max=1.1,
    )
    attribution = analysis["stage_difference_attribution"]["counts"]
    labels = ["Consistent", "Normalization", "Grounding", "Planner"]
    values = [attribution["consistent"], attribution["normalization"], attribution["grounding"], attribution["planner"]]
    colors = [LIGHT_GRAY, BLUE, ORANGE, GREEN]
    left, top, right, bottom = right_box
    plot_left, plot_top, plot_right, plot_bottom = left + 170, top + 100, right - 45, bottom - 50
    slot = (plot_bottom - plot_top) / len(labels)
    max_value = max(values) + 1
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = plot_top + slot * (index + 0.5)
        width = (value / max_value) * (plot_right - plot_left)
        draw.rounded_rectangle((plot_left, y - 28, plot_left + width, y + 28), radius=5, fill=color)
        draw.text((plot_left - 15, y), label, font=font(18), fill=DARK, anchor="rm")
        draw.text((plot_left + width + 12, y), str(value), font=font(18, bold=True), fill=DARK, anchor="lm")
    draw.text((plot_left, plot_bottom + 5), "matched equivalent pairs (n=20)", font=font(15), fill=GRAY)
    save(canvas, "five-stage-results.png")


def provider_operations() -> None:
    rows = [load_json(path) for path in sorted((PAID_DIR / "rows").glob("*.json"))]
    model_ids = ["mistral-small-3.2-24b", "qwen3.5-plus-02-15"]
    names = ["Mistral", "Qwen"]
    costs: list[float] = []
    latencies: list[float] = []
    reasoning: list[int] = []
    for model in model_ids:
        subset = [row for row in rows if row["model_budget_id"] == model]
        costs.append(sum(float((row["provider"].get("usage") or {}).get("cost") or 0.0) for row in subset))
        latencies.append(sum(float(row["provider"].get("latency_seconds") or 0.0) for row in subset) / len(subset))
        reasoning.append(
            sum(
                int(
                    (((row["provider"].get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"))
                    or 0
                )
                for row in subset
            )
        )
    canvas, draw = base_canvas(2100, 720, "Qwen dominated operational cost and latency")
    panels = [
        ((40, 95, 680, 680), costs, "Provider cost (USD)", ".4f"),
        ((730, 95, 1370, 680), latencies, "Mean latency (seconds)", ".1f"),
        ((1420, 95, 2060, 680), reasoning, "Billed reasoning tokens", ",.0f"),
    ]
    for box, values, title, fmt in panels:
        panel_title(draw, box, title)
        left, top, right, bottom = box
        plot_left, plot_top, plot_right, plot_bottom = left + 60, top + 95, right - 35, bottom - 75
        positive = [float(value) for value in values if float(value) > 0]
        floor = min(positive) if positive else 1.0
        transformed = [0.0 if value <= 0 else math.log10(float(value) / floor) + 1 for value in values]
        maximum = max(transformed) * 1.15 if transformed else 1.0
        slot = (plot_right - plot_left) / 2
        for index, (name, raw, transformed_value, color) in enumerate(zip(names, values, transformed, (BLUE, ORANGE))):
            center = plot_left + slot * (index + 0.5)
            y = plot_bottom if transformed_value == 0 else plot_bottom - (transformed_value / maximum) * (plot_bottom - plot_top)
            draw.rounded_rectangle((center - 70, y, center + 70, plot_bottom), radius=5, fill=color)
            label = "0" if raw == 0 else format(raw, fmt)
            draw.text((center, y - 12), label, font=font(17, bold=True), fill=DARK, anchor="mb")
            draw.text((center, plot_bottom + 30), name, font=font(18), fill=DARK, anchor="ma")
        draw.text((plot_left, bottom - 22), "log-scaled height; labels show exact values", font=font(13), fill=GRAY)
    save(canvas, "provider-operations.png")


def main() -> int:
    analysis = load_json(ANALYSIS_DIR / "ANALYSIS.json")
    primary = load_json(ANALYSIS_DIR / "PRIMARY_EXECUTIONS.json")
    blocks = {block["block_id"]: block for block in load_json(DRY_RUN_DIR / "BLOCKS.json")}
    environment_overview()
    consistency_results(analysis)
    trajectory_propagation(primary, blocks)
    environment_results(analysis)
    five_stage_results(analysis)
    provider_operations()
    print(json.dumps({"output": str(OUTPUT_DIR), "figures": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
