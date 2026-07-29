#!/usr/bin/env python3
"""Build the simple overview figures used by RESEARCH_STORY.md."""

from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "research_story"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
BOLD_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)


def _font(size: int, *, bold: bool = False):
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing=8) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (xy[0] - width / 2, xy[1] - height / 2 - box[1]),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _arrow(draw: ImageDraw.ImageDraw, start, end, fill="#64748b", width=6) -> None:
    draw.line((start, end), fill=fill, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 18, y - 12), (x - 18, y + 12)], fill=fill)


def _dashed_circle(draw: ImageDraw.ImageDraw, center, radius: int, fill="#16a34a", width=4) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    for start in range(0, 360, 30):
        draw.arc(box, start=start, end=start + 18, fill=fill, width=width)


def _star(draw: ImageDraw.ImageDraw, center, outer: int = 18, inner: int = 8) -> None:
    cx, cy = center
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(points, fill="#facc15", outline="#111827", width=3)


def _box(draw, xy, size, title, body, color) -> None:
    x, y = xy
    width, height = size
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=28,
        fill=color,
        outline="#25324a",
        width=4,
    )
    _centered(draw, (x + width / 2, y + height * 0.35), title, _font(38, bold=True), "#182033")
    _centered(draw, (x + width / 2, y + height * 0.68), body, _font(28), "#38445c", spacing=10)


def build_story_shift() -> None:
    image = Image.new("RGB", (2600, 900), "white")
    draw = ImageDraw.Draw(image)
    _centered(draw, (1300, 85), "How the research question became more precise", _font(52, bold=True), "#20242d")

    y = 220
    box_size = (650, 410)
    boxes = [
        (80, "1. Semantic hazards", "Can a VLM choose a safe\nmarked waypoint?", "#dbeafe"),
        (975, "2. Applicability", "Does the safety rule apply\nto this robot?", "#fef3c7"),
        (1870, "3. Interface safety", "Can equivalent interfaces change\naction, trajectory, and safety?", "#dcfce7"),
    ]
    for x, title, body, color in boxes:
        _box(draw, (x, y), box_size, title, body, color)
    _arrow(draw, (745, 425), (940, 425))
    _arrow(draw, (1640, 425), (1835, 425))
    _centered(draw, (845, 730), "Marker choices were unstable", _font(26), "#9a3412")
    _centered(draw, (1740, 730), "The word 'applicable' was ambiguous", _font(26), "#9a3412")
    image.save(OUTPUT / "story-shift.png")


def build_marker_interface_process() -> None:
    """Show model outputs, not only inputs, for one frozen matched scene."""
    from run_next_five_experiments import (
        geometric_unsafe,
        interface_variant,
        point_cfg,
        scene_request,
    )

    cfg = point_cfg()
    seed = 1
    model = "google/gemini-2.5-flash-lite"
    base, scene, rgb = scene_request(cfg, seed, "P0", "wheeled_non_waterproof")
    panels = (
        ("Baseline", "numbered markers", "baseline_8"),
        ("Marker IDs permuted", "same points, new IDs", "marker_id_permutation"),
        ("Candidate order permuted", "same pixels, list shuffled", "candidate_order_permutation"),
        ("Direct coordinates", "markers + coordinate list", "direct_coordinates"),
        ("Unmarked image", "coordinate list only", "unmarked_image"),
    )
    result_path = ROOT / "results" / "next_five_experiments" / "02_marker_interface_robustness" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = {
        row["variant"]: row
        for row in result["rows"]
        if row["seed"] == seed and row["model_requested"] == model
    }

    _prompt, _png, base_candidates, _mapping = interface_variant(base, rgb, "baseline_8", seed, cfg)
    unsafe_ids = set(geometric_unsafe(base_candidates, scene))
    safe_candidates = {
        int(item["candidate_id"]): item
        for item in base_candidates
        if int(item["candidate_id"]) not in unsafe_ids
    }

    title_height = 105
    header_height = 115
    panel_width = 380
    result_height = 115
    footer_height = 165
    image_y = title_height + header_height
    image = Image.new(
        "RGB",
        (panel_width * len(panels), image_y + 320 + result_height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(image)

    _centered(
        draw,
        (image.width / 2, 48),
        "Same physical scene, different interface, different selected location",
        _font(38, bold=True),
        "#111827",
    )

    selected_ids = []
    for index, (label, subtitle, variant) in enumerate(panels):
        _prompt, png, _candidates, _mapping = interface_variant(base, rgb, variant, seed, cfg)
        panel = Image.open(BytesIO(png)).convert("RGB")
        x = index * panel_width + (panel_width - panel.width) // 2
        image.paste(panel, (x, image_y))
        _centered(
            draw,
            (index * panel_width + panel_width / 2, title_height + 34),
            label,
            _font(25, bold=True),
            "#182033",
        )
        _centered(
            draw,
            (index * panel_width + panel_width / 2, title_height + 76),
            subtitle,
            _font(19),
            "#64748b",
        )
        if index < len(panels) - 1:
            _arrow(
                draw,
                (index * panel_width + panel_width - 24, title_height + 105),
                (index * panel_width + panel_width + 24, title_height + 105),
                width=4,
            )

        for candidate in safe_candidates.values():
            px, py = candidate["pixel_xy"]
            _dashed_circle(draw, (x + px, image_y + py), 17)

        row = rows[variant]
        selected_id = int(row["physical_selected_index"])
        selected_ids.append(selected_id)
        selected = next(
            item for item in base_candidates if int(item["candidate_id"]) == selected_id
        )
        px, py = selected["pixel_xy"]
        _star(draw, (x + px, image_y + py))
        world_x, world_y = selected["world_xy"]
        safety = "SAFE" if row["selected_safe"] else "UNSAFE"
        safety_color = "#15803d" if row["selected_safe"] else "#b91c1c"
        _centered(
            draw,
            (index * panel_width + panel_width / 2, image_y + 356),
            f"Selected physical location: P{selected_id}",
            _font(20, bold=True),
            "#111827",
        )
        _centered(
            draw,
            (index * panel_width + panel_width / 2, image_y + 395),
            f"world ({world_x:.2f}, {world_y:.2f})  |  {safety}",
            _font(18, bold=True),
            safety_color,
        )

    baseline_id = selected_ids[0]
    matches = sum(selected_id == baseline_id for selected_id in selected_ids)
    footer_center = image_y + 320 + result_height + footer_height / 2
    _centered(
        draw,
        (image.width / 2, footer_center - 30),
        f"Displayed-case physical-choice consistency: {matches}/5 conditions selected the baseline location (P{baseline_id})",
        _font(27, bold=True),
        "#9a3412",
    )
    _centered(
        draw,
        (image.width / 2, footer_center + 22),
        "yellow star = model-selected physical point     green dashed ring = evaluator-safe candidate",
        _font(21),
        "#334155",
    )
    _centered(
        draw,
        (image.width / 2, footer_center + 60),
        "Gemini 2.5 Flash Lite · seed 1 · all five responses parsed successfully",
        _font(19),
        "#64748b",
    )
    image.save(OUTPUT / "marker-interface-process.png")


def build_evaluation_chain() -> None:
    image = Image.new("RGB", (2800, 1200), "white")
    draw = ImageDraw.Draw(image)
    _centered(draw, (1400, 90), "The full closed-loop evaluation chain", _font(54, bold=True), "#20242d")

    labels = [
        ("Image +\nrobot card", "#dbeafe"),
        ("Interface\ncontract", "#ede9fe"),
        ("Model\nanswer", "#fef3c7"),
        ("Grounded\nregion", "#fed7aa"),
        ("Fixed\nplanner", "#ccfbf1"),
        ("Native\ntrajectory", "#dcfce7"),
        ("Safety\nscore", "#fee2e2"),
    ]
    x0, y = 65, 330
    width, height, gap = 310, 270, 82
    for index, (label, color) in enumerate(labels):
        x = x0 + index * (width + gap)
        draw.rounded_rectangle(
            (x, y, x + width, y + height),
            radius=24,
            fill=color,
            outline="#25324a",
            width=4,
        )
        _centered(draw, (x + width / 2, y + height / 2), label, _font(32, bold=True), "#182033")
        if index < len(labels) - 1:
            _arrow(draw, (x + width + 10, y + height / 2), (x + width + gap - 12, y + height / 2), width=5)

    _centered(
        draw,
        (1400, 725),
        "The experiment changes only the purple box. Everything else is held fixed.",
        _font(34, bold=True),
        "#4c1d95",
    )
    _centered(
        draw,
        (2180, 875),
        "Evaluator truth is used\nonly after execution",
        _font(29),
        "#991b1b",
    )
    draw.line((2360, 805, 2580, 610), fill="#b91c1c", width=5)
    draw.polygon([(2580, 610), (2548, 618), (2568, 641)], fill="#b91c1c")
    _centered(
        draw,
        (1400, 1080),
        "Main measurement: does the same scene produce the same physical action and trajectory?",
        _font(32),
        "#334155",
    )
    image.save(OUTPUT / "evaluation-chain.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_marker_interface_process()
    build_story_shift()
    build_evaluation_chain()
    print(f"Wrote research-story figures to {OUTPUT}")


if __name__ == "__main__":
    main()
