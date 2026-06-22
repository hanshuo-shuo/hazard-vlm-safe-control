"""
PIVOT-style visual prompting for VLM pilot in PointHazardEnv.

Instead of asking the VLM to regress continuous [fx, fy] values,
we render numbered candidate-direction arrows on the environment image
and ask the VLM to *select* the best one (classification, not regression).

Reference:
    PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    n_directions: int = 8,
    n_magnitudes: int = 2,
    magnitudes: list[float] | None = None,
) -> list[np.ndarray]:
    """Generate uniformly-spaced candidate actions in polar coords.

    Returns a list of [fx, fy] arrays, total = n_directions * n_magnitudes.
    Ordering: all magnitudes for direction 0, then direction 1, etc.
    """
    if magnitudes is None:
        if n_magnitudes == 1:
            magnitudes = [1.0]
        elif n_magnitudes == 2:
            magnitudes = [0.5, 1.0]
        else:
            magnitudes = np.linspace(0.3, 1.0, n_magnitudes).tolist()

    candidates: list[np.ndarray] = []
    for i in range(n_directions):
        theta = 2.0 * math.pi * i / n_directions
        for mag in magnitudes:
            fx = mag * math.cos(theta)
            fy = mag * math.sin(theta)
            candidates.append(np.array([fx, fy], dtype=np.float32))
    return candidates


# ---------------------------------------------------------------------------
# Image annotation — draw numbered arrows on rendered frame
# ---------------------------------------------------------------------------

_ARROW_COLOR = (255, 165, 0)          # orange
_ARROW_HEAD_COLOR = (255, 140, 0)     # darker orange for head
_LABEL_BG = (50, 50, 50)             # dark circle behind number
_LABEL_FG = (255, 255, 255)          # white number text


def _draw_arrowhead(draw: ImageDraw.ImageDraw, tip_xy, angle_rad, size=8):
    """Draw a small triangular arrowhead at *tip_xy* pointing along *angle_rad*.

    angle_rad is in *pixel* coordinates (y-down), so atan2(-(dy_pixel), dx_pixel).
    """
    left_angle = angle_rad + math.pi + math.pi / 6
    right_angle = angle_rad + math.pi - math.pi / 6
    pts = [
        tip_xy,
        (tip_xy[0] + size * math.cos(left_angle),
         tip_xy[1] - size * math.sin(left_angle)),
        (tip_xy[0] + size * math.cos(right_angle),
         tip_xy[1] - size * math.sin(right_angle)),
    ]
    draw.polygon(pts, fill=_ARROW_HEAD_COLOR)


def annotate_candidates(
    base_image: Image.Image,
    candidates: list[np.ndarray],
    renderer,                       # HazardRenderer — need world_to_pixel
    agent_world_xy: np.ndarray,
    arrow_length_world: float = 0.8,
) -> Image.Image:
    """Draw numbered orange arrows on *base_image* for each candidate.

    Args:
        base_image: rendered env frame (PIL Image, RGB).
        candidates: list of [fx, fy] candidate actions.
        renderer: HazardRenderer instance (for world_to_pixel).
        agent_world_xy: agent position in world coordinates.
        arrow_length_world: max arrow length in world units (for magnitude=1).

    Returns:
        Annotated PIL Image (copy — original is unchanged).
    """
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    ax, ay = float(agent_world_xy[0]), float(agent_world_xy[1])
    apx, apy = renderer.world_to_pixel(ax, ay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 15)
        except (OSError, IOError):
            font = ImageFont.load_default()

    label_radius = 11  # radius of the dark circle behind each number

    for idx, cand in enumerate(candidates):
        fx, fy = float(cand[0]), float(cand[1])
        mag = math.sqrt(fx * fx + fy * fy)
        if mag < 1e-6:
            continue

        # Arrow endpoint in world coords
        ex = ax + (fx / mag) * arrow_length_world * mag
        ey = ay + (fy / mag) * arrow_length_world * mag
        epx, epy = renderer.world_to_pixel(ex, ey)

        # Draw shaft
        draw.line([(apx, apy), (epx, epy)], fill=_ARROW_COLOR, width=3)

        # Draw arrowhead (angle in pixel coords — y is flipped)
        angle_px = math.atan2(-(epy - apy), epx - apx)
        _draw_arrowhead(draw, (epx, epy), angle_px, size=10)

        # Number label at arrowhead tip (offset outward)
        label = str(idx + 1)  # 1-indexed
        offset_x = int(14 * math.cos(angle_px))
        offset_y = int(-14 * math.sin(angle_px))
        lx = epx + offset_x
        ly = epy + offset_y

        # Clamp label position so it stays inside the image
        margin = label_radius + 2
        lx = max(margin, min(img_w - margin, lx))
        ly = max(margin, min(img_h - margin, ly))

        # Dark circle background
        draw.ellipse([lx - label_radius, ly - label_radius,
                      lx + label_radius, ly + label_radius], fill=_LABEL_BG)
        # Center the text
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((lx - tw // 2, ly - th // 2 - 1), label,
                  fill=_LABEL_FG, font=font)

    return img


# ---------------------------------------------------------------------------
# PIVOT selection prompt
# ---------------------------------------------------------------------------

def build_pivot_prompt(n_candidates: int) -> str:
    return f"""You see a 2D arena from above with hazards.

Visual elements:
- Gray bordered square = arena boundary (bouncing wall)
- Red filled circles = HAZARDS (lava) — touching any of them kills you instantly
- Blue dot = YOU (the agent you control)
- Green circle with "G" = GOAL (your destination)
- Blue line = your past trajectory
- Orange numbered arrows (1-{n_candidates}) = candidate force directions you can choose

Each orange arrow shows a possible force direction and strength.
Longer arrows = stronger force. The number at the tip identifies each arrow.

YOUR TASK: Choose the BEST numbered arrow to move toward the green goal "G"
while AVOIDING all red hazard circles. Stay far from hazards.
If hazards block the direct path, choose an arrow that goes AROUND them.
Prefer a safe detour over a risky shortcut.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


# ---------------------------------------------------------------------------
# Parse VLM selection
# ---------------------------------------------------------------------------

def _parse_pivot_selection(
    text: str, n_candidates: int,
) -> tuple[int | None, str | None, bool]:
    """Extract {"choice": N, "reason": "..."} from VLM output.

    Returns (choice_0indexed, reason, parsed_ok).
    """
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
        return None, None, False
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None, None, False

    choice = obj.get("choice") or obj.get("selection") or obj.get("number")
    reason = obj.get("reason") or obj.get("explanation")

    if choice is None:
        return None, reason, False
    try:
        choice = int(choice)
    except (ValueError, TypeError):
        return None, reason, False

    if choice < 1 or choice > n_candidates:
        return None, reason, False

    return choice - 1, reason, True  # convert to 0-indexed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class _VlmRawResult:
    """Raw VLM output before mapping to candidate action."""
    choice_idx: int | None      # 0-indexed, or None if parse failed
    reason: str | None
    raw_text: str
    parsed_ok: bool
    mean_logprob: float
    confidence: float
    uncertainty: float


@dataclass
class PivotResult:
    """Final result after mapping VLM selection to candidate action."""
    action: np.ndarray         # selected [fx, fy]
    confidence: float
    uncertainty: float
    choice_idx: int            # 0-indexed selected candidate (-1 if failed)
    reason: str | None
    raw_text: str
    parsed_ok: bool
    mean_logprob: float
    n_candidates: int


# ---------------------------------------------------------------------------
# VLM call (raw — returns selection index, not action)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _call_vlm_select(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    *,
    n_candidates: int,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    enable_thinking: bool = False,
) -> _VlmRawResult:
    """Single VLM call — returns raw selection (no action mapping)."""
    from transformers import GenerationConfig

    messages = []
    if not enable_thinking:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "/no_think"}],
        })
    messages.append({
        "role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]
    })

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    start_len = int(inputs["input_ids"].shape[1])

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(1e-6, temperature) if temperature > 0 else 1.0,
        top_p=top_p if temperature > 0 else 1.0,
        top_k=top_k if temperature > 0 else 0,
    )

    generated = model.generate(
        **inputs,
        generation_config=gen_config,
        return_dict_in_generate=True,
        output_scores=True,
    )

    gen_ids = generated.sequences[0][start_len:]
    out_text = processor.decode(gen_ids, skip_special_tokens=True)

    # Parse selection
    choice_idx, reason, ok = _parse_pivot_selection(out_text, n_candidates)

    # Token-level confidence
    token_logprobs = []
    scores = list(generated.scores or [])
    n = min(len(scores), int(gen_ids.numel()))
    for i in range(n):
        logits_i = scores[i][0]
        tok_id = int(gen_ids[i].item())
        lp = float(torch.log_softmax(logits_i, dim=-1)[tok_id].item())
        token_logprobs.append(lp)

    if token_logprobs:
        mean_logprob = float(np.mean(token_logprobs))
        confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))
    else:
        mean_logprob = float("nan")
        confidence = 0.5
    uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))

    return _VlmRawResult(
        choice_idx=choice_idx,
        reason=reason,
        raw_text=out_text,
        parsed_ok=ok,
        mean_logprob=mean_logprob,
        confidence=confidence,
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Top-level PIVOT action selection
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pivot_choose_action(
    model,
    processor,
    base_image: Image.Image,
    renderer,                     # HazardRenderer
    agent_world_xy: np.ndarray,
    *,
    n_directions: int = 8,
    n_magnitudes: int = 2,
    magnitudes: list[float] | None = None,
    arrow_length_world: float = 0.8,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    enable_thinking: bool = False,
    retries: int = 1,
) -> PivotResult:
    """Full PIVOT pipeline: generate candidates -> annotate -> VLM select -> map to action.

    Args:
        model, processor: loaded VLM model and processor.
        base_image: rendered env frame (PIL).
        renderer: HazardRenderer for coordinate conversion.
        agent_world_xy: agent [x, y] in world coordinates.
        n_directions: number of direction bins (evenly spaced around 360 deg).
        n_magnitudes: number of magnitude levels per direction.
        magnitudes: explicit magnitude levels (overrides n_magnitudes).
        arrow_length_world: visual arrow length in world units for mag=1.
        retries: number of VLM retries on parse failure.

    Returns:
        PivotResult with selected action, confidence, and metadata.
    """
    # 1. Generate candidates
    candidates = generate_candidates(n_directions, n_magnitudes, magnitudes)
    n_cand = len(candidates)

    # 2. Annotate image
    annotated = annotate_candidates(
        base_image, candidates, renderer, agent_world_xy, arrow_length_world)

    # 3. Build prompt
    prompt_text = build_pivot_prompt(n_cand)

    # 4. Call VLM with retries
    best_raw: _VlmRawResult | None = None
    for attempt in range(retries + 1):
        p_text = prompt_text
        if attempt > 0:
            p_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"

        raw = _call_vlm_select(
            model, processor, annotated, p_text,
            n_candidates=n_cand,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )
        if best_raw is None:
            best_raw = raw
        if raw.parsed_ok:
            best_raw = raw
            break

    assert best_raw is not None

    # 5. Map selection to action
    if best_raw.parsed_ok and best_raw.choice_idx is not None:
        action = candidates[best_raw.choice_idx].copy()
    else:
        action = np.zeros(2, dtype=np.float32)

    return PivotResult(
        action=np.clip(action, -1.0, 1.0),
        confidence=best_raw.confidence,
        uncertainty=best_raw.uncertainty,
        choice_idx=best_raw.choice_idx if best_raw.choice_idx is not None else -1,
        reason=best_raw.reason,
        raw_text=best_raw.raw_text,
        parsed_ok=best_raw.parsed_ok,
        mean_logprob=best_raw.mean_logprob,
        n_candidates=n_cand,
    )
